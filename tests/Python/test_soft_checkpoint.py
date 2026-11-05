"""The per-pair soft pass's checkpointed forward (soft_kernel_impl.inl, "Checkpoints").

Large pairs keep the first forward row of every block of k rows and recompute the block in
the backward.  The recomputation must reproduce the first pass bit for bit (checked by
NWGRAD_SOFT_CKPT_CHECK, which hashes every row both times: test_recompute_is_bit_exact);
against the full tables the results agree to rounding (the block buffer's rows sit at
other addresses, so a vectorized sum may peel differently).  k is forced through the
``soft_ckpt_rows`` gate: 1, 2, 7 and 64 rows, and None (one block of every row: the
checkpointed code with no recomputation).
"""

import os
import subprocess
import sys

import numpy as np
import pytest

import nwgrad
from nwgrad import _tuning

KS = [1, 2, 7, 64, None]
MODES = [(g, m) for g in ("linear", "affine") for m in ("global", "local", "glocal")]
CLASSES = {"double": nwgrad.SeqPairBatchDouble, "float32": nwgrad.SeqPairBatch}
# Against the full tables: rounding of the reductions only (double ~1e-16, float32 ~1e-7).
TOL = {"double": (1e-12, 1e-12), "float32": (1e-5, 1e-5)}


@pytest.fixture(autouse=True)
def _reset():
    _tuning.reset_gates()
    yield
    _tuning.reset_gates()


def rand_seq(rng, alpha, n):
    return "".join(rng.choice(list(alpha), n))


def params_for(rng, alpha):
    na = len(alpha)
    M = rng.normal(-0.3, 1.0, (na, na)) + 2.0 * np.eye(na)
    return nwgrad.AlignParams(nwgrad.SubstMatrix(M, alpha), 2.0, 0.5, 3.0, 1.0)


def pairs(rng, alpha, n, lo=1, hi=90):
    lens = [(0, 0), (0, 5), (5, 0), (1, 1), (1, 9), (9, 1), (64, 3), (65, 70), (128, 40)]
    lens += [(int(rng.integers(lo, hi)), int(rng.integers(lo, hi))) for _ in range(n)]
    A = [rand_seq(rng, alpha, a) for a, _ in lens]
    B = []
    for a, (_, nb) in zip(A, lens):   # related pairs for half of them: a real path
        if a and rng.random() < 0.5:
            b = list(a)
            for i in range(len(b)):
                if rng.random() < 0.2:
                    b[i] = alpha[int(rng.integers(len(alpha)))]
            B.append("".join(b[: max(1, nb)]))
        else:
            B.append(rand_seq(rng, alpha, nb))
    return A, B


def run(cls, A, B, P, gm, mode, rows, T=1.0, guide="eager", band=None, tracks=None):
    with _tuning.override(soft_ckpt_rows=rows):
        kw = {}
        if tracks is not None:
            kw = dict(tracks=tracks[0], gap_tracks=tracks[1])
        b = cls(A, B, P, n_threads=2, gap_model=gm, mode=mode, grad_mode="soft", **kw)
        b.fill = "striped"   # every pair its own per-pair soft pass
        b.soft_temperature = T
        b.soft_guide = guide
        b.score_and_grad()
        if band is not None:
            b.banded_grad(band)
        mats, gaps = b.grads()
        out = [np.asarray(b.scores()), np.asarray(mats), np.asarray(gaps)]
        if guide == "posterior":
            out.append(np.concatenate([np.asarray(b[i].guide_j) for i in range(len(A))]))
        return out


def check(ref, got, prec, exact_guide=True):
    rel, atol = TOL[prec]
    for x, y in zip(ref[:3], got[:3]):
        np.testing.assert_allclose(y, x, rtol=rel, atol=atol)
    if len(ref) > 3 and exact_guide:
        assert np.array_equal(ref[3], got[3])


@pytest.mark.parametrize("prec", list(CLASSES))
@pytest.mark.parametrize("gm,mode", MODES)
@pytest.mark.parametrize("alpha", ["ACGT", nwgrad.PROTEIN.symbols])
def test_checkpointed_matches_full_tables(prec, gm, mode, alpha):
    rng = np.random.default_rng(abs(hash((prec, gm, mode, len(alpha)))) % 2**32)
    P = params_for(rng, alpha)
    A, B = pairs(rng, alpha, 20)
    cls = CLASSES[prec]
    ref = run(cls, A, B, P, gm, mode, 0)
    for k in KS:
        check(ref, run(cls, A, B, P, gm, mode, k), prec)


@pytest.mark.parametrize("prec", list(CLASSES))
@pytest.mark.parametrize("gm,mode", MODES)
def test_checkpointed_temperature_and_posterior_guide(prec, gm, mode):
    rng = np.random.default_rng(5)
    P = params_for(rng, "ACGT")
    A, B = pairs(rng, "ACGT", 12)
    cls = CLASSES[prec]
    for T in (0.5, 2.0):
        ref = run(cls, A, B, P, gm, mode, 0, T=T, guide="posterior")
        for k in KS:
            # The posterior guide is an argmax: a rounding-level tie may flip in float32.
            check(ref, run(cls, A, B, P, gm, mode, k, T=T, guide="posterior"), prec,
                  exact_guide=(prec == "double"))


@pytest.mark.parametrize("prec", list(CLASSES))
@pytest.mark.parametrize("gm,mode", MODES)
@pytest.mark.parametrize("band", [1, 3, 12])
def test_checkpointed_banded(prec, gm, mode, band):
    """GuideBanded soft pairs: per-row spans, zeroed right before each row is computed."""
    rng = np.random.default_rng(17 + band)
    P = params_for(rng, "ACGT")
    A, B = pairs(rng, "ACGT", 12)
    cls = CLASSES[prec]
    ref = run(cls, A, B, P, gm, mode, 0, band=band)
    for k in KS:
        check(ref, run(cls, A, B, P, gm, mode, k, band=band), prec)


@pytest.mark.parametrize("prec", list(CLASSES))
@pytest.mark.parametrize("gm,mode", MODES)
def test_checkpointed_tracked(prec, gm, mode):
    rng = np.random.default_rng(29)
    K, alpha = 3, "ACGT"
    A, B = pairs(rng, alpha, 12)
    mats = rng.normal(0, 1.0, size=(K, 4, 4))
    gaps = rng.uniform(0.3, 1.5, size=(K, 4))
    P = nwgrad.AlignParams.from_arrays(np.ascontiguousarray(mats), np.ascontiguousarray(gaps), alpha)
    tracks = ([list(rng.integers(0, K, size=len(a))) for a in A],
              [list(rng.integers(0, K, size=len(a) + 1)) for a in A])
    cls = CLASSES[prec]
    ref = run(cls, A, B, P, gm, mode, 0, T=0.9, tracks=tracks)
    for k in KS:
        check(ref, run(cls, A, B, P, gm, mode, k, T=0.9, tracks=tracks), prec)


@pytest.mark.parametrize("prec", list(CLASSES))
@pytest.mark.parametrize("gm", ["linear", "affine"])
def test_large_pair_checkpoints_by_default(prec, gm):
    """A pair whose forward tables pass soft_ckpt_min_bytes checkpoints with the automatic
    k; the same pair on the full tables (the switch off) agrees to rounding."""
    rng = np.random.default_rng(3)
    alpha = nwgrad.PROTEIN.symbols
    P = params_for(rng, alpha)
    a = rand_seq(rng, alpha, 420)
    b = list(a)
    for i in range(len(b)):
        if rng.random() < 0.25:
            b[i] = alpha[int(rng.integers(20))]
    A, B = [a], ["".join(b[:400])]
    cls = CLASSES[prec]
    for mode in ("global", "local"):
        auto = run(cls, A, B, P, gm, mode, "default")
        with _tuning.override(soft_ckpt_min_bytes=None):
            full = run(cls, A, B, P, gm, mode, "default")
        check(full, auto, prec)


CHECK_SCRIPT = r"""
import numpy as np, nwgrad
from nwgrad import _tuning
rng = np.random.default_rng(1)
alpha = "ACGT"
M = rng.normal(0, 1, (4, 4)) + 2 * np.eye(4)
P = nwgrad.AlignParams(nwgrad.SubstMatrix(M, alpha), 2.0, 0.5, 3.0, 1.0)
A = ["".join(rng.choice(list(alpha), int(rng.integers(1, 120)))) for _ in range(12)]
B = ["".join(rng.choice(list(alpha), int(rng.integers(1, 120)))) for _ in range(12)]
n = 0
for cls in (nwgrad.SeqPairBatchDouble, nwgrad.SeqPairBatch):
    for gm in ("linear", "affine"):
        for mode in ("global", "local", "glocal"):
            for k in (1, 2, 7, 64):
                with _tuning.override(soft_ckpt_rows=k):
                    b = cls(A, B, P, n_threads=2, gap_model=gm, mode=mode, grad_mode="soft")
                    b.fill = "striped"
                    b.soft_temperature = 0.3   # rescales: the exponents must recompute too
                    b.score_and_grad()
                    b.banded_grad(4)
                    n += 1
print("ok", n)
"""


def test_recompute_is_bit_exact():
    """Every recomputed forward row hashes as it did in the first pass (M, X, Y over its
    span, its exponent and maximum); a mismatch raises inside the pass."""
    env = dict(os.environ, NWGRAD_SOFT_CKPT_CHECK="1")
    r = subprocess.run([sys.executable, "-c", CHECK_SCRIPT], env=env, capture_output=True,
                       text=True, cwd=os.path.dirname(os.path.abspath(__file__)))
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["ok", str(2 * 2 * 3 * 4)]


def test_env_rows_override():
    """NWGRAD_SOFT_CKPT_ROWS forces k while the gate is unset (read once per process)."""
    script = (
        "import nwgrad, numpy as np\n"
        "p = nwgrad.AlignParams(nwgrad.SubstMatrix(np.eye(4), 'ACGT'), 1.0, 0.5, 1.0, 0.5)\n"
        "b = nwgrad.SeqPairBatchDouble(['ACGTACGTAAC'*5], ['ACGTTCGTAC'*5], p, grad_mode='soft')\n"
        "b.fill = 'striped'\n"
        "b.score_and_grad()\n"
        "print(repr(float(b.scores()[0])))\n")
    outs = []
    for k in ("0", "3"):
        env = dict(os.environ, NWGRAD_SOFT_CKPT_ROWS=k, NWGRAD_SOFT_DEBUG="1")
        r = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True,
                           text=True, cwd=os.path.dirname(os.path.abspath(__file__)))
        assert r.returncode == 0, r.stderr
        assert f"ck={k}" in r.stderr
        outs.append(float(r.stdout))
    assert outs[0] == pytest.approx(outs[1], rel=1e-13)
