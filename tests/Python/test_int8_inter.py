"""The inter-pair int8 score tier (inter8_kernel_impl.inl, BatchEngine::score_only_).

A score-only integer batch runs its short pairs in saturating int8 lanes and re-runs, at
the batch's type, every lane whose result the kernel cannot prove (Local: 127 reached;
Global / Glocal: score <= G - 128).  Every score must equal the tier-off scores (the int16 /
int32 inter-pair kernel) and the per-pair striped fills, at every ISA level, with problems
on both sides of each overflow test.
"""
import numpy as np
import pytest

import nwgrad
from nwgrad import _tuning

DNA = "ACGT"
CLS = {"int16": nwgrad.SeqPairBatchInt16, "int32": nwgrad.SeqPairBatchInt32}


@pytest.fixture(autouse=True)
def _gates():
    _tuning.reset_gates()
    yield
    _tuning.reset_gates()


def _scores(A, B, p, t, gm, mode, gate, fill="interpair", threads=2):
    _tuning.set_gate("inter_score_int8", gate)
    b = CLS[t](A, B, p, gap_model=gm, mode=mode, grad_mode="none", n_threads=threads)
    b.fill = fill
    b.score_and_grad()
    first = list(b.scores())
    b.score_and_grad()                 # the cached plan
    assert list(b.scores()) == first
    return first


def _check(A, B, p, t, gm, mode):
    on = _scores(A, B, p, t, gm, mode, 2)
    assert on == _scores(A, B, p, t, gm, mode, 0)
    assert on == _scores(A, B, p, t, gm, mode, 0, fill="striped")
    return on


def _matrix(alpha, match, mismatch):
    n = len(alpha)
    return nwgrad.SubstMatrix(np.where(np.eye(n, dtype=bool), float(match), float(mismatch)), alphabet=alpha)


@pytest.mark.parametrize("t", ["int16", "int32"])
@pytest.mark.parametrize("gm", ["affine", "linear"])
@pytest.mark.parametrize("mode", ["global", "local", "glocal"])
def test_mirna_like_every_level(t, gm, mode):
    rng = np.random.default_rng(5)
    go = 0.0 if gm == "linear" else 4.0
    p = nwgrad.AlignParams(_matrix(DNA, 2, -4), go, 2.0, go, 2.0)
    A, B = [], []
    for k in range(70):
        a = "".join(rng.choice(list(DNA), 22 + int(rng.integers(-2, 3))))
        b = list(rng.choice(list(DNA), 50 + int(rng.integers(-3, 4))))
        if k % 2:
            off = int(rng.integers(0, len(b) - len(a)))
            b[off:off + len(a)] = list(a)
        A.append(a); B.append("".join(b))
    prev = nwgrad.get_isa_level()
    for level in nwgrad.available_isa_levels():
        if level == "scalar_fallback":
            continue
        nwgrad.set_isa_level(level)
        try:
            _check(A, B, p, t, gm, mode)
        finally:
            nwgrad.set_isa_level(prev)


@pytest.mark.parametrize("gm", ["affine", "linear"])
def test_local_straddles_127(gm):
    # match 1: a planted copy's Local score is its length, 115..127 around the int8 top
    p = nwgrad.AlignParams(_matrix(DNA, 1, -1), 0.0 if gm == "linear" else 2.0, 1.0, 0.0 if gm == "linear" else 2.0, 1.0)
    rng = np.random.default_rng(3)
    A, B = [], []
    for L in list(range(115, 128)) * 3:
        a = "".join(rng.choice(list(DNA), L))
        A.append(a); B.append("G" + a)
    _tuning.set_gate("inter_score_small_cells", None)
    got = _check(A, B, p, "int16", gm, "local")
    assert got == [float(len(a)) for a in A]


@pytest.mark.parametrize("gm", ["affine", "linear"])
@pytest.mark.parametrize("mode", ["global", "glocal"])
def test_global_straddles_the_gain_test(gm, mode):
    # match 2: G = 2 * min(m, n); unrelated pairs score far below G - 128 and re-run
    go = 0.0 if gm == "linear" else 4.0
    p = nwgrad.AlignParams(_matrix(DNA, 2, -2), go, 2.0, go, 2.0)
    rng = np.random.default_rng(4)
    A, B = [], []
    for k in range(96):
        L = int(rng.choice([10, 40, 62, 63, 64, 65]))
        a = "".join(rng.choice(list(DNA), L))
        b = a if k % 3 == 0 else "".join(rng.choice(list(DNA), L + int(rng.integers(0, 40))))
        A.append(a); B.append(b)
    _tuning.set_gate("inter_score_small_cells", None)
    _check(A, B, p, "int16", gm, mode)
    _check(A, B, p, "int32", gm, mode)


def test_int8_limits_and_small_alphabets():
    rng = np.random.default_rng(9)
    for alpha in ("AC", "ACGTN", "ACDEFGHI"):
        n = len(alpha)
        for style in range(3):
            if style == 0:
                m = np.where(np.eye(n, dtype=bool), 127, rng.choice([-127, -1, 0], (n, n)))
                g = [int(x) for x in rng.choice([0, 1, 63, 64, 127], 4)]
                g[1] = min(g[1], 127 - g[0]); g[3] = min(g[3], 127 - g[2])
            elif style == 1:
                m = np.where(np.eye(n, dtype=bool), rng.integers(1, 6, (n, n)), rng.integers(-5, 2, (n, n)))
                g = [int(x) for x in rng.integers(0, 6, 4)]
            else:
                m = np.where(np.eye(n, dtype=bool), rng.integers(10, 40, (n, n)), rng.integers(-40, 4, (n, n)))
                g = [int(x) for x in rng.integers(0, 20, 4)]
            p = nwgrad.AlignParams(nwgrad.SubstMatrix(m.astype(float), alphabet=alpha), *map(float, g))
            A = ["".join(rng.choice(list(alpha), int(rng.integers(1, 60)))) for _ in range(50)]
            B = [a + "".join(rng.choice(list(alpha), int(rng.integers(0, 30)))) if k % 2 else
                 "".join(rng.choice(list(alpha), int(rng.integers(1, 80)))) for k, a in enumerate(A)]
            _tuning.set_gate("inter_score_small_cells", None)
            for mode in ("global", "local", "glocal"):
                _check(A, B, p, "int16", "affine", mode)
                _check(A, B, p, "int16", "linear", mode)
