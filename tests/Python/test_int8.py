"""The int8 score tier (score8_kernel_impl.inl, Aligner::score_i8_) and the int8 names.

The contract: integer score-only runs the int8 kernel first where the parameters fit and
re-runs wider whenever the kernel cannot prove its result, so a saturated value never
leaks -- every score equals the int16 / int32 / double one with the tier on or off.
Checked across gap models, modes, every ISA level (kernel=), parameters at the int8 limits
and problems straddling both overflow tests (a Local score of 127; Global / Glocal's
score > G - 128 and G <= 126).  The tier's counters (_score8_stats) show it really ran and
really overflowed, so neither branch passes vacuously.
"""
import numpy as np
import pytest

import nwgrad
from nwgrad import _tuning
from nwgrad import nwgrad_ext as E

DNA = "ACGT"
MODES = ["global", "local", "glocal"]
SCORE_FN = {("linear", "global"): "nw_score", ("linear", "local"): "sw_score",
            ("linear", "glocal"): "glocal_score", ("affine", "global"): "nw_score_affine",
            ("affine", "local"): "sw_score_affine", ("affine", "glocal"): "glocal_score_affine"}


@pytest.fixture(autouse=True)
def _stats():
    _tuning.reset_gates()
    E._score8_stats(True, True)
    yield
    E._score8_stats(True, False)
    _tuning.reset_gates()


def _params(rng, alpha, style):
    n = len(alpha)
    if style == "small":
        m = np.where(np.eye(n, dtype=bool), rng.integers(1, 6, (n, n)), rng.integers(-5, 2, (n, n)))
        g = rng.integers(0, 6, 4)
    elif style == "large":
        m = np.where(np.eye(n, dtype=bool), rng.integers(10, 40, (n, n)), rng.integers(-40, 4, (n, n)))
        g = rng.integers(0, 20, 4)
    else:   # the int8 limits
        m = np.where(np.eye(n, dtype=bool), 127, rng.choice([-127, -1, 0], (n, n)))
        g = rng.choice([0, 1, 63, 64, 127], 4)
        g[1] = min(g[1], 127 - g[0]); g[3] = min(g[3], 127 - g[2])
    return nwgrad.AlignParams(nwgrad.SubstMatrix(m.astype(float), alphabet=alpha), *map(float, g))


def _pair(rng, alpha, la, lb, homology):
    a = "".join(rng.choice(list(alpha), la))
    if homology:
        b = list(a)
        for _ in range(int(rng.integers(0, 4))):
            b[int(rng.integers(0, len(b)))] = str(rng.choice(list(alpha)))
        b = "".join(rng.choice(list(alpha), int(rng.integers(0, 8)))) + "".join(b)
    else:
        b = "".join(rng.choice(list(alpha), lb))
    return a, b


def _all(fn, a, b, p, kernel="auto"):
    """(tier on, tier off) at int16 and int32, int8's name, and the double reference."""
    out = []

    def run(name):   # int16 (and int8, which is int16) may refuse a problem past its bound
        try:
            out.append(getattr(nwgrad, name)(a, b, p, kernel=kernel))
        except OverflowError:
            assert not name.endswith("_int32")
    for gate in (2, 0):
        with _tuning.override(score_int8=gate):
            run(fn + "_int16")
            run(fn + "_int32")
    run(fn + "_int8")
    out.append(getattr(nwgrad, fn + "_double")(a, b, p))
    return out


@pytest.mark.parametrize("gm", ["affine", "linear"])
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("style", ["small", "large", "limits"])
def test_scores_equal_every_wider_type(gm, mode, style):
    rng = np.random.default_rng(hash((gm, mode, style)) % 2**32)
    fn = SCORE_FN[(gm, mode)]
    for it in range(60):
        alpha = DNA if it % 3 else "ACDEFGHIKLMNPQRSTVWY"
        p = _params(rng, alpha, style)
        a, b = _pair(rng, alpha, int(rng.integers(1, 90)), int(rng.integers(1, 90)), it % 2 == 0)
        got = _all(fn, a, b, p)
        assert len(set(got)) == 1, (a, b, got)


@pytest.mark.parametrize("gm", ["affine", "linear"])
@pytest.mark.parametrize("mode", MODES)
def test_every_isa_level(gm, mode):
    rng = np.random.default_rng(7)
    fn = SCORE_FN[(gm, mode)]
    cases = [(_params(rng, DNA, "small"), *_pair(rng, DNA, int(rng.integers(1, 120)),
                                                 int(rng.integers(1, 120)), k % 2 == 0)) for k in range(25)]
    for level in nwgrad.available_isa_levels():
        for p, a, b in cases:
            got = _all(fn, a, b, p, kernel=level)
            assert len(set(got)) == 1, (level, a, b, got)


def test_local_straddles_127():
    # match 1, so the Local score of a planted copy is its length: 120..135 straddles 127
    m = np.where(np.eye(4, dtype=bool), 1.0, -1.0)
    p = nwgrad.AlignParams(nwgrad.SubstMatrix(m, alphabet=DNA), 2.0, 1.0, 2.0, 1.0)
    rng = np.random.default_rng(3)
    E._score8_stats(True)
    for L in range(120, 136):
        a = "".join(rng.choice(list(DNA), L))
        for fn in ("sw_score", "sw_score_affine"):
            got = _all(fn, a, "GG" + a + "TT", p)
            assert got[-1] == L and len(set(got)) == 1, (L, got)
    runs, over = E._score8_stats()
    assert runs > 0 and over > 0 and runs > over   # both branches taken (L < 127 held)


def test_global_straddles_the_gain_test():
    # G = min(m, n) * 2; at L = 63, G = 126 (the precondition's edge), at 64 refused up front.
    # A score <= G - 128 re-runs at int16 (unrelated pairs score low).
    m = np.where(np.eye(4, dtype=bool), 2.0, -2.0)
    p = nwgrad.AlignParams(nwgrad.SubstMatrix(m, alphabet=DNA), 4.0, 2.0, 4.0, 2.0)
    rng = np.random.default_rng(4)
    for L in (10, 40, 62, 63, 64, 65):
        for hom in (True, False):
            a, b = _pair(rng, DNA, L, L, hom)
            for fn in ("nw_score", "nw_score_affine", "glocal_score", "glocal_score_affine"):
                got = _all(fn, a, b, p)
                assert len(set(got)) == 1, (L, fn, got)
    runs, over = E._score8_stats()
    assert runs > over > 0


def test_negative_gaps_and_tracks_skip_the_tier():
    m = np.where(np.eye(4, dtype=bool), 2.0, -1.0)
    p = nwgrad.AlignParams(nwgrad.SubstMatrix(m, alphabet=DNA), -1.0, 1.0, 2.0, 1.0)
    E._score8_stats(True)
    assert len(set(_all("sw_score_affine", "ACGTACGT", "ACGAACGT", p))) == 1
    assert E._score8_stats() == (0, 0)


def test_narrow_off_disables_the_tier():
    import os, subprocess, sys
    code = ("import nwgrad, numpy as np; from nwgrad import nwgrad_ext as E; E._score8_stats(True, True);"
            "p = nwgrad.AlignParams(nwgrad.SubstMatrix(np.where(np.eye(4, dtype=bool), 2.0, -1.0), alphabet='ACGT'), 2.0, 1.0, 2.0, 1.0);"
            "s = nwgrad.sw_score_int32('ACGTAC', 'ACGTTC', p); print(int(s), E._score8_stats()[0])")
    for env, runs in (({"NWGRAD_INT_NARROW": "0"}, 0), ({"NWGRAD_INT8": "0"}, 0), ({}, 1)):
        e = dict(os.environ, **env)
        out = subprocess.run([sys.executable, "-c", code], env=e, capture_output=True, text=True, check=True).stdout.split()
        if nwgrad.get_isa_level() != "scalar_fallback":
            assert int(out[1]) == runs, (env, out)


def test_int8_names_check_the_parameter_range():
    m = np.where(np.eye(4, dtype=bool), 128.0, -1.0)
    p = nwgrad.AlignParams(nwgrad.SubstMatrix(m, alphabet=DNA), 2.0, 1.0, 2.0, 1.0)
    with pytest.raises(ValueError, match="int8"):
        nwgrad.sw_score_int8("ACGT", "ACGT", p)
    with pytest.raises(ValueError, match="int8"):
        nwgrad.SeqPairInt8("ACGT", "ACGT", p)
    with pytest.raises(ValueError, match="int8"):
        nwgrad.SeqPairBatchInt8(["ACGT"], ["ACGT"], p)
    half = nwgrad.AlignParams(nwgrad.SubstMatrix(np.where(np.eye(4, dtype=bool), 2.0, -1.0), alphabet=DNA),
                              0.5, 1.0, 2.0, 1.0)
    with pytest.raises(ValueError, match="int8"):
        nwgrad.nw_score_affine_int8("ACGT", "ACGT", half)
    assert not hasattr(nwgrad, "nw_soft_grad_int8")


def test_int8_pairs_and_batches():
    rng = np.random.default_rng(11)
    p = _params(rng, DNA, "small")
    A = ["".join(rng.choice(list(DNA), int(rng.integers(1, 200)))) for _ in range(40)]
    B = [a[: len(a) // 2] + "".join(rng.choice(list(DNA), 10)) + a[len(a) // 2:] for a in A]
    for mode in MODES:
        for gm in ("affine", "linear"):
            ref = nwgrad.SeqPairBatchInt16(A, B, p, gap_model=gm, mode=mode, grad_mode="none", n_threads=2)
            ref.fill = "striped"
            ref.score_and_grad()
            b8 = nwgrad.SeqPairBatchInt8(A, B, p, gap_model=gm, mode=mode, grad_mode="none", n_threads=2)
            b8.fill = "striped"
            b8.score_and_grad()
            assert list(b8.scores()) == list(ref.scores())
            # paths and gradients run at int16 (the same integers)
            sp = nwgrad.SeqPairInt8(A[0], B[0], p, gap_model=gm, mode=mode, grad_mode="hard")
            s16 = nwgrad.SeqPairInt16(A[0], B[0], p, gap_model=gm, mode=mode, grad_mode="hard")
            assert sp.align() == s16.align() and sp.aligned() == s16.aligned()


def test_default_mode_is_local_with_back_off():
    m = np.where(np.eye(4, dtype=bool), 2.0, -2.0)
    p = nwgrad.AlignParams(nwgrad.SubstMatrix(m, alphabet=DNA), 2.0, 1.0, 2.0, 1.0)
    a = "ACGTTGCA" * 4
    with _tuning.override(score_int8=2):   # a success clears this thread's back-off
        nwgrad.sw_score_int16(a, a, p)
    E._score8_stats(True)
    nwgrad.nw_score_int16(a, a, p); nwgrad.glocal_score_affine_int16(a, a, p)
    assert E._score8_stats()[0] == 0                 # Global / Glocal: only at score_int8=2
    nwgrad.sw_score_int16(a, a, p)
    if nwgrad.get_isa_level() != "scalar_fallback":
        assert E._score8_stats(True) == (1, 0)
        big = "ACGT" * 40                              # Local score 320: overflows every time
        got = [nwgrad.sw_score_int16(big, big, p) for _ in range(20)]
        assert len(set(got)) == 1 and got[0] == nwgrad.sw_score_double(big, big, p)
        runs, over = E._score8_stats()
        assert runs == over and 1 <= runs <= 5       # tried at 0, 2, 6, 14: backing off
