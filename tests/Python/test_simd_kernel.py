"""The kernel= selector, from the Python side.

tests/cpp/test_simd_bitexact.cpp already proves the two Viterbi kernels write
bit-identical DP tables.  What it cannot prove is that the kernel= keyword actually
*reaches* them through the bindings — a binding that quietly ignored the argument
would leave every one of those C++ assertions passing and still ship a knob that
does nothing.  So these tests check the wiring, not the arithmetic.

Bit-exactness means equality here is exact.  Nothing is compared with a tolerance,
deliberately: if any of these ever needs one, the central claim in aligner_simd.hpp
is false and the simd kernel is not safe to ship.
"""

import random

import pytest

import nwgrad
from nwgrad.matrices import BLOSUM62

# BLOSUM62 is integer-valued, which is the point: exact ties between competing
# alignments are everywhere, so a kernel that broke ties differently would show up
# immediately in the gradients below.
AA = "ARNDCQEGHILKMFPSTWYV"


def params(gap_open=11.0, gap_extend=1.0):
    return nwgrad.AlignParams(BLOSUM62, gap_open, gap_extend, gap_open, gap_extend)


def seq(rng, length):
    return "".join(rng.choice(AA) for _ in range(length))


def test_simd_isa_is_reported():
    assert nwgrad.simd_isa() in ("baseline", "avx2", "avx512", "neon")


def test_bad_kernel_name_raises():
    # A typo that silently selected the slow path would be undetectable by any test,
    # so it throws rather than falling back.
    p = params()
    for bad in ("SIMD", "sse", "avx2", ""):
        with pytest.raises(Exception, match="kernel"):
            nwgrad.nw_score_affine("ACDE", "ACDE", p, kernel=bad)


@pytest.mark.parametrize("fn", ["nw_score_affine", "sw_score_affine"])
def test_scores_are_exactly_equal(fn):
    rng = random.Random(1234)
    p = params()
    f = getattr(nwgrad, fn)
    for _ in range(60):
        a, b = seq(rng, rng.randint(1, 80)), seq(rng, rng.randint(1, 80))
        assert f(a, b, p, kernel="scalar") == f(a, b, p, kernel="simd")


@pytest.mark.parametrize("fn", ["nw_affine_grad", "sw_affine_grad"])
def test_hard_gradients_are_exactly_equal(fn):
    """The strongest check available from Python.

    A hard gradient depends on *which* optimal path the traceback walked.  With ties
    as common as BLOSUM62 makes them, an equal gradient means the vectorized kernel
    did not merely reach the same score — it reproduced the same tie-breaking, cell
    for cell, which is what the exact-float-equality traceback requires.
    """
    rng = random.Random(99)
    p = params()
    f = getattr(nwgrad, fn)
    for _ in range(40):
        a, b = seq(rng, rng.randint(2, 70)), seq(rng, rng.randint(2, 70))
        s_scalar, g_scalar = f(a, b, p, kernel="scalar")
        s_simd, g_simd = f(a, b, p, kernel="simd")

        assert s_scalar == s_simd
        assert g_scalar.matrix.to_matrix().tobytes() == g_simd.matrix.to_matrix().tobytes()
        assert g_scalar.gap_open_a == g_simd.gap_open_a
        assert g_scalar.gap_extend_a == g_simd.gap_extend_a
        assert g_scalar.gap_open_b == g_simd.gap_open_b
        assert g_scalar.gap_extend_b == g_simd.gap_extend_b


def test_batch_aligner_kernels_agree():
    rng = random.Random(7)
    p = params()
    sa = [seq(rng, 60) for _ in range(200)]
    sb = [seq(rng, 60) for _ in range(200)]

    out = {}
    for kern in ("scalar", "simd"):
        ba = nwgrad.BatchAligner(p, gap_model="affine", mode="global",
                                 grad_mode="hard", n_threads=4, kernel=kern)
        out[kern] = ba.align(sa, sb)

    assert list(out["scalar"].scores) == list(out["simd"].scores)
    assert (out["scalar"].grad.matrix.to_matrix().tobytes()
            == out["simd"].grad.matrix.to_matrix().tobytes())


def test_seq_pair_kernels_agree_including_banded_realign():
    rng = random.Random(2024)
    p = params()
    a, b = seq(rng, 120), seq(rng, 120)

    out = {}
    for kern in ("scalar", "simd"):
        sp = nwgrad.SeqPair(a, b, p, gap_model="affine", mode="global",
                            grad_mode="hard", kernel=kern)
        sp.alloc_dp()
        sp.align_full()
        full = sp.score
        # realign_banded takes the per-row gather path inside the simd kernel rather
        # than the query profile, so it is genuinely different code — not just the
        # same kernel over fewer columns.
        sp.realign_banded(12)
        out[kern] = (full, sp.score, sp.aligned())

    assert out["scalar"] == out["simd"]


def test_linear_kernel_is_a_documented_no_op():
    """The linear gap model has no simd kernel, on purpose.

    Its recurrence collapses to a single carry that is a pure latency chain, and the
    scalar loop already sits on that floor — a vectorized version was written,
    measured slower, and deleted.  kernel="simd" therefore stays *legal* for a linear
    aligner and quietly returns the fastest linear kernel there is.  It must not throw.
    """
    rng = random.Random(5)
    p = params(gap_open=0.0, gap_extend=1.0)
    for _ in range(20):
        a, b = seq(rng, rng.randint(1, 60)), seq(rng, rng.randint(1, 60))
        assert nwgrad.nw_score(a, b, p, kernel="simd") == nwgrad.nw_score(a, b, p, kernel="scalar")
