"""traceback="hirschberg_pmax" — Hirschberg with a CLOSED-FORM prefix-max gap carry.

This is the one mode in the library that can return a *suboptimal* path, so it is held to
a third standard, different from both of the others:

  * "pointers" and "scores" are bit-identical to each other.
  * "hirschberg" is exactly optimal but may return a different optimum (its split cannot
    reproduce the backward-greedy M>X>Y tie-break) — test_hirschberg.py asserts that.
  * "hirschberg_pmax" replaces the serial VY carry
        VY[c] = max(open[c-1], VY[c-1] - ge_a)
    with the algebraically-equal closed form
        VY[c] = ( max over k <= c-1 of ( open[k] + k*ge_a ) ) - (c-1)*ge_a
    which adds the ramp k*ge_a and subtracts it again.  That round-trip rounds, so the
    argmax can move and the returned path can score BELOW the optimum.

Two contracts survive the inexactness, and this file pins both:

  1. SELF-CONSISTENCY.  Every quantity in the closed form depends on the ABSOLUTE column
     index and on nothing else — not on the vector width, not on the segment count — and
     max is exact and associative.  So pmax must be bit-identical across every ISA level
     the CPU offers, exactly as the exact kernels are.  `kernel=` stays a speed knob.
  2. ONE-SIDED OPTIMALITY.  pmax may come back below the optimum; it may never come back
     ABOVE it.  A score higher than the true optimum would mean the reported number is
     not the score of any real path, which is a bug and not a tradeoff.

WHAT THE ERROR ACTUALLY DEPENDS ON (measured 2026-07-28, `tools/char_hb_pmax.py`).  The
intuition that the error scales with the winning gap-run length L is WRONG — a per-run
rebase would give that, but a single global prefix max does not.  The intermediate is
O(k*ge_a) for the absolute column index k, so the error scales with SEQUENCE LENGTH times
gap_extend_a.  Measured against a T=double Pointers oracle, the shortfall attributable to
the ramp (i.e. beyond what exact Hirschberg already gives up) over lengths 1000-12000:

    gap_extend_a = 1.0   float32  0        double  0
    gap_extend_a = 0.1   float32  <= 4.9e-4   double  <= 9.1e-13
    learned-style real-valued matrix, either ge   float32  0    double  0

For scale, float32 Pointers *alone* already deviates from the double oracle by up to
1.7e-1 on the same fixtures.  The ramp is not the dominant error term at either precision;
it is not even close.  The bounds below are those measurements plus headroom, not
tolerances widened until the suite went green.
"""
import numpy as np
import pytest

import nwgrad

AA = "ARNDCQEGHILKMFPSTWYV"

# Bounds from the measurements in the docstring, with an order of magnitude of headroom.
# If a change pushes past these, that is a real regression in the carry and wants looking
# at — not a bigger number here.
RAMP_BOUND_F64 = 1e-11
RAMP_BOUND_F32 = 1e-2


@pytest.fixture(scope="module")
def params():
    """Integral +4/-1: ties are COMMON, so the tie-breaking freedom is exercised hard."""
    m = np.full((20, 20), -1.0)
    np.fill_diagonal(m, 4.0)
    return nwgrad.AlignParams(nwgrad.SubstMatrix(m, alphabet=AA),
                              gap_open_a=11.0, gap_extend_a=1.0,
                              gap_open_b=11.0, gap_extend_b=1.0)


@pytest.fixture(scope="module")
def lossy_params():
    """gap_extend = 0.1 is not representable in binary, so the ramp k*ge_a genuinely
    rounds.  This is the fixture under which pmax deviates at all; with integral gap
    costs the ramp is exact and there is nothing to see."""
    m = np.full((20, 20), -1.0)
    np.fill_diagonal(m, 4.0)
    return nwgrad.AlignParams(nwgrad.SubstMatrix(m, alphabet=AA),
                              gap_open_a=11.0, gap_extend_a=0.1,
                              gap_open_b=11.0, gap_extend_b=0.1)


def _seqs(n, lo, hi, seed):
    rng = np.random.default_rng(seed)
    return ["".join(rng.choice(list(AA), int(l))) for l in rng.integers(lo, hi, n)]


def _pair(a, b, p, tb, cutoff=None, kernel="auto", dtype="double", mode="global"):
    cls = nwgrad.SeqPairDouble if dtype == "double" else nwgrad.SeqPair
    sp = cls(a, b, p, gap_model="affine", mode=mode, grad_mode="hard",
             traceback=tb, kernel=kernel)
    if cutoff is not None and tb.startswith("hirschberg"):
        sp.hb_cutoff = cutoff
    sp.alloc_dp()
    sp.align_full()
    sp.compute_grad()
    return sp


def _dot(g, p):
    """The score implied by a gradient.  The score is linear in every parameter, so this
    must reproduce it exactly for ANY valid subgradient, whichever path it came from."""
    M = np.asarray(p.matrix.to_matrix())
    return (float((g["matrix"] * M).sum())
            + g["gap_open_a"] * p.gap_open_a + g["gap_extend_a"] * p.gap_extend_a
            + g["gap_open_b"] * p.gap_open_b + g["gap_extend_b"] * p.gap_extend_b)


def _rescore(a_al, b_al, params):
    """Score a gapped alignment from scratch — independent of the aligner, which is what
    makes "the path scores what it claims" a real check."""
    M = np.asarray(params.matrix.to_matrix())
    idx = {c: i for i, c in enumerate(AA)}
    s, prev = 0.0, "M"
    for x, y in zip(a_al, b_al):
        if x != "-" and y != "-":
            s += M[idx[x], idx[y]]; prev = "M"
        elif y == "-":
            if prev != "X": s -= params.gap_open_b
            s -= params.gap_extend_b; prev = "X"
        else:
            if prev != "Y": s -= params.gap_open_a
            s -= params.gap_extend_a; prev = "Y"
    return s


# ── guard rails ───────────────────────────────────────────────────────────────
#
# Same reasoning as test_hirschberg.py's, and more pressing here: if pmax silently fell
# back to the exact sweep, every assertion below would still pass and every benchmark of
# this mode would be a benchmark of something else.

def test_selectable_and_reported(params):
    sp = _pair(*_seqs(2, 40, 60, 0), params, "hirschberg_pmax")
    assert sp.traceback == "hirschberg_pmax"


def test_unknown_traceback_still_rejected():
    with pytest.raises(ValueError, match="unknown traceback"):
        nwgrad.SeqPairBatch(traceback="hirschberg_pmaxx")


def test_default_is_split_by_precision(params):
    """`auto` resolves to pmax at float32 and to the exact carry at double.

    The split is the whole safety argument for defaulting to a mode that can return a
    suboptimal path: at float32 the ramp costs ~5e-4 against the ~8e-3 the precision
    already costs by itself, so it is lost in noise the caller has agreed to; at double
    it would be the LARGEST error term in the computation, and double is chosen by people
    who want exactness.  If this test ever has to be relaxed, the argument has changed."""
    a, b = _seqs(2, 600, 700, 1)
    assert _pair(a, b, params, "auto", dtype="float").traceback == "hirschberg_pmax"
    assert _pair(a, b, params, "auto", dtype="double").traceback == "hirschberg"


def test_default_split_does_not_leak_to_other_problem_types(params):
    """Only affine+global+full has a Hirschberg mode at all.  Local has one but keeps
    pointers by choice; linear and banded have none and would throw if `auto` reached
    for pmax there.  float32 must not change any of that."""
    a, b = _seqs(2, 600, 700, 1)
    assert _pair(a, b, params, "auto", dtype="float", mode="local").traceback == "pointers"
    sp = nwgrad.SeqPair(a[0], b[0], params, gap_model="linear", mode="global",
                        grad_mode="hard", traceback="auto")
    assert sp.traceback == "pointers"


def test_float32_default_costs_nothing_beyond_float32(params):
    """The justification for the float32 default, measured rather than asserted.

    On pairs well above hb_cutoff (so the recursion really splits and the pmax carry
    really runs), the three float32 arms must deviate from a double oracle by the SAME
    amount: the ramp adds nothing that float32 was not already paying.  This is the test
    to look at if anyone asks why a mode that can be suboptimal is a default.  It is also
    a dispatch-liveness check — `auto` must be bit-identical to explicit pmax, not to the
    exact sweep, which is what a silent fallback would produce."""
    s = _seqs(12, 1400, 1600, 11)
    pairs = list(zip(s[:6], s[6:]))

    def sc(tb, cls):
        bt = cls(n_threads=2, traceback=tb)
        if tb.startswith("hirschberg"):
            bt.hb_cutoff = 512
        bt.add_many([x for x, _ in pairs], [y for _, y in pairs], params,
                    gap_model="affine", mode="global", grad_mode="hard", kernel="auto")
        bt.score_and_grad()
        return np.array([bt[i].score for i in range(len(pairs))], dtype=float)

    ref = sc("pointers", nwgrad.SeqPairBatchDouble)
    auto = sc("auto", nwgrad.SeqPairBatch)
    pmax = sc("hirschberg_pmax", nwgrad.SeqPairBatch)
    ptr = sc("pointers", nwgrad.SeqPairBatch)

    # auto IS pmax at float32 — not merely close to it.
    assert np.array_equal(auto, pmax)
    # and the ramp costs nothing beyond what plain float32 pointers already costs.
    assert abs(ref - auto).max() <= abs(ref - ptr).max()


def test_linear_gap_model_throws(params):
    a, b = _seqs(2, 40, 60, 2)
    with pytest.raises(RuntimeError, match="affine gap model only"):
        sp = nwgrad.SeqPairDouble(a, b, params, gap_model="linear", mode="global",
                                  grad_mode="hard", traceback="hirschberg_pmax")
        sp.alloc_dp(); sp.align_full()


def test_local_is_supported(lossy_params):
    """Local shares the recursion, so pmax applies there too."""
    a, b = _seqs(2, 700, 800, 3)
    sp = _pair(a, b, lossy_params, "hirschberg_pmax", cutoff=64, mode="local")
    ptr = _pair(a, b, lossy_params, "pointers", mode="local")
    assert sp.score <= ptr.score + RAMP_BOUND_F64


# ── contract 1: bit-identical across every ISA level ──────────────────────────

@pytest.mark.parametrize("dtype", ["double", "float32"])
def test_bit_identical_across_isa_levels(lossy_params, dtype):
    """The closed form depends on the absolute column index, never on the vector width,
    so W=2/4/8 and the scalar walk must land on identical bits.  A cutoff well below the
    sequence length forces the recursion to split, so the sweep genuinely runs."""
    levels = nwgrad.available_isa_levels()
    a, b = _seqs(2, 900, 1000, 4)
    ref = _pair(a, b, lossy_params, "hirschberg_pmax", cutoff=32,
                kernel="scalar_fallback", dtype=dtype)
    ref_g = ref.grad.to_dict()
    for lvl in levels:
        got = _pair(a, b, lossy_params, "hirschberg_pmax", cutoff=32,
                    kernel=lvl, dtype=dtype)
        assert got.score == ref.score, f"{lvl}: score moved"
        assert got.aligned() == ref.aligned(), f"{lvl}: alignment moved"
        g = got.grad.to_dict()
        assert np.array_equal(g["matrix"], ref_g["matrix"]), f"{lvl}: gradient moved"
        for k in ("gap_open_a", "gap_extend_a", "gap_open_b", "gap_extend_b"):
            assert g[k] == ref_g[k], f"{lvl}: {k} moved"


# ── contract 2: one-sided optimality, within the measured bound ───────────────

@pytest.mark.parametrize("fixture,bound", [("params", 1e-12), ("lossy_params", RAMP_BOUND_F64)])
def test_never_scores_above_the_optimum(request, fixture, bound):
    """The direction matters more than the magnitude.  Below the optimum is the documented
    tradeoff; ABOVE it would mean the reported score belongs to no real path."""
    p = request.getfixturevalue(fixture)
    for a, b in zip(_seqs(12, 700, 1400, 5), _seqs(12, 700, 1400, 6)):
        opt = _pair(a, b, p, "pointers").score
        got = _pair(a, b, p, "hirschberg_pmax", cutoff=128).score
        assert got <= opt + 1e-12, "pmax scored ABOVE the true optimum"
        assert opt - got <= bound


def test_ramp_cost_beyond_exact_hirschberg(lossy_params):
    """Isolate the ramp: compare pmax against EXACT Hirschberg, which shares the split and
    the tie-break, so the only difference left is the carry."""
    worst = 0.0
    for a, b in zip(_seqs(10, 1200, 2200, 7), _seqs(10, 1200, 2200, 8)):
        hb = _pair(a, b, lossy_params, "hirschberg", cutoff=128).score
        pm = _pair(a, b, lossy_params, "hirschberg_pmax", cutoff=128).score
        worst = max(worst, hb - pm)
    assert worst <= RAMP_BOUND_F64, f"ramp cost {worst:g} exceeds the measured envelope"


def test_error_does_not_track_gap_run_length(lossy_params):
    """The L*eps model is wrong, and this pins the consequence rather than the model: the
    shortfall stays inside the same bound as the forced gap runs grow by two orders of
    magnitude.  If the error really scaled with run length, the last rows would blow up."""
    rng = np.random.default_rng(12)
    for la, lb in ((1500, 1500), (900, 1500), (300, 1500), (60, 1500)):
        a = "".join(rng.choice(list(AA), la))
        b = "".join(rng.choice(list(AA), lb))
        opt = _pair(a, b, lossy_params, "pointers").score
        got = _pair(a, b, lossy_params, "hirschberg_pmax", cutoff=128).score
        assert opt - got <= RAMP_BOUND_F64, f"|a|={la} |b|={lb}"


# ── the path is real, and the gradient is that path's ─────────────────────────

def test_alignment_reconstructs_inputs(lossy_params):
    for a, b in zip(_seqs(6, 800, 1200, 9), _seqs(6, 800, 1200, 10)):
        sp = _pair(a, b, lossy_params, "hirschberg_pmax", cutoff=64)
        aa, bb = sp.aligned()
        assert len(aa) == len(bb)
        assert aa.replace("-", "") == a
        assert bb.replace("-", "") == b


def test_path_scores_what_it_claims(lossy_params):
    """A suboptimal path is allowed.  A path whose score is not the reported score is not."""
    for a, b in zip(_seqs(6, 800, 1200, 13), _seqs(6, 800, 1200, 14)):
        sp = _pair(a, b, lossy_params, "hirschberg_pmax", cutoff=64)
        assert _rescore(*sp.aligned(), lossy_params) == pytest.approx(sp.score, abs=1e-9)


def test_gradient_is_the_gradient_of_that_path(lossy_params):
    """grad . params == score.  Holds for any valid subgradient, so it survives both the
    tie-breaking freedom and the ramp's suboptimality."""
    for a, b in zip(_seqs(6, 900, 1500, 15), _seqs(6, 900, 1500, 16)):
        sp = _pair(a, b, lossy_params, "hirschberg_pmax", cutoff=64)
        assert _dot(sp.grad.to_dict(), lossy_params) == pytest.approx(sp.score, abs=1e-9)


# ── below the cutoff there is no sweep, so there is no ramp ───────────────────

def test_short_pairs_add_nothing_over_exact_hirschberg(lossy_params):
    """A pair no longer than hb_cutoff never splits, so the SWEEP never runs and the ramp
    cannot bite: pmax must be bit-identical to exact Hirschberg there.  That is what makes
    a large cutoff safe, and it is the property attributable to this variant.

    Deliberately NOT compared to Pointers elementwise.  Exact Hirschberg's own alignment
    already diverges from Pointers on some short pairs under a non-representable
    gap_extend (measured: 3/20 at ge=0.1, 0/20 at ge=1.0) — a pre-existing property of
    that mode, not of this one.  Asserting it here would pin unrelated behaviour to this
    file.  The score is compared, since that does agree."""
    for a, b in zip(_seqs(8, 60, 200, 17), _seqs(8, 60, 200, 18)):
        ptr = _pair(a, b, lossy_params, "pointers")
        hb = _pair(a, b, lossy_params, "hirschberg", cutoff=512)
        pm = _pair(a, b, lossy_params, "hirschberg_pmax", cutoff=512)
        assert pm.score == hb.score
        assert pm.aligned() == hb.aligned()
        assert np.array_equal(pm.grad.to_dict()["matrix"], hb.grad.to_dict()["matrix"])
        assert pm.score == pytest.approx(ptr.score, abs=1e-12)


def test_batch_agrees_with_single_pair(lossy_params):
    a_s, b_s = _seqs(8, 700, 1100, 19), _seqs(8, 700, 1100, 20)
    b = nwgrad.SeqPairBatchDouble(n_threads=2, traceback="hirschberg_pmax")
    b.hb_cutoff = 64
    b.add_many(a_s, b_s, lossy_params, gap_model="affine", mode="global",
               grad_mode="hard", kernel="auto")
    b.score_and_grad()
    for i, (a, bb) in enumerate(zip(a_s, b_s)):
        assert b[i].score == _pair(a, bb, lossy_params, "hirschberg_pmax", cutoff=64).score


def test_degenerate_shapes(lossy_params):
    """Empty and one-residue inputs.  Compared to exact Hirschberg bit-for-bit; to
    Pointers only within a tolerance, because both Hirschberg modes replay the score from
    the recovered path while Pointers reads it off the table, and a pure gap run
    accumulates those subtractions in a different order (measured: -11.4 vs
    -11.399999999999999 for ("", "ACDE")).  That gap predates this variant."""
    for a, b in (("", ""), ("", "ACDE"), ("ACDE", ""), ("A", "A"), ("A", "CDEFGH")):
        pm = _pair(a, b, lossy_params, "hirschberg_pmax", cutoff=2)
        assert pm.score == _pair(a, b, lossy_params, "hirschberg", cutoff=2).score
        assert pm.score == pytest.approx(_pair(a, b, lossy_params, "pointers").score,
                                         abs=1e-12)
