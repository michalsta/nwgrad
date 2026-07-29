"""traceback="hirschberg" — linear-space Myers-Miller divide and conquer.

This mode is held to a DIFFERENT standard than the other two, and the difference is the
point of this file.

"pointers" and "scores" are *bit-identical* to each other: same score, same alignment,
same gradient, ties included.  test_traceback_modes.py asserts exactly that, and
Hirschberg is deliberately absent from it.  It cannot meet that bar by construction — its
split picks a midpoint argmax, and that choice cannot reproduce the backward-greedy
M>X>Y tie-break, because the tie-break depends on the rows below the split that
Hirschberg has already discarded.  Where alignments tie it returns a *different* optimal
path, hence a valid but different subgradient.

So what is actually asserted here is optimality rather than identity:

  * the score equals the other modes' score (exactly, at double precision)
  * the returned path is a real alignment of the two inputs
  * that path scores what the aligner claims it scores
  * the gradient is the gradient OF THAT PATH

Plus the guard rails: Hirschberg is implemented for affine + full + global only, and the
unsupported combinations must THROW rather than fall back.  A silent fallback would be the
worst possible failure here — every benchmark of this mode would silently be measuring
pointers instead, and the numbers would look fine.

The second half of this file ("The identity suite, relaxed") mirrors
test_traceback_modes.py case for case with equality weakened to closeness — and documents,
with measurements, which of those cases survive as exact and which diverge so far that no
tolerance could cover them.  The short version: the score is close, the gradient is not.
"""
import numpy as np
import pytest

import nwgrad

AA = "ARNDCQEGHILKMFPSTWYV"


@pytest.fixture(scope="module")
def params():
    """Integral +4/-1 with integral gap costs, so ties are COMMON.

    Under a random real-valued matrix exact ties essentially never occur, and a
    tie-handling bug would sail straight through.  Here Hirschberg is forced to make
    arbitrary choices constantly — and must still land on an optimal score every time.
    """
    m = np.full((20, 20), -1.0)
    np.fill_diagonal(m, 4.0)
    mat = nwgrad.SubstMatrix(m, alphabet=AA)
    return nwgrad.AlignParams(mat, gap_open_a=11.0, gap_extend_a=1.0,
                              gap_open_b=11.0, gap_extend_b=1.0)


@pytest.fixture(scope="module")
def asym_params():
    """Real-valued, asymmetric gap costs: gaps are frequent and ties are not.

    Complements the integral fixture — this one exercises the affine boundary handling
    across splits, where the integral matrix mostly exercises tie-breaking.
    """
    rng = np.random.default_rng(11)
    m = rng.normal(0, 3, (20, 20))
    m = (m + m.T) / 2
    np.fill_diagonal(m, 6.0)
    return nwgrad.AlignParams(nwgrad.SubstMatrix(m, alphabet=AA),
                              gap_open_a=7.5, gap_extend_a=0.7,
                              gap_open_b=4.2, gap_extend_b=1.3)


def _seqs(n, lo, hi, seed):
    rng = np.random.default_rng(seed)
    return ["".join(rng.choice(list(AA), int(l))) for l in rng.integers(lo, hi, n)]


def _rescore(a_al, b_al, params):
    """Score a gapped alignment from scratch, affine, in the DP's own convention.

    Independent of the aligner: this is what makes "the path scores what it claims" a
    real check rather than the aligner agreeing with itself.  '-' in b_al is a gap in
    sequence B and is charged gap_open_b/gap_extend_b.
    """
    M = np.asarray(params.matrix.to_matrix())
    idx = {c: i for i, c in enumerate(AA)}
    s, prev = 0.0, "M"
    for x, y in zip(a_al, b_al):
        if x != "-" and y != "-":
            s += M[idx[x], idx[y]]
            prev = "M"
        elif y == "-":
            if prev != "X":
                s -= params.gap_open_b
            s -= params.gap_extend_b
            prev = "X"
        else:
            if prev != "Y":
                s -= params.gap_open_a
            s -= params.gap_extend_a
            prev = "Y"
    return s


# ── the guard rails ───────────────────────────────────────────────────────────
#
# These matter more than they look.  Hirschberg is a different ALGORITHM, not a faster
# spelling of the same one; if an unsupported combination quietly ran pointers instead,
# nothing would fail and every measurement of this mode would be a measurement of
# something else.

def test_local_no_longer_throws(params):
    """Local Hirschberg used to throw "global alignment only".  It is now implemented
    (the endpoint reduction: forward clamped scan for the end cell, reverse global-suffix
    scan for the start, global alignment of the box between), so this must run and land
    on the Smith-Waterman optimum.  The Local suite below holds it to that."""
    sp = nwgrad.SeqPair("ACDEFGHIK", "WWACDEFGHIKWW", params, gap_model="affine",
                        mode="local", grad_mode="hard", traceback="hirschberg")
    sp.alloc_dp()
    sp.align_full()                       # no throw
    ref = nwgrad.SeqPair("ACDEFGHIK", "WWACDEFGHIKWW", params, gap_model="affine",
                         mode="local", grad_mode="hard", traceback="pointers")
    ref.alloc_dp()
    ref.align_full()
    assert sp.score == pytest.approx(ref.score, rel=1e-5, abs=1e-3)


def test_local_linear_still_throws(params):
    """The gap model, not the alignment mode, is what Local Hirschberg cannot do: there
    is no affine gap-open state to carry across a row cut in the linear model."""
    sp = nwgrad.SeqPair("ACDEFGHIK", "ACDWFGHIK", params, gap_model="linear",
                        mode="local", grad_mode="hard", traceback="hirschberg")
    sp.alloc_dp()
    with pytest.raises(RuntimeError, match="affine gap model only"):
        sp.align_full()


def test_linear_gap_model_throws(params):
    sp = nwgrad.SeqPair("ACDEFGHIK", "ACDWFGHIK", params, gap_model="linear",
                        mode="global", grad_mode="hard", traceback="hirschberg")
    sp.alloc_dp()
    with pytest.raises(RuntimeError, match="affine gap model only"):
        sp.align_full()


def test_unknown_traceback_still_rejected():
    with pytest.raises(ValueError, match="unknown traceback"):
        nwgrad.SeqPairBatch(n_threads=1, traceback="hirshberg")   # sic: typo


def test_selectable_and_reported(params):
    assert nwgrad.SeqPairBatch(n_threads=1, traceback="hirschberg").traceback == "hirschberg"
    sp = nwgrad.SeqPair("ACDE", "ACDE", params, traceback="hirschberg")
    assert sp.traceback == "hirschberg"


def test_is_the_default_where_it_applies(params):
    """Hirschberg is now the default — but only via the "auto" sentinel, which resolves
    per problem: Hirschberg for affine+global+full (where it exists), pointers otherwise.
    A blanket Hirschberg default would break Local/linear/banded, which throw.

    Which CARRY that Hirschberg uses then splits again on precision — pmax at float32,
    the exact chain at double (see test_hirschberg_pmax.py).  Both are Hirschberg; this
    test is about the algorithm, so it accepts either."""
    assert nwgrad.SeqPairBatch(n_threads=1).traceback == "auto"
    # affine + global + full resolves to a hirschberg mode
    assert nwgrad.SeqPair("ACDE", "ACDE", params, gap_model="affine",
                          mode="global").traceback == "hirschberg_pmax"
    assert nwgrad.SeqPairDouble("ACDE", "ACDE", params, gap_model="affine",
                                mode="global").traceback == "hirschberg"
    # everything Hirschberg does not implement resolves to pointers, not a throw
    assert nwgrad.SeqPair("ACDE", "ACDE", params, gap_model="affine",
                          mode="local").traceback == "pointers"
    assert nwgrad.SeqPair("ACDE", "ACDE", params, gap_model="linear",
                          mode="global").traceback == "pointers"


# ── optimality ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("fixture", ["params", "asym_params"])
def test_score_matches_pointers_exactly_at_double(request, fixture):
    """At T=double the two must agree to the bit — modulo float associativity.

    Both compute the same optimum; they may walk different paths to it, and summing a
    different path re-associates the same additions.  1e-9 is nine orders above the
    observed spread (worst seen: 4e-13) and far below any real disagreement, which would
    be a whole gap open or a substitution swap.
    """
    p = request.getfixturevalue(fixture)
    A = _seqs(60, 1, 400, 3)
    B = _seqs(60, 1, 400, 4)
    for a, b in zip(A, B):
        ref = nwgrad.SeqPairDouble(a, b, p, gap_model="affine", mode="global",
                                   grad_mode="hard", traceback="pointers")
        hb = nwgrad.SeqPairDouble(a, b, p, gap_model="affine", mode="global",
                                  grad_mode="hard", traceback="hirschberg")
        for sp in (ref, hb):
            sp.alloc_dp()
            sp.align_full()
        assert hb.score == pytest.approx(ref.score, abs=1e-9), (a, b)


def test_score_matches_pointers_at_float32(params):
    """Same claim at the shipped precision, with float32's own tolerance.

    Measured: both modes deviate from the double reference by the same amount (mean
    1.56e-3 on scores up to ~1500) — that spread is float32's, not Hirschberg's, and
    pointers already pays it.
    """
    A = _seqs(80, 1, 300, 5)
    B = _seqs(80, 1, 300, 6)
    for a, b in zip(A, B):
        ref = nwgrad.SeqPair(a, b, params, gap_model="affine", mode="global",
                             grad_mode="hard", traceback="pointers")
        hb = nwgrad.SeqPair(a, b, params, gap_model="affine", mode="global",
                            grad_mode="hard", traceback="hirschberg")
        for sp in (ref, hb):
            sp.alloc_dp()
            sp.align_full()
        assert hb.score == pytest.approx(ref.score, rel=1e-5, abs=1e-3), (a, b)


def test_recursion_actually_runs(params):
    """A sequence far longer than the base-case cutoff, so splits genuinely happen.

    Without this, every other test here could pass against a base case that never
    recursed — i.e. against the pointers fill wearing a different name.
    """
    a = _seqs(1, 900, 901, 21)[0]
    b = _seqs(1, 900, 901, 22)[0]
    ref = nwgrad.SeqPairDouble(a, b, params, gap_model="affine", mode="global",
                               grad_mode="hard", traceback="pointers")
    hb = nwgrad.SeqPairDouble(a, b, params, gap_model="affine", mode="global",
                              grad_mode="hard", traceback="hirschberg")
    for sp in (ref, hb):
        sp.alloc_dp()
        sp.align_full()
    assert hb.score == pytest.approx(ref.score, abs=1e-9)


# ── the path is real, and is what was scored ──────────────────────────────────

@pytest.mark.parametrize("fixture", ["params", "asym_params"])
def test_alignment_reconstructs_inputs(request, fixture):
    p = request.getfixturevalue(fixture)
    A = _seqs(40, 1, 250, 7)
    B = _seqs(40, 1, 250, 8)
    for a, b in zip(A, B):
        sp = nwgrad.SeqPairDouble(a, b, p, gap_model="affine", mode="global",
                                  grad_mode="hard", traceback="hirschberg")
        sp.alloc_dp()
        sp.align_full()
        x, y = sp.aligned()
        assert len(x) == len(y)
        assert x.replace("-", "") == a
        assert y.replace("-", "") == b
        assert not any(u == "-" and v == "-" for u, v in zip(x, y))


@pytest.mark.parametrize("fixture", ["params", "asym_params"])
def test_path_scores_what_it_claims(request, fixture):
    """The reported score must be the score of the returned alignment.

    Note what this does and does not catch.  The aligner derives its score by replaying
    the recovered path, so "reported == path" is nearly tautological on the C++ side;
    what is actually under test is that the replay's affine convention agrees with an
    INDEPENDENT implementation of it (_rescore, above) — gap opens charged once per run,
    to the right one of the two sequences, in the right precision.

    It does NOT catch a bad split.  Verified by mutation: deleting the spanning-X refund
    in hb_solve makes the path genuinely suboptimal, and this check still passes — the
    path is merely a worse alignment, scored consistently.  The tests that caught it were
    the ones comparing against pointers (5 of them).  Optimality has to be checked against
    an oracle; self-consistency cannot see it.
    """
    p = request.getfixturevalue(fixture)
    A = _seqs(40, 1, 300, 9)
    B = _seqs(40, 1, 300, 10)
    for a, b in zip(A, B):
        sp = nwgrad.SeqPairDouble(a, b, p, gap_model="affine", mode="global",
                                  grad_mode="hard", traceback="hirschberg")
        sp.alloc_dp()
        sp.align_full()
        x, y = sp.aligned()
        assert _rescore(x, y, p) == pytest.approx(sp.score, abs=1e-9), (a, b)


def test_gradient_is_the_gradient_of_that_path(asym_params):
    """The gradient must count exactly the path that was returned — not merely be
    *a* plausible gradient.  Counting matched pairs off the alignment strings and
    comparing to the matrix block is independent of how the DP produced either."""
    A = _seqs(30, 20, 250, 11)
    B = _seqs(30, 20, 250, 12)
    idx = {c: i for i, c in enumerate(AA)}
    for a, b in zip(A, B):
        sp = nwgrad.SeqPairDouble(a, b, asym_params, gap_model="affine", mode="global",
                                  grad_mode="hard", traceback="hirschberg")
        sp.alloc_dp()
        sp.align_full()
        sp.compute_grad()
        g = sp.grad.to_dict()
        x, y = sp.aligned()
        want = np.zeros((20, 20))
        opens_a = opens_b = ext_a = ext_b = 0
        prev = "M"
        for u, v in zip(x, y):
            if u != "-" and v != "-":
                want[idx[u], idx[v]] += 1.0
                prev = "M"
            elif v == "-":
                if prev != "X":
                    opens_b += 1
                ext_b += 1
                prev = "X"
            else:
                if prev != "Y":
                    opens_a += 1
                ext_a += 1
                prev = "Y"
        np.testing.assert_array_equal(g["matrix"], want)
        # Gap fields of a gradient are NON-POSITIVE: the score subtracts penalties, so
        # the derivative w.r.t. a penalty is minus its count.  See the sign convention.
        assert g["gap_open_a"] == -opens_a
        assert g["gap_open_b"] == -opens_b
        assert g["gap_extend_a"] == -ext_a
        assert g["gap_extend_b"] == -ext_b


# ── boundaries ────────────────────────────────────────────────────────────────

def test_degenerate_shapes(params):
    """Empty and single-residue inputs exercise the block ORIGIN — the M/X seeding that
    carries affine state across a cut, and the H==0 / W==0 blocks the recursion bottoms
    out on."""
    odd = ["", "A", "AC", "ACDEFGHIK"]
    for a in odd:
        for b in odd:
            ref = nwgrad.SeqPairDouble(a, b, params, gap_model="affine", mode="global",
                                       grad_mode="hard", traceback="pointers")
            hb = nwgrad.SeqPairDouble(a, b, params, gap_model="affine", mode="global",
                                      grad_mode="hard", traceback="hirschberg")
            for sp in (ref, hb):
                sp.alloc_dp()
                sp.align_full()
            assert hb.score == pytest.approx(ref.score, abs=1e-9), (a, b)
            x, y = hb.aligned()
            assert x.replace("-", "") == a
            assert y.replace("-", "") == b


def test_batch_matches_single_pair(params):
    """The batch path must agree with the per-pair path, and be reproducible."""
    A = _seqs(120, 1, 200, 13)
    B = _seqs(120, 1, 200, 14)
    b1 = nwgrad.SeqPairBatch(n_threads=4, traceback="hirschberg")
    b1.add_many(A, B, params, gap_model="affine", mode="global",
                grad_mode="hard", kernel="auto")
    t1 = b1.score_and_grad()
    b2 = nwgrad.SeqPairBatch(n_threads=1, traceback="hirschberg")
    b2.add_many(A, B, params, gap_model="affine", mode="global",
                grad_mode="hard", kernel="auto")
    assert b2.score_and_grad() == t1          # thread count must not change the answer


def test_batch_total_matches_pointers(params):
    """The headline claim, at batch level: same optimum, different route to it."""
    A = _seqs(200, 1, 250, 15)
    B = _seqs(200, 1, 250, 16)
    out = {}
    for tb in ("pointers", "hirschberg"):
        batch = nwgrad.SeqPairBatch(n_threads=4, traceback=tb)
        batch.add_many(A, B, params, gap_model="affine", mode="global",
                       grad_mode="hard", kernel="auto")
        out[tb] = batch.score_and_grad()
    assert out["hirschberg"] == pytest.approx(out["pointers"], rel=1e-5, abs=1e-2)


# ══════════════════════════════════════════════════════════════════════════════
# Local (Smith-Waterman) linear space
# ══════════════════════════════════════════════════════════════════════════════
#
# Local Hirschberg is the endpoint reduction: a forward CLAMPED scan finds the end cell
# (ie, je) and score S, a reverse UNCLAMPED global-suffix scan finds the start (is, js),
# and the box A[is..ie) x B[js..je) is aligned GLOBALLY by the same recursion global uses.
# It is held to the same bar — optimality, not identity — with the SW-specific twists:
#   * the oracle is LOCAL pointers (Smith-Waterman), not global;
#   * `aligned()` returns only the aligned sub-region, so a reconstruction check asserts a
#     contiguous SUBSTRING of each input rather than the whole input;
#   * self-pairs are a unique optimum (the whole diagonal), so there Hirschberg is
#     BIT-exact — score and alignment — which also proves the reverse-scan start is right.
# The reverse scan is global-suffix, not clamped, precisely so its argmax box always
# scores S: a clamped reverse could pick, at a tie, a start whose box misses (ie, je).


def _local(a, b, p, tb, cutoff=None, dtype="double"):
    cls = nwgrad.SeqPairDouble if dtype == "double" else nwgrad.SeqPair
    sp = cls(a, b, p, gap_model="affine", mode="local", grad_mode="hard", traceback=tb)
    if cutoff is not None and tb == "hirschberg":
        sp.hb_cutoff = cutoff
    sp.alloc_dp()
    sp.align_full()
    return sp


@pytest.mark.parametrize("fixture", ["params", "asym_params"])
def test_local_score_matches_pointers(request, fixture):
    """The headline claim: Smith-Waterman optimum, reached in linear space.  cutoff=64
    forces splits on the longer pairs so this is not secretly testing the pointers fill."""
    p = request.getfixturevalue(fixture)
    A = _seqs(80, 1, 400, 41)
    B = _seqs(80, 1, 400, 42)
    for a, b in zip(A, B):
        ref = _local(a, b, p, "pointers")
        hb = _local(a, b, p, "hirschberg", cutoff=64)
        assert hb.score == pytest.approx(ref.score, abs=1e-9), (a, b)


def test_local_self_pair_is_bit_exact(params):
    """A self-pair has a unique optimum (the full diagonal) — no tie to break — so local
    Hirschberg must reproduce local pointers to the bit, alignment included.  This also
    forces a genuine split (length 900 >> cutoff 32) and pins the reverse-scan start: a
    wrong start would shift the whole alignment."""
    a = _seqs(1, 900, 901, 51)[0]
    ref = _local(a, a, params, "pointers")
    hb = _local(a, a, params, "hirschberg", cutoff=32)
    assert hb.score == ref.score
    assert hb.aligned() == ref.aligned()


@pytest.mark.parametrize("fixture", ["params", "asym_params"])
def test_local_alignment_is_a_real_substring_alignment(request, fixture):
    p = request.getfixturevalue(fixture)
    A = _seqs(50, 1, 300, 43)
    B = _seqs(50, 1, 300, 44)
    for a, b in zip(A, B):
        sp = _local(a, b, p, "hirschberg", cutoff=64)
        x, y = sp.aligned()
        assert len(x) == len(y)
        assert x.replace("-", "") in a          # the aligned A-region is a contiguous substring
        assert y.replace("-", "") in b
        assert not any(u == "-" and v == "-" for u, v in zip(x, y))
        assert _rescore(x, y, p) == pytest.approx(sp.score, abs=1e-9), (a, b)


def test_local_gradient_is_the_gradient_of_that_path(asym_params):
    """Same as global: the hard gradient must count exactly the returned local path."""
    A = _seqs(30, 20, 250, 45)
    B = _seqs(30, 20, 250, 46)
    idx = {c: i for i, c in enumerate(AA)}
    for a, b in zip(A, B):
        sp = _local(a, b, asym_params, "hirschberg", cutoff=64)
        sp.compute_grad()
        g = sp.grad.to_dict()
        x, y = sp.aligned()
        want = np.zeros((20, 20))
        opens_a = opens_b = ext_a = ext_b = 0
        prev = "M"
        for u, v in zip(x, y):
            if u != "-" and v != "-":
                want[idx[u], idx[v]] += 1.0
                prev = "M"
            elif v == "-":
                if prev != "X":
                    opens_b += 1
                ext_b += 1
                prev = "X"
            else:
                if prev != "Y":
                    opens_a += 1
                ext_a += 1
                prev = "Y"
        np.testing.assert_array_equal(g["matrix"], want)
        assert g["gap_open_a"] == -opens_a
        assert g["gap_open_b"] == -opens_b
        assert g["gap_extend_a"] == -ext_a
        assert g["gap_extend_b"] == -ext_b


def test_local_degenerate_shapes(params):
    """Empty and tiny inputs: the S<=0 / empty-alignment early-out, and boxes so small the
    recursion never splits (bit-exact there by construction)."""
    odd = ["", "A", "AC", "ACDEFGHIK"]
    for a in odd:
        for b in odd:
            ref = _local(a, b, params, "pointers")
            hb = _local(a, b, params, "hirschberg")
            assert hb.score == pytest.approx(ref.score, abs=1e-9), (a, b)
            x, y = hb.aligned()
            assert x.replace("-", "") in a
            assert y.replace("-", "") in b


def test_local_batch_matches_pointers(params):
    """Batch level, same optimum as local pointers, reproducible across thread counts."""
    A = _seqs(150, 1, 250, 47)
    B = _seqs(150, 1, 250, 48)
    out = {}
    for tb in ("pointers", "hirschberg"):
        batch = nwgrad.SeqPairBatch(n_threads=4, traceback=tb)
        batch.add_many(A, B, params, gap_model="affine", mode="local",
                       grad_mode="hard", kernel="auto")
        out[tb] = batch.score_and_grad()          # scalar: total score over the batch
    assert out["hirschberg"] == pytest.approx(out["pointers"], rel=1e-5, abs=1e-2)


# ══════════════════════════════════════════════════════════════════════════════
# The identity suite, relaxed
# ══════════════════════════════════════════════════════════════════════════════
#
# These mirror test_traceback_modes.py case for case, with "exactly equal" weakened to
# "close".  Measuring first was worth it, because the weakening is NOT uniform — some of
# those cases survive as exact, and some fail so badly that no tolerance would save them.
# Measured, 120 random pairs of 20-300 aa, hirschberg vs pointers at T=double:
#
#   quantity                       unique optimum      ties (integral matrix)
#   ---------------------------------------------------------------------------
#   score                          identical           identical
#   soft gradient                  BIT-identical       BIT-identical
#   grad . params == score         exact               exact
#   M+X==m, M+Y==n                 exact               exact
#   gradient matrix                120/120 identical   96/120; L1 diff up to 1.42x
#   aligned strings                106/120 identical   96/120
#   guide_j                        22/120 identical    29/120
#   banded re-alignment            (derived from guide_j — differs materially)
#
# So the honest summary is: **the score is close, the gradient is not.**  Under ties the
# L1 difference between the two gradient matrices reaches 1.42x the gradient's own
# magnitude — they can share almost nothing.  That is not a defect; it is what "a valid
# but different subgradient" means when nearly every cell is a tie.  Asserting elementwise
# closeness there would be asserting something false.
#
# What replaces it is `grad . params == score`: every parameter enters the score
# linearly, so a gradient is consistent with its own path exactly when that dot product
# reproduces the score.  It holds to the bit for both modes and is the sharp statement
# that survives the tie-breaking freedom.


def _dot(g, p):
    """The score implied by a gradient.  Linear in every parameter, so this must equal
    the score exactly — for ANY valid subgradient, whichever path it came from."""
    M = np.asarray(p.matrix.to_matrix())
    return (float((g["matrix"] * M).sum())
            + g["gap_open_a"] * p.gap_open_a + g["gap_extend_a"] * p.gap_extend_a
            + g["gap_open_b"] * p.gap_open_b + g["gap_extend_b"] * p.gap_extend_b)


def _pair(a, b, p, tb, cutoff=None):
    sp = nwgrad.SeqPairDouble(a, b, p, gap_model="affine", mode="global",
                              grad_mode="hard", traceback=tb)
    if cutoff is not None and tb == "hirschberg":
        sp.hb_cutoff = cutoff
    sp.alloc_dp()
    sp.align_full()
    sp.compute_grad()
    return sp


@pytest.mark.parametrize("fixture", ["params", "asym_params"])
def test_relaxed_score_and_grad(request, fixture):
    """Analog of test_score_and_grad_identical.

    Score: close.  Gradient: NOT compared elementwise — see the table above.  What is
    asserted instead is that each gradient reproduces its own score, which is the
    property elementwise equality was standing in for.
    """
    p = request.getfixturevalue(fixture)
    A = _seqs(60, 20, 300, 31)
    B = _seqs(60, 20, 300, 32)
    for a, b in zip(A, B):
        ref, hb = _pair(a, b, p, "pointers"), _pair(a, b, p, "hirschberg")
        assert hb.score == pytest.approx(ref.score, abs=1e-9), (a, b)
        for sp in (ref, hb):
            assert _dot(sp.grad.to_dict(), p) == pytest.approx(sp.score, abs=1e-9)


@pytest.mark.parametrize("fixture", ["params", "asym_params"])
def test_gradient_structural_identities_exact(request, fixture):
    """Exact even where the gradient itself is not: matched pairs plus gaps-in-B must
    account for every residue of A, and likewise for B.  Integer identities, so they
    hold for any optimal path and are immune to the tie-breaking freedom."""
    p = request.getfixturevalue(fixture)
    A = _seqs(40, 20, 300, 33)
    B = _seqs(40, 20, 300, 34)
    for a, b in zip(A, B):
        for tb in ("pointers", "hirschberg"):
            g = _pair(a, b, p, tb).grad.to_dict()
            matched = g["matrix"].sum()
            assert matched - g["gap_extend_b"] == len(a), (tb, a, b)
            assert matched - g["gap_extend_a"] == len(b), (tb, a, b)


def test_gradient_identical_when_optimum_is_unique(asym_params):
    """With a real-valued matrix the optimum is unique, so there is no freedom left and
    the relaxation should not be needed: the gradients must match EXACTLY.  Measured
    120/120.  This is the test that would catch a genuinely wrong path — one that the
    tie-tolerant tests above would wave through."""
    A = _seqs(80, 20, 300, 35)
    B = _seqs(80, 20, 300, 36)
    for a, b in zip(A, B):
        gp = _pair(a, b, asym_params, "pointers").grad.to_dict()
        gh = _pair(a, b, asym_params, "hirschberg").grad.to_dict()
        np.testing.assert_array_equal(gp["matrix"], gh["matrix"])
        for f in ("gap_open_a", "gap_extend_a", "gap_open_b", "gap_extend_b"):
            assert gp[f] == gh[f], (f, a, b)


def test_gradients_really_do_diverge_under_ties(params):
    """A characterization test, deliberately asserting that the two DISAGREE.

    Forces a small hb_cutoff so the recursion actually SPLITS — divergence only happens
    at splits (a non-splitting Hirschberg is bit-exact with pointers, verified elsewhere),
    so this must pin the cutoff below the sequence lengths regardless of the default.

    If this ever fails, Hirschberg has become tie-exact with pointers even when splitting —
    which would be good news, but it would also make this module's whole framing (and the
    CLAUDE.md paragraph, and its exclusion from test_traceback_modes.py) wrong.
    """
    A = _seqs(120, 20, 300, 3)
    B = _seqs(120, 20, 300, 4)
    differ = 0
    for a, b in zip(A, B):
        gp = _pair(a, b, params, "pointers").grad.to_dict()["matrix"]
        gh = _pair(a, b, params, "hirschberg", cutoff=8).grad.to_dict()["matrix"]
        if not np.array_equal(gp, gh):
            differ += 1
    assert differ > 0, ("hirschberg now agrees with pointers on every tied gradient — "
                        "if this is real, update CLAUDE.md and consider promoting it "
                        "into test_traceback_modes.py")


def test_large_cutoff_is_bit_exact_with_pointers(params):
    """With hb_cutoff above the sequence length the recursion never splits, so the whole
    problem is one (vectorized) base case — which IS the Pointers fill.  It must then be
    bit-identical to pointers: score, alignment AND gradient, ties included.

    This is the property that makes Hirschberg safe as a default: for any pair short
    enough to fit the cutoff it degrades exactly to the previous default, and only pairs
    longer than the cutoff (the ones where Pointers' memory is the problem) take the
    different-subgradient linear-space path.
    """
    A = _seqs(80, 1, 250, 51)
    B = _seqs(80, 1, 250, 52)
    for a, b in zip(A, B):
        gp = _pair(a, b, params, "pointers")
        gh = _pair(a, b, params, "hirschberg", cutoff=100000)
        assert gh.score == gp.score, (a, b)
        assert gh.aligned() == gp.aligned(), (a, b)
        np.testing.assert_array_equal(gh.grad.to_dict()["matrix"],
                                      gp.grad.to_dict()["matrix"])


def test_soft_gradient_unaffected(params):
    """Analog of test_soft_gradient_unaffected.

    The soft path is forward-backward in double and never runs a Viterbi traceback, so
    the traceback mode is a pure no-op for it — pointers and hirschberg exercise
    identical code.  The comparison is nonetheless tolerant, not exact, and the reason
    is NOT hirschberg: score_and_grad reduces per-pair contributions across threads in a
    non-deterministic order, so two batch runs differ in the last ULP whatever the
    traceback.  (An earlier version asserted exact equality and was flaky under
    pytest-randomly for exactly this reason — the sibling test in test_traceback_modes.py
    uses the same tolerance.)
    """
    A = _seqs(40, 5, 200, 37)
    B = _seqs(40, 5, 200, 38)
    out = {}
    for tb in ("pointers", "hirschberg"):
        batch = nwgrad.SeqPairBatch(n_threads=2, traceback=tb)
        batch.add_many(A, B, params, gap_model="affine", mode="global",
                       grad_mode="soft", kernel="auto")
        out[tb] = (batch.score_and_grad(), batch.compute_grad().to_dict())
    assert out["hirschberg"][0] == pytest.approx(out["pointers"][0], rel=1e-12)
    np.testing.assert_allclose(out["hirschberg"][1]["matrix"],
                               out["pointers"][1]["matrix"], atol=1e-10)


@pytest.mark.parametrize("band", [8, 32])
def test_banded_composes_but_does_not_agree(params, band):
    """Analog of test_banded_grad_identical, and the case that relaxes FURTHEST.

    Banded re-alignment is constrained around the guide extracted from the full
    alignment, and guide_j is where the two modes diverge most (22-29 of 120 identical).
    Different guide, different band, materially different banded score: measured
    pointers -12588 vs hirschberg -12152 at band=8 on this fixture — 3.6% apart, which no
    sane tolerance would admit.  (Hirschberg came out HIGHER in both bands tested; not
    investigated, and not claimed as a property.)

    What IS asserted is the invariant that must hold regardless: a band can only restrict
    the search, so a banded score can never exceed the full one, for either mode.
    """
    A = _seqs(60, 20, 300, 39)
    B = _seqs(60, 20, 300, 40)
    for tb in ("pointers", "hirschberg"):
        batch = nwgrad.SeqPairBatch(n_threads=2, traceback=tb)
        batch.add_many(A, B, params, gap_model="affine", mode="global",
                       grad_mode="hard", kernel="auto")
        full = batch.score_and_grad()
        banded = batch.banded_grad(band)
        assert banded <= full + 1e-9, (tb, band, banded, full)


def test_self_pair_closed_form(params):
    """Self-alignment has a UNIQUE optimum (every diagonal entry is positive), so here
    Hirschberg has no tie to break and must reproduce the closed form exactly —
    score and gradient both.  This is the one case where it is provably bit-exact."""
    seqs = _seqs(40, 5, 300, 17)
    batch = nwgrad.SeqPairBatch(n_threads=4, traceback="hirschberg")
    batch.add_many(seqs, seqs, params, gap_model="affine", mode="global",
                   grad_mode="hard", kernel="auto")
    total = batch.score_and_grad()
    idx = {c: i for i, c in enumerate(AA)}
    want_total = 0.0
    want_m = np.zeros((20, 20))
    for s in seqs:
        for c in s:
            want_total += 4.0
            want_m[idx[c], idx[c]] += 1.0
    assert total == pytest.approx(want_total, rel=1e-6)
    g = batch.compute_grad().to_dict()
    np.testing.assert_array_equal(g["matrix"], want_m)
    for f in ("gap_open_a", "gap_extend_a", "gap_open_b", "gap_extend_b"):
        assert g[f] == 0.0, f
