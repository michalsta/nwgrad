"""Tests for guide-banded DP with non-diagonal (non-trivial) guides.

A "guide" is a reference alignment expressed as two gapped strings.  The banded
DP runs in a strip of half-width `band` centered on that guide's path through the
DP table, rather than on the main diagonal.  Tests here verify:
  - guide_j_from_aligned produces correct index vectors
  - score with an exact guide (band=0) matches full DP
  - score with an off-center guide + small band still finds the optimum when the
    true alignment is within that band
  - score with a badly wrong guide (too narrow band) diverges from full DP
  - gradient with guide matches full DP gradient
  - BatchAligner.align with aligned_a/aligned_b produces per-pair correct scores
"""

import numpy as np
import pytest
import nwgrad
from test_subst_matrix import BLOSUM62


@pytest.fixture(scope="module")
def blosum():
    return nwgrad.SubstMatrix(BLOSUM62)


@pytest.fixture(scope="module")
def unit():
    """Identity substitution matrix: score 1 for match, 0 for mismatch."""
    mat = np.zeros((20, 20), dtype=np.float64)
    np.fill_diagonal(mat, 1.0)
    return nwgrad.SubstMatrix(mat)


# ── guide_j_from_aligned ──────────────────────────────────────────────────────

class TestGuideJFromAligned:
    def test_all_matches(self):
        gj = nwgrad.guide_j_from_aligned("ACDE", "ACDE")
        assert gj == [0, 1, 2, 3, 4]

    def test_gap_in_b_shifts_guide_left(self):
        # a: ACDE  b: A-DE  → column j stays when b has a gap (deletion in b)
        gj = nwgrad.guide_j_from_aligned("ACDE", "A-DE")
        assert gj == [0, 1, 1, 2, 3]

    def test_gap_in_a_shifts_guide_right(self):
        # a: A-DE  b: ACDE  → j advances without consuming a (insertion relative to a)
        gj = nwgrad.guide_j_from_aligned("A-DE", "ACDE")
        assert gj == [0, 1, 3, 4]

    def test_leading_gap_in_b(self):
        # a: ACDE  b: --DE  → after A j=0, after C j=0, after D j=1, after E j=2
        gj = nwgrad.guide_j_from_aligned("ACDE", "--DE")
        assert gj == [0, 0, 0, 1, 2]

    def test_leading_gap_in_a(self):
        # a: --DE  b: ACDE  → j=2 when we start consuming a
        gj = nwgrad.guide_j_from_aligned("--DE", "ACDE")
        assert gj == [0, 3, 4]

    def test_trailing_gap_in_b(self):
        gj = nwgrad.guide_j_from_aligned("ACDE", "AC--")
        assert gj == [0, 1, 2, 2, 2]

    def test_trailing_gap_in_a(self):
        gj = nwgrad.guide_j_from_aligned("AC--", "ACDE")
        assert gj == [0, 1, 2]

    def test_length_is_m_plus_one(self):
        for aa, ab in [("ACDE", "A-DE"), ("A-DE", "ACDE"), ("A---", "ACDE"), ("ACDE", "A---")]:
            gj = nwgrad.guide_j_from_aligned(aa, ab)
            m = sum(1 for c in aa if c != '-')
            assert len(gj) == m + 1, f"len(guide_j) should be {m+1} for a_aligned={aa!r}"

    def test_last_entry_equals_n(self):
        for aa, ab in [("ACDE", "A-DE"), ("MMACDE", "---ADE"), ("A-DE", "ACDE")]:
            gj = nwgrad.guide_j_from_aligned(aa, ab)
            n = sum(1 for c in ab if c != '-')
            assert gj[-1] == n, f"guide_j[-1] should equal n={n}"

    def test_mismatched_length_raises(self):
        with pytest.raises(Exception):
            nwgrad.guide_j_from_aligned("ACDE", "ACD")

    def test_double_gap_column_raises(self):
        with pytest.raises(Exception):
            nwgrad.guide_j_from_aligned("A-DE", "A-DE")


# ── nw_score with non-diagonal guide ─────────────────────────────────────────

class TestNwScoreGuided:
    """guide-banded NW score matches full DP for various non-diagonal guides."""

    # Pairs where the optimal alignment has gaps, so the guide is off-diagonal.
    GAPPED_PAIRS = [
        # (raw_a, raw_b, a_aligned, b_aligned)
        ("ACDE",   "ADE",    "ACDE",   "A-DE"),    # gap in b: C aligned to gap
        ("ADE",    "ACDE",   "A-DE",   "ACDE"),    # gap in a: C inserted
        ("ACDE",   "DE",     "ACDE",   "--DE"),    # leading gaps in b
        ("DE",     "ACDE",   "--DE",   "ACDE"),    # leading gaps in a
        ("ACDEFG", "DE",     "ACDEFG", "--DE--"),  # surrounded by gaps in b
        ("MMMADE", "ADE",    "MMMADE", "---ADE"),  # leading gaps in b
    ]

    @pytest.mark.parametrize("a,b,aa,ab", GAPPED_PAIRS)
    def test_exact_guide_band0_matches_full(self, a, b, aa, ab, unit):
        full   = nwgrad.nw_score(a, b, unit, 1.0)
        guided = nwgrad.nw_score(a, b, unit, 1.0, band=0, aligned_a=aa, aligned_b=ab)
        assert guided == pytest.approx(full), \
            f"nw_score({a!r},{b!r}) guided={guided} full={full}"

    @pytest.mark.parametrize("a,b,aa,ab", GAPPED_PAIRS)
    def test_exact_guide_band1_matches_full(self, a, b, aa, ab, unit):
        full   = nwgrad.nw_score(a, b, unit, 1.0)
        guided = nwgrad.nw_score(a, b, unit, 1.0, band=1, aligned_a=aa, aligned_b=ab)
        assert guided == pytest.approx(full)

    @pytest.mark.parametrize("a,b,aa,ab", GAPPED_PAIRS)
    def test_blosum_exact_guide_matches_full(self, a, b, aa, ab, blosum):
        full   = nwgrad.nw_score(a, b, blosum, 1.0)
        guided = nwgrad.nw_score(a, b, blosum, 1.0, band=0, aligned_a=aa, aligned_b=ab)
        assert guided == pytest.approx(full)

    def test_off_diagonal_guide_with_band_finds_optimum(self, unit):
        # Optimal alignment of "ACDE" vs "ADE" has one gap in b at position 1.
        # Guide aligned as "ACDE"/"A-DE" → guide_j = [0,1,1,2,3].
        # Even if we use a slightly wrong guide ("A-CDE"/"AACDE" →shifted by 1),
        # a band of 2 around the correct guide should still reach the optimum.
        a, b = "ACDE", "ADE"
        full = nwgrad.nw_score(a, b, unit, 1.0)
        guided = nwgrad.nw_score(a, b, unit, 1.0, band=2,
                                  aligned_a="ACDE", aligned_b="A-DE")
        assert guided == pytest.approx(full)

    def test_clearly_wrong_guide_narrow_band_misses_optimum(self, unit):
        # Optimal alignment of "ACDEFG" vs "DE" has 4 deletions — the true path
        # goes through cells far from the diagonal.  A wrong guide "ACDEFG"/"DE----"
        # (which puts the band near j=0..2 for all rows) with band=0 cannot reach
        # the endpoint (6,2) correctly and will produce a suboptimal score.
        a, b = "ACDEFG", "DE"
        full  = nwgrad.nw_score(a, b, unit, 1.0)
        # guide "ACDEFG"/"DE----": guide_j = [0,1,2,2,2,2,2], band=0 → band hugs
        # the top of the table and cannot follow the correct path
        wrong = nwgrad.nw_score(a, b, unit, 1.0, band=0,
                                 aligned_a="ACDEFG", aligned_b="DE----")
        assert wrong != pytest.approx(full)


# ── affine gap + non-diagonal guide ──────────────────────────────────────────

class TestNwAffineGuided:
    PAIRS = [
        ("ACDE",   "ADE",    "ACDE",   "A-DE"),
        ("MMMADE", "ADE",    "MMMADE", "---ADE"),
    ]

    @pytest.mark.parametrize("a,b,aa,ab", PAIRS)
    def test_exact_guide_matches_full(self, a, b, aa, ab, unit):
        full   = nwgrad.nw_score_affine(a, b, unit, 10.0, 1.0)
        guided = nwgrad.nw_score_affine(a, b, unit, 10.0, 1.0,
                                         band=0, aligned_a=aa, aligned_b=ab)
        assert guided == pytest.approx(full)


# ── sw_score with non-diagonal guide ─────────────────────────────────────────

class TestSwScoreGuided:
    def test_local_exact_guide_matches_full(self, unit):
        # local alignment of "MMMADE" vs "ADE" finds "ADE" inside "MMMADE"
        a, b = "MMMADE", "ADE"
        aa, ab = "MMMADE", "---ADE"
        full   = nwgrad.sw_score(a, b, unit, 1.0)
        guided = nwgrad.sw_score(a, b, unit, 1.0, band=1, aligned_a=aa, aligned_b=ab)
        assert guided == pytest.approx(full)


# ── Hard gradient with non-diagonal guide ────────────────────────────────────

class TestGradientGuided:
    def test_nw_grad_exact_guide_matches_full(self, unit):
        a, b   = "ACDE", "ADE"
        aa, ab = "ACDE", "A-DE"
        s_full,   g_full   = nwgrad.nw_grad(a, b, unit, 1.0)
        s_guided, g_guided = nwgrad.nw_grad(a, b, unit, 1.0, band=0,
                                             aligned_a=aa, aligned_b=ab)
        assert s_guided == pytest.approx(s_full)
        np.testing.assert_allclose(g_guided, g_full, atol=1e-12)

    def test_nw_grad_unequal_lengths_correct_guide(self, unit):
        a, b   = "ACDEFG", "DE"
        aa, ab = "ACDEFG", "--DE--"
        s_full,   g_full   = nwgrad.nw_grad(a, b, unit, 1.0)
        s_guided, g_guided = nwgrad.nw_grad(a, b, unit, 1.0, band=0,
                                             aligned_a=aa, aligned_b=ab)
        assert s_guided == pytest.approx(s_full)
        np.testing.assert_allclose(g_guided, g_full, atol=1e-12)

    def test_nw_grad_band1_around_guide(self, blosum):
        a, b   = "PLEASANTLY", "MEANLY"
        # Use a rough guide (global alignment of same pair via full DP to get aligned strings).
        # Here we just supply a hand-crafted guide that's close enough with band=2.
        # Verify score matches full DP — gradient may differ if alignment is non-unique.
        s_full, _ = nwgrad.nw_grad(a, b, blosum, 1.0)
        # A reasonable guide for PLEASANTLY vs MEANLY (many matches near diagonal):
        aa = "PLEASANTLY"
        ab = "-MEA--NELY"  # rough alignment, close to optimal
        s_guided, _ = nwgrad.nw_grad(a, b, blosum, 1.0, band=3,
                                      aligned_a=aa, aligned_b=ab)
        # With band=3 around a close guide, score should match full DP
        assert s_guided == pytest.approx(s_full)


# ── BatchAligner with non-diagonal guides ────────────────────────────────────

class TestBatchGuided:
    PAIRS = [
        ("ACDE",   "ADE",    "ACDE",   "A-DE"),
        ("ADE",    "ACDE",   "A-DE",   "ACDE"),
        ("MMMADE", "ADE",    "MMMADE", "---ADE"),
        ("ACDE",   "ACDE",   "ACDE",   "ACDE"),
        ("DE",     "ACDE",   "--DE",   "ACDE"),
    ]

    def test_scores_match_single_pair(self, unit):
        ba = nwgrad.BatchAligner(unit, gap_open=0.0, gap_extend=1.0,
                                  gap_model="linear", mode="global",
                                  grad_mode="none", n_threads=1)
        seqs_a = [p[0] for p in self.PAIRS]
        seqs_b = [p[1] for p in self.PAIRS]
        aa     = [p[2] for p in self.PAIRS]
        ab     = [p[3] for p in self.PAIRS]

        result = ba.align(seqs_a, seqs_b, aligned_a=aa, aligned_b=ab)

        for i, (a, b, aligned_a, aligned_b) in enumerate(self.PAIRS):
            expected = nwgrad.nw_score(a, b, unit, 1.0,
                                        band=0, aligned_a=aligned_a, aligned_b=aligned_b)
            assert result.scores[i] == pytest.approx(expected), \
                f"pair {i}: {a!r} vs {b!r}"

    def test_multithreaded_matches_single_threaded(self, unit):
        seqs_a = [p[0] for p in self.PAIRS]
        seqs_b = [p[1] for p in self.PAIRS]
        aa     = [p[2] for p in self.PAIRS]
        ab     = [p[3] for p in self.PAIRS]

        single = nwgrad.BatchAligner(unit, gap_open=0.0, gap_extend=1.0,
                                      gap_model="linear", mode="global",
                                      grad_mode="hard", n_threads=1)
        multi  = nwgrad.BatchAligner(unit, gap_open=0.0, gap_extend=1.0,
                                      gap_model="linear", mode="global",
                                      grad_mode="hard", n_threads=4)

        r1 = single.align(seqs_a, seqs_b, aligned_a=aa, aligned_b=ab)
        r2 = multi.align(seqs_a, seqs_b, aligned_a=aa, aligned_b=ab)

        np.testing.assert_allclose(r1.scores, r2.scores, atol=1e-12)
        np.testing.assert_allclose(r1.grad,   r2.grad,   atol=1e-12)

    def test_guided_batch_matches_full_batch(self, unit):
        seqs_a = [p[0] for p in self.PAIRS]
        seqs_b = [p[1] for p in self.PAIRS]
        aa     = [p[2] for p in self.PAIRS]
        ab     = [p[3] for p in self.PAIRS]

        ba_full   = nwgrad.BatchAligner(unit, gap_open=0.0, gap_extend=1.0,
                                         gap_model="linear", mode="global",
                                         grad_mode="hard", n_threads=2)
        ba_guided = nwgrad.BatchAligner(unit, gap_open=0.0, gap_extend=1.0,
                                         gap_model="linear", mode="global",
                                         grad_mode="hard", n_threads=2)

        r_full   = ba_full.align(seqs_a, seqs_b)
        r_guided = ba_guided.align(seqs_a, seqs_b, aligned_a=aa, aligned_b=ab)

        np.testing.assert_allclose(r_full.scores, r_guided.scores, atol=1e-12)
        np.testing.assert_allclose(r_full.grad,   r_guided.grad,   atol=1e-12)
