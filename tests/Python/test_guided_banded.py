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

IDENTITY = np.zeros((20, 20), dtype=np.float64)
np.fill_diagonal(IDENTITY, 1.0)


def make_params(mat, gap_extend, gap_open=0.0):
    return nwgrad.AlignParams(np.asarray(mat, dtype=np.float64),
                               gap_open_a=gap_open, gap_extend_a=gap_extend,
                               gap_open_b=gap_open, gap_extend_b=gap_extend)


@pytest.fixture(scope="module")
def blosum():
    return make_params(BLOSUM62, 1.0)


@pytest.fixture(scope="module")
def unit():
    return make_params(IDENTITY, 1.0)


# ── guide_j_from_aligned ──────────────────────────────────────────────────────

class TestGuideJFromAligned:
    def test_all_matches(self):
        gj = nwgrad.guide_j_from_aligned("ACDE", "ACDE")
        assert gj == [0, 1, 2, 3, 4]

    def test_gap_in_b_shifts_guide_left(self):
        gj = nwgrad.guide_j_from_aligned("ACDE", "A-DE")
        assert gj == [0, 1, 1, 2, 3]

    def test_gap_in_a_shifts_guide_right(self):
        gj = nwgrad.guide_j_from_aligned("A-DE", "ACDE")
        assert gj == [0, 1, 3, 4]

    def test_leading_gap_in_b(self):
        gj = nwgrad.guide_j_from_aligned("ACDE", "--DE")
        assert gj == [0, 0, 0, 1, 2]

    def test_leading_gap_in_a(self):
        gj = nwgrad.guide_j_from_aligned("--DE", "ACDE")
        assert gj == [0, 3, 4]

    def test_trailing_gap_in_b(self):
        gj = nwgrad.guide_j_from_aligned("ACDE", "AC--")
        assert gj == [0, 1, 2, 2, 2]

    def test_trailing_gap_in_a(self):
        gj = nwgrad.guide_j_from_aligned("AC--", "ACDE")
        assert gj == [0, 1, 2]

    def test_length_is_m_plus_one(self):
        for a_aln, b_aln in [("ACDE", "A-DE"), ("A-DE", "ACDE"),
                              ("A---", "ACDE"), ("ACDE", "A---")]:
            gj = nwgrad.guide_j_from_aligned(a_aln, b_aln)
            m = sum(1 for c in a_aln if c != '-')
            assert len(gj) == m + 1, \
                f"len(guide_j) should be {m+1} for a_aligned={a_aln!r}"

    def test_last_entry_equals_n(self):
        for a_aln, b_aln in [("ACDE", "A-DE"), ("MMACDE", "---ADE"), ("A-DE", "ACDE")]:
            gj = nwgrad.guide_j_from_aligned(a_aln, b_aln)
            n = sum(1 for c in b_aln if c != '-')
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

    GAPPED_PAIRS = [
        ("ACDE",   "ADE",    "ACDE",   "A-DE"),
        ("ADE",    "ACDE",   "A-DE",   "ACDE"),
        ("ACDE",   "DE",     "ACDE",   "--DE"),
        ("DE",     "ACDE",   "--DE",   "ACDE"),
        ("ACDEFG", "DE",     "ACDEFG", "--DE--"),
        ("MMMADE", "ADE",    "MMMADE", "---ADE"),
    ]

    @pytest.mark.parametrize("a,b,aa,ab", GAPPED_PAIRS)
    def test_exact_guide_band0_matches_full(self, a, b, aa, ab, unit):
        full   = nwgrad.nw_score(a, b, unit)
        guided = nwgrad.nw_score(a, b, unit, band=0, aligned_a=aa, aligned_b=ab)
        assert guided == pytest.approx(full), \
            f"nw_score({a!r},{b!r}) guided={guided} full={full}"

    @pytest.mark.parametrize("a,b,aa,ab", GAPPED_PAIRS)
    def test_exact_guide_band1_matches_full(self, a, b, aa, ab, unit):
        full   = nwgrad.nw_score(a, b, unit)
        guided = nwgrad.nw_score(a, b, unit, band=1, aligned_a=aa, aligned_b=ab)
        assert guided == pytest.approx(full)

    @pytest.mark.parametrize("a,b,aa,ab", GAPPED_PAIRS)
    def test_blosum_exact_guide_matches_full(self, a, b, aa, ab, blosum):
        full   = nwgrad.nw_score(a, b, blosum)
        guided = nwgrad.nw_score(a, b, blosum, band=0, aligned_a=aa, aligned_b=ab)
        assert guided == pytest.approx(full)

    def test_off_diagonal_guide_with_band_finds_optimum(self, unit):
        a, b = "ACDE", "ADE"
        full = nwgrad.nw_score(a, b, unit)
        guided = nwgrad.nw_score(a, b, unit, band=2,
                                  aligned_a="ACDE", aligned_b="A-DE")
        assert guided == pytest.approx(full)

    def test_clearly_wrong_guide_narrow_band_misses_optimum(self, unit):
        a, b = "ACDEFG", "DE"
        full  = nwgrad.nw_score(a, b, unit)
        wrong = nwgrad.nw_score(a, b, unit, band=0,
                                 aligned_a="ACDEFG", aligned_b="DE----")
        assert wrong != pytest.approx(full)


# ── affine gap + non-diagonal guide ──────────────────────────────────────────

class TestNwAffineGuided:
    PAIRS = [
        ("ACDE",   "ADE",    "ACDE",   "A-DE"),
        ("MMMADE", "ADE",    "MMMADE", "---ADE"),
    ]

    @pytest.mark.parametrize("a,b,aa,ab", PAIRS)
    def test_exact_guide_matches_full(self, a, b, aa, ab):
        p = make_params(IDENTITY, 1.0, 10.0)
        full   = nwgrad.nw_score_affine(a, b, p)
        guided = nwgrad.nw_score_affine(a, b, p, band=0, aligned_a=aa, aligned_b=ab)
        assert guided == pytest.approx(full)


# ── sw_score with non-diagonal guide ─────────────────────────────────────────

class TestSwScoreGuided:
    def test_local_exact_guide_matches_full(self, unit):
        a, b = "MMMADE", "ADE"
        aa, ab = "MMMADE", "---ADE"
        full   = nwgrad.sw_score(a, b, unit)
        guided = nwgrad.sw_score(a, b, unit, band=1, aligned_a=aa, aligned_b=ab)
        assert guided == pytest.approx(full)


# ── Hard gradient with non-diagonal guide ────────────────────────────────────

class TestGradientGuided:
    def test_nw_grad_exact_guide_matches_full(self, unit):
        a, b   = "ACDE", "ADE"
        aa, ab = "ACDE", "A-DE"
        s_full,   g_full   = nwgrad.nw_grad(a, b, unit)
        s_guided, g_guided = nwgrad.nw_grad(a, b, unit, band=0,
                                             aligned_a=aa, aligned_b=ab)
        assert s_guided == pytest.approx(s_full)
        np.testing.assert_allclose(g_guided.matrix.to_matrix(),
                                    g_full.matrix.to_matrix(), atol=1e-12)

    def test_nw_grad_unequal_lengths_correct_guide(self, unit):
        a, b   = "ACDEFG", "DE"
        aa, ab = "ACDEFG", "--DE--"
        s_full,   g_full   = nwgrad.nw_grad(a, b, unit)
        s_guided, g_guided = nwgrad.nw_grad(a, b, unit, band=0,
                                             aligned_a=aa, aligned_b=ab)
        assert s_guided == pytest.approx(s_full)
        np.testing.assert_allclose(g_guided.matrix.to_matrix(),
                                    g_full.matrix.to_matrix(), atol=1e-12)

    def test_nw_grad_band1_around_guide(self, blosum):
        a, b = "PLEASANTLY", "MEANLY"
        s_full, _ = nwgrad.nw_grad(a, b, blosum)
        aligned_a = "PLEASANTLY"
        aligned_b = "-MEA--NELY"
        s_guided, _ = nwgrad.nw_grad(a, b, blosum, band=3,
                                      aligned_a=aligned_a, aligned_b=aligned_b)
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
        ba = nwgrad.BatchAligner(params=unit, gap_model="linear", mode="global",
                                  grad_mode="none", n_threads=1)
        seqs_a = [p[0] for p in self.PAIRS]
        seqs_b = [p[1] for p in self.PAIRS]
        aa     = [p[2] for p in self.PAIRS]
        ab     = [p[3] for p in self.PAIRS]

        result = ba.align(seqs_a, seqs_b, aligned_a=aa, aligned_b=ab)

        for i, (a, b, aligned_a, aligned_b) in enumerate(self.PAIRS):
            expected = nwgrad.nw_score(a, b, unit, band=0,
                                        aligned_a=aligned_a, aligned_b=aligned_b)
            assert result.scores[i] == pytest.approx(expected), \
                f"pair {i}: {a!r} vs {b!r}"

    def test_multithreaded_matches_single_threaded(self, unit):
        seqs_a = [p[0] for p in self.PAIRS]
        seqs_b = [p[1] for p in self.PAIRS]
        aa     = [p[2] for p in self.PAIRS]
        ab     = [p[3] for p in self.PAIRS]

        single = nwgrad.BatchAligner(params=unit, gap_model="linear", mode="global",
                                      grad_mode="hard", n_threads=1)
        multi  = nwgrad.BatchAligner(params=unit, gap_model="linear", mode="global",
                                      grad_mode="hard", n_threads=4)

        r1 = single.align(seqs_a, seqs_b, aligned_a=aa, aligned_b=ab)
        r2 = multi.align(seqs_a, seqs_b, aligned_a=aa, aligned_b=ab)

        np.testing.assert_allclose(r1.scores, r2.scores, atol=1e-12)
        np.testing.assert_allclose(r1.grad.matrix.to_matrix(),
                                    r2.grad.matrix.to_matrix(), atol=1e-12)

    def test_guided_batch_matches_full_batch(self, unit):
        seqs_a = [p[0] for p in self.PAIRS]
        seqs_b = [p[1] for p in self.PAIRS]
        aa     = [p[2] for p in self.PAIRS]
        ab     = [p[3] for p in self.PAIRS]

        ba_full   = nwgrad.BatchAligner(params=unit, gap_model="linear", mode="global",
                                         grad_mode="hard", n_threads=2)
        ba_guided = nwgrad.BatchAligner(params=unit, gap_model="linear", mode="global",
                                         grad_mode="hard", n_threads=2)

        r_full   = ba_full.align(seqs_a, seqs_b)
        r_guided = ba_guided.align(seqs_a, seqs_b, aligned_a=aa, aligned_b=ab)

        np.testing.assert_allclose(r_full.scores, r_guided.scores, atol=1e-12)
        np.testing.assert_allclose(r_full.grad.matrix.to_matrix(),
                                    r_guided.grad.matrix.to_matrix(), atol=1e-12)
