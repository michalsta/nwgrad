import numpy as np
import pytest
import nwgrad
from test_subst_matrix import BLOSUM62

# ── Pure-Python reference SW (linear gap) ────────────────────────────────────

def ref_sw_linear(a, b, matrix, gap_extend):
    """Reference Smith-Waterman with linear gap penalty."""
    m, n = len(a), len(b)
    H = [[0.0] * (n + 1) for _ in range(m + 1)]
    # Boundary rows/cols remain 0 for local alignment.
    best = 0.0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            v = max(
                0.0,
                H[i-1][j-1] + matrix.score(a[i-1], b[j-1]),
                H[i-1][j]   - gap_extend,
                H[i][j-1]   - gap_extend,
            )
            H[i][j] = v
            if v > best:
                best = v
    return best


@pytest.fixture(scope="module")
def blosum():
    return nwgrad.SubstMatrix(BLOSUM62)


# ── Correctness vs. reference ─────────────────────────────────────────────────

PAIRS = [
    ("A",          "A"),
    ("A",          "C"),
    ("ACDE",       "ACDE"),
    ("ACDE",       "ACDF"),
    ("A",          "AC"),
    ("ACDEFG",     "ACDE"),
    ("PLEASANTLY", "MEANLY"),
    ("ACDEFGHIKL", "CDEFGHIKLM"),
    ("MADEEKLF",   "MADEEKLF"),
    # sequences with no positive-scoring alignment
    ("ACDE",       "NPQR"),
]

@pytest.mark.parametrize("a,b", PAIRS)
def test_matches_reference(a, b, blosum):
    gap = 1.0
    expected = ref_sw_linear(a, b, blosum, gap)
    got = nwgrad.sw_score(a, b, blosum, gap)
    assert got == pytest.approx(expected), (
        f"sw_score({a!r}, {b!r}) = {got}, expected {expected}"
    )


# ── Structural properties ─────────────────────────────────────────────────────

@pytest.mark.parametrize("a,b", PAIRS)
def test_score_nonnegative(a, b, blosum):
    """Local alignment score is always >= 0."""
    assert nwgrad.sw_score(a, b, blosum, 1.0) >= 0.0


@pytest.mark.parametrize("a,b", PAIRS)
def test_symmetric(a, b, blosum):
    """sw_score is symmetric."""
    gap = 1.0
    assert nwgrad.sw_score(a, b, blosum, gap) == pytest.approx(
        nwgrad.sw_score(b, a, blosum, gap)
    )


@pytest.mark.parametrize("a,b", PAIRS)
def test_local_ge_global(a, b, blosum):
    """Local score >= global score (local can always ignore bad flanking regions)."""
    gap = 1.0
    assert nwgrad.sw_score(a, b, blosum, gap) >= nwgrad.nw_score(a, b, blosum, gap)


# ── Identity and single-char cases ───────────────────────────────────────────

def test_identity_score(blosum):
    """Aligning a sequence with itself: local == global (no benefit from trimming)."""
    seq = "ACDEFG"
    AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
    expected = sum(BLOSUM62[AA_ORDER.index(c), AA_ORDER.index(c)] for c in seq)
    assert nwgrad.sw_score(seq, seq, blosum, 1.0) == pytest.approx(expected)


def test_single_char_same(blosum):
    assert nwgrad.sw_score("A", "A", blosum, 1.0) == pytest.approx(4.0)


def test_single_char_diff_nonneg(blosum):
    """score(A, C) = 0 (BLOSUM62 A-C = 0, and local can choose empty alignment)."""
    assert nwgrad.sw_score("A", "C", blosum, 1.0) == pytest.approx(0.0)


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_empty_vs_empty(blosum):
    assert nwgrad.sw_score("", "", blosum, 1.0) == pytest.approx(0.0)


def test_empty_vs_seq(blosum):
    """Aligning empty string to anything: local score is 0 (empty alignment)."""
    seq = "ACDE"
    assert nwgrad.sw_score("", seq, blosum, 1.0) == pytest.approx(0.0)
    assert nwgrad.sw_score(seq, "", blosum, 1.0) == pytest.approx(0.0)


def test_no_positive_scoring_pairs(blosum):
    """When all substitution scores are negative, SW returns 0."""
    # Build a matrix of all -10s
    arr = np.full((20, 20), -10.0)
    bad_mat = nwgrad.SubstMatrix(arr)
    assert nwgrad.sw_score("ACDE", "ACDE", bad_mat, 1.0) == pytest.approx(0.0)
