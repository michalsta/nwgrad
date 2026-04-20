import numpy as np
import pytest
import nwgrad
from test_blosum import BLOSUM62

# ── Pure-Python reference NW (linear gap) ────────────────────────────────────

def ref_nw_linear(a, b, matrix, gap_extend):
    """Reference Needleman-Wunsch with linear gap penalty."""
    m, n = len(a), len(b)
    H = [[0.0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        H[i][0] = -i * gap_extend
    for j in range(n + 1):
        H[0][j] = -j * gap_extend
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            diag = H[i-1][j-1] + matrix.score(a[i-1], b[j-1])
            up   = H[i-1][j]   - gap_extend
            left = H[i][j-1]   - gap_extend
            H[i][j] = max(diag, up, left)
    return H[m][n]


@pytest.fixture(scope="module")
def blosum():
    return nwgrad.BlosumMatrix(BLOSUM62)


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
]

@pytest.mark.parametrize("a,b", PAIRS)
def test_matches_reference(a, b, blosum):
    gap = 1.0
    expected = ref_nw_linear(a, b, blosum, gap)
    got = nwgrad.nw_score(a, b, blosum, gap)
    assert got == pytest.approx(expected), f"nw_score({a!r}, {b!r}) = {got}, expected {expected}"


@pytest.mark.parametrize("a,b", PAIRS)
def test_symmetric(a, b, blosum):
    """Reversing both sequences gives the same score."""
    gap = 1.0
    assert nwgrad.nw_score(a, b, blosum, gap) == pytest.approx(
        nwgrad.nw_score(b, a, blosum, gap)
    )


# ── Identity alignment ────────────────────────────────────────────────────────

def test_identity_score(blosum):
    """Aligning a sequence with itself yields the sum of its self-substitution scores."""
    seq = "ACDEFG"
    AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
    expected = sum(BLOSUM62[AA_ORDER.index(c), AA_ORDER.index(c)] for c in seq)
    assert nwgrad.nw_score(seq, seq, blosum, 1.0) == pytest.approx(expected)


# ── Gap penalty ───────────────────────────────────────────────────────────────

def test_gap_penalty_scaling(blosum):
    """Higher gap penalty should reduce (or equal) the score for mismatched-length seqs."""
    a, b = "ACDEFG", "ACDE"
    score_low  = nwgrad.nw_score(a, b, blosum, gap_extend=0.1)
    score_high = nwgrad.nw_score(a, b, blosum, gap_extend=10.0)
    assert score_low >= score_high


def test_single_char_same(blosum):
    assert nwgrad.nw_score("A", "A", blosum, 1.0) == pytest.approx(4.0)


def test_single_char_diff(blosum):
    # max(score(A,C)=0, gap+gap=-2) = 0
    assert nwgrad.nw_score("A", "C", blosum, 1.0) == pytest.approx(0.0)


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_empty_vs_empty(blosum):
    assert nwgrad.nw_score("", "", blosum, 1.0) == pytest.approx(0.0)


def test_empty_vs_seq(blosum):
    """Aligning empty string to a sequence of length n costs n * gap_extend."""
    seq = "ACDE"
    assert nwgrad.nw_score("", seq, blosum, 2.0) == pytest.approx(-len(seq) * 2.0)
    assert nwgrad.nw_score(seq, "", blosum, 2.0) == pytest.approx(-len(seq) * 2.0)
