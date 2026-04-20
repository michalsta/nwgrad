import numpy as np
import pytest
import nwgrad
from test_blosum import BLOSUM62

# ── Pure-Python reference NW (affine gap) ────────────────────────────────────

def ref_nw_affine(a, b, matrix, gap_open, gap_extend):
    """
    Reference Needleman-Wunsch with affine gap penalty.
    gap_cost(k) = gap_open + gap_extend * k
    Tables: M (match), X (gap in b / consuming a), Y (gap in a / consuming b).
    """
    NEG_INF = float("-inf")
    m, n = len(a), len(b)

    M = [[NEG_INF] * (n + 1) for _ in range(m + 1)]
    X = [[NEG_INF] * (n + 1) for _ in range(m + 1)]
    Y = [[NEG_INF] * (n + 1) for _ in range(m + 1)]

    M[0][0] = 0.0
    for i in range(1, m + 1):
        X[i][0] = -(gap_open + i * gap_extend)
    for j in range(1, n + 1):
        Y[0][j] = -(gap_open + j * gap_extend)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            diag  = max(M[i-1][j-1], X[i-1][j-1], Y[i-1][j-1])
            M[i][j] = diag + matrix.score(a[i-1], b[j-1])

            X[i][j] = max(
                M[i-1][j] - gap_open - gap_extend,
                X[i-1][j] - gap_extend,
                Y[i-1][j] - gap_open - gap_extend,  # Y→X switch
            )
            Y[i][j] = max(
                M[i][j-1] - gap_open - gap_extend,
                X[i][j-1] - gap_open - gap_extend,  # X→Y switch
                Y[i][j-1] - gap_extend,
            )

    return max(M[m][n], X[m][n], Y[m][n])


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

GAP_PARAMS = [(0.0, 1.0), (1.0, 0.5), (2.0, 1.0), (5.0, 2.0)]


@pytest.mark.parametrize("gap_open,gap_extend", GAP_PARAMS)
@pytest.mark.parametrize("a,b", PAIRS)
def test_matches_reference(a, b, gap_open, gap_extend, blosum):
    expected = ref_nw_affine(a, b, blosum, gap_open, gap_extend)
    got      = nwgrad.nw_score_affine(a, b, blosum, gap_open, gap_extend)
    assert got == pytest.approx(expected, abs=1e-9), (
        f"nw_score_affine({a!r},{b!r}, open={gap_open}, ext={gap_extend}): "
        f"got {got}, expected {expected}"
    )


# ── Linear is a special case of affine (gap_open = 0) ────────────────────────

@pytest.mark.parametrize("a,b", PAIRS)
def test_affine_with_zero_open_matches_linear(a, b, blosum):
    """Affine with gap_open=0 should equal linear."""
    gap = 1.0
    assert nwgrad.nw_score_affine(a, b, blosum, 0.0, gap) == pytest.approx(
        nwgrad.nw_score(a, b, blosum, gap), abs=1e-9
    )


# ── Structural properties ─────────────────────────────────────────────────────

@pytest.mark.parametrize("a,b", PAIRS)
def test_symmetric(a, b, blosum):
    gap_open, gap_extend = 1.0, 0.5
    assert nwgrad.nw_score_affine(a, b, blosum, gap_open, gap_extend) == pytest.approx(
        nwgrad.nw_score_affine(b, a, blosum, gap_open, gap_extend), abs=1e-9
    )


@pytest.mark.parametrize("a,b", PAIRS)
def test_higher_open_lowers_score(a, b, blosum):
    """Higher gap_open should not increase the score (gaps become more expensive)."""
    gap_extend = 0.5
    score_low  = nwgrad.nw_score_affine(a, b, blosum, 0.0, gap_extend)
    score_high = nwgrad.nw_score_affine(a, b, blosum, 5.0, gap_extend)
    assert score_low >= score_high


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_identity_score(blosum):
    """Aligning a sequence with itself: sum of diagonal BLOSUM62 scores."""
    seq = "ACDEFG"
    AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
    expected = sum(BLOSUM62[AA_ORDER.index(c), AA_ORDER.index(c)] for c in seq)
    assert nwgrad.nw_score_affine(seq, seq, blosum, 1.0, 0.5) == pytest.approx(expected)


def test_empty_vs_empty(blosum):
    assert nwgrad.nw_score_affine("", "", blosum, 1.0, 0.5) == pytest.approx(0.0)


def test_empty_vs_seq(blosum):
    """Empty vs length-n sequence: costs gap_open + n * gap_extend."""
    seq = "ACDE"
    expected = -(1.0 + len(seq) * 0.5)
    assert nwgrad.nw_score_affine("", seq, blosum, 1.0, 0.5) == pytest.approx(expected)
    assert nwgrad.nw_score_affine(seq, "", blosum, 1.0, 0.5) == pytest.approx(expected)
