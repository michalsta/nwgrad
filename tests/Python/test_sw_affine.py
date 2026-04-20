import numpy as np
import pytest
import nwgrad
from test_blosum import BLOSUM62

# ── Pure-Python reference SW (affine gap) ────────────────────────────────────

def ref_sw_affine(a, b, matrix, gap_open, gap_extend):
    """
    Reference Smith-Waterman with affine gap penalty.
    gap_cost(k) = gap_open + gap_extend * k
    """
    NEG_INF = float("-inf")
    m, n = len(a), len(b)

    M = [[NEG_INF] * (n + 1) for _ in range(m + 1)]
    X = [[NEG_INF] * (n + 1) for _ in range(m + 1)]
    Y = [[NEG_INF] * (n + 1) for _ in range(m + 1)]

    # Boundary: fresh alignment can start anywhere along each sequence.
    for i in range(m + 1):
        M[i][0] = 0.0
    for j in range(n + 1):
        M[0][j] = 0.0
    best = 0.0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            diag  = max(M[i-1][j-1], X[i-1][j-1], Y[i-1][j-1])
            m_val = max(diag + matrix.score(a[i-1], b[j-1]), 0.0)

            # Only M gets the zero floor; X/Y may be negative.
            x_val = max(
                M[i-1][j] - gap_open - gap_extend,
                X[i-1][j] - gap_extend,
                Y[i-1][j] - gap_open - gap_extend,  # Y→X switch
            )
            y_val = max(
                M[i][j-1] - gap_open - gap_extend,
                X[i][j-1] - gap_open - gap_extend,  # X→Y switch
                Y[i][j-1] - gap_extend,
            )

            M[i][j] = m_val
            X[i][j] = x_val
            Y[i][j] = y_val
            best = max(best, m_val, x_val, y_val)

    return best


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
    ("ACDE",       "NPQR"),   # no positive-scoring alignment
]

GAP_PARAMS = [(0.0, 1.0), (1.0, 0.5), (2.0, 1.0), (5.0, 2.0)]


@pytest.mark.parametrize("gap_open,gap_extend", GAP_PARAMS)
@pytest.mark.parametrize("a,b", PAIRS)
def test_matches_reference(a, b, gap_open, gap_extend, blosum):
    expected = ref_sw_affine(a, b, blosum, gap_open, gap_extend)
    got      = nwgrad.sw_score_affine(a, b, blosum, gap_open, gap_extend)
    assert got == pytest.approx(expected, abs=1e-9), (
        f"sw_score_affine({a!r},{b!r}, open={gap_open}, ext={gap_extend}): "
        f"got {got}, expected {expected}"
    )


# ── Structural properties ─────────────────────────────────────────────────────

@pytest.mark.parametrize("a,b", PAIRS)
def test_score_nonnegative(a, b, blosum):
    assert nwgrad.sw_score_affine(a, b, blosum, 1.0, 0.5) >= 0.0


@pytest.mark.parametrize("a,b", PAIRS)
def test_symmetric(a, b, blosum):
    gap_open, gap_extend = 1.0, 0.5
    assert nwgrad.sw_score_affine(a, b, blosum, gap_open, gap_extend) == pytest.approx(
        nwgrad.sw_score_affine(b, a, blosum, gap_open, gap_extend), abs=1e-9
    )


@pytest.mark.parametrize("a,b", PAIRS)
def test_local_ge_global(a, b, blosum):
    gap_open, gap_extend = 1.0, 0.5
    assert nwgrad.sw_score_affine(a, b, blosum, gap_open, gap_extend) >= \
           nwgrad.nw_score_affine(a, b, blosum, gap_open, gap_extend)


@pytest.mark.parametrize("a,b", PAIRS)
def test_affine_with_zero_open_matches_linear_sw(a, b, blosum):
    """Affine with gap_open=0 should equal linear SW."""
    gap = 1.0
    assert nwgrad.sw_score_affine(a, b, blosum, 0.0, gap) == pytest.approx(
        nwgrad.sw_score(a, b, blosum, gap), abs=1e-9
    )


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_empty_vs_empty(blosum):
    assert nwgrad.sw_score_affine("", "", blosum, 1.0, 0.5) == pytest.approx(0.0)


def test_empty_vs_seq(blosum):
    """Local alignment of empty string is always 0."""
    assert nwgrad.sw_score_affine("", "ACDE", blosum, 1.0, 0.5) == pytest.approx(0.0)
    assert nwgrad.sw_score_affine("ACDE", "", blosum, 1.0, 0.5) == pytest.approx(0.0)


def test_no_positive_scoring_pairs(blosum):
    """All-negative matrix → SW returns 0."""
    arr = np.full((20, 20), -10.0)
    bad_mat = nwgrad.BlosumMatrix(arr)
    assert nwgrad.sw_score_affine("ACDE", "ACDE", bad_mat, 1.0, 0.5) == pytest.approx(0.0)
