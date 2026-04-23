import numpy as np
import pytest
import nwgrad
from test_subst_matrix import BLOSUM62


def make_params(gap_extend, gap_open=0.0, matrix_arr=None):
    arr = BLOSUM62 if matrix_arr is None else matrix_arr
    return nwgrad.AlignParams(arr, gap_open_a=gap_open, gap_extend_a=gap_extend,
                                   gap_open_b=gap_open, gap_extend_b=gap_extend)


# ── Pure-Python reference SW (affine gap) ────────────────────────────────────

def ref_sw_affine(a, b, matrix, gap_open, gap_extend):
    NEG_INF = float("-inf")
    m, n = len(a), len(b)

    M = [[NEG_INF] * (n + 1) for _ in range(m + 1)]
    X = [[NEG_INF] * (n + 1) for _ in range(m + 1)]
    Y = [[NEG_INF] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        M[i][0] = 0.0
    for j in range(n + 1):
        M[0][j] = 0.0
    best = 0.0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            diag  = max(M[i-1][j-1], X[i-1][j-1], Y[i-1][j-1])
            m_val = max(diag + matrix.score(a[i-1], b[j-1]), 0.0)
            x_val = max(
                M[i-1][j] - gap_open - gap_extend,
                X[i-1][j] - gap_extend,
                Y[i-1][j] - gap_open - gap_extend,
            )
            y_val = max(
                M[i][j-1] - gap_open - gap_extend,
                X[i][j-1] - gap_open - gap_extend,
                Y[i][j-1] - gap_extend,
            )
            M[i][j] = m_val
            X[i][j] = x_val
            Y[i][j] = y_val
            best = max(best, m_val, x_val, y_val)

    return best


@pytest.fixture(scope="module")
def blosum():
    return nwgrad.SubstMatrix(BLOSUM62)


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
    ("ACDE",       "NPQR"),
]

GAP_PARAMS = [(0.0, 1.0), (1.0, 0.5), (2.0, 1.0), (5.0, 2.0)]


@pytest.mark.parametrize("gap_open,gap_extend", GAP_PARAMS)
@pytest.mark.parametrize("a,b", PAIRS)
def test_matches_reference(a, b, gap_open, gap_extend, blosum):
    expected = ref_sw_affine(a, b, blosum, gap_open, gap_extend)
    got      = nwgrad.sw_score_affine(a, b, make_params(gap_extend, gap_open))
    assert got == pytest.approx(expected, abs=1e-9), (
        f"sw_score_affine({a!r},{b!r}, open={gap_open}, ext={gap_extend}): "
        f"got {got}, expected {expected}"
    )


@pytest.mark.parametrize("a,b", PAIRS)
def test_score_nonnegative(a, b):
    assert nwgrad.sw_score_affine(a, b, make_params(0.5, 1.0)) >= 0.0


@pytest.mark.parametrize("a,b", PAIRS)
def test_symmetric(a, b):
    p = make_params(0.5, 1.0)
    assert nwgrad.sw_score_affine(a, b, p) == pytest.approx(
        nwgrad.sw_score_affine(b, a, p), abs=1e-9
    )


@pytest.mark.parametrize("a,b", PAIRS)
def test_local_ge_global(a, b):
    p = make_params(0.5, 1.0)
    assert nwgrad.sw_score_affine(a, b, p) >= nwgrad.nw_score_affine(a, b, p)


@pytest.mark.parametrize("a,b", PAIRS)
def test_affine_with_zero_open_matches_linear_sw(a, b):
    gap = 1.0
    assert nwgrad.sw_score_affine(a, b, make_params(gap, 0.0)) == pytest.approx(
        nwgrad.sw_score(a, b, make_params(gap)), abs=1e-9
    )


def test_empty_vs_empty():
    assert nwgrad.sw_score_affine("", "", make_params(0.5, 1.0)) == pytest.approx(0.0)


def test_empty_vs_seq():
    assert nwgrad.sw_score_affine("", "ACDE", make_params(0.5, 1.0)) == pytest.approx(0.0)
    assert nwgrad.sw_score_affine("ACDE", "", make_params(0.5, 1.0)) == pytest.approx(0.0)


def test_no_positive_scoring_pairs():
    arr = np.full((20, 20), -10.0)
    assert nwgrad.sw_score_affine("ACDE", "ACDE", make_params(0.5, 1.0, arr)) == pytest.approx(0.0)
