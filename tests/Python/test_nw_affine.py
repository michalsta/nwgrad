import numpy as np
import pytest
import nwgrad
from test_subst_matrix import BLOSUM62


def make_params(gap_extend, gap_open=0.0):
    return nwgrad.AlignParams(BLOSUM62, gap_open_a=gap_open, gap_extend_a=gap_extend,
                                        gap_open_b=gap_open, gap_extend_b=gap_extend)


# ── Pure-Python reference NW (affine gap) ────────────────────────────────────

def ref_nw_affine(a, b, matrix, gap_open, gap_extend):
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
                Y[i-1][j] - gap_open - gap_extend,
            )
            Y[i][j] = max(
                M[i][j-1] - gap_open - gap_extend,
                X[i][j-1] - gap_open - gap_extend,
                Y[i][j-1] - gap_extend,
            )

    return max(M[m][n], X[m][n], Y[m][n])


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
]

GAP_PARAMS = [(0.0, 1.0), (1.0, 0.5), (2.0, 1.0), (5.0, 2.0)]


@pytest.mark.parametrize("gap_open,gap_extend", GAP_PARAMS)
@pytest.mark.parametrize("a,b", PAIRS)
def test_matches_reference(a, b, gap_open, gap_extend, blosum):
    expected = ref_nw_affine(a, b, blosum, gap_open, gap_extend)
    got      = nwgrad.nw_score_affine(a, b, make_params(gap_extend, gap_open))
    assert got == pytest.approx(expected, abs=1e-9), (
        f"nw_score_affine({a!r},{b!r}, open={gap_open}, ext={gap_extend}): "
        f"got {got}, expected {expected}"
    )


@pytest.mark.parametrize("a,b", PAIRS)
def test_affine_with_zero_open_matches_linear(a, b):
    gap = 1.0
    assert nwgrad.nw_score_affine(a, b, make_params(gap, 0.0)) == pytest.approx(
        nwgrad.nw_score(a, b, make_params(gap)), abs=1e-9
    )


@pytest.mark.parametrize("a,b", PAIRS)
def test_symmetric(a, b):
    gap_open, gap_extend = 1.0, 0.5
    assert nwgrad.nw_score_affine(a, b, make_params(gap_extend, gap_open)) == pytest.approx(
        nwgrad.nw_score_affine(b, a, make_params(gap_extend, gap_open)), abs=1e-9
    )


@pytest.mark.parametrize("a,b", PAIRS)
def test_higher_open_lowers_score(a, b):
    gap_extend = 0.5
    score_low  = nwgrad.nw_score_affine(a, b, make_params(gap_extend, 0.0))
    score_high = nwgrad.nw_score_affine(a, b, make_params(gap_extend, 5.0))
    assert score_low >= score_high


def test_identity_score(blosum):
    seq = "ACDEFG"
    AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
    expected = sum(BLOSUM62[AA_ORDER.index(c), AA_ORDER.index(c)] for c in seq)
    assert nwgrad.nw_score_affine(seq, seq, make_params(0.5, 1.0)) == pytest.approx(expected)


def test_empty_vs_empty():
    assert nwgrad.nw_score_affine("", "", make_params(0.5, 1.0)) == pytest.approx(0.0)


def test_empty_vs_seq():
    seq = "ACDE"
    expected = -(1.0 + len(seq) * 0.5)
    assert nwgrad.nw_score_affine("", seq, make_params(0.5, 1.0)) == pytest.approx(expected)
    assert nwgrad.nw_score_affine(seq, "", make_params(0.5, 1.0)) == pytest.approx(expected)
