import numpy as np
import pytest
import nwgrad
from test_subst_matrix import BLOSUM62


def make_params(gap_extend, gap_open=0.0, matrix_arr=None):
    arr = BLOSUM62 if matrix_arr is None else matrix_arr
    return nwgrad.AlignParams(arr, gap_open_a=gap_open, gap_extend_a=gap_extend,
                                   gap_open_b=gap_open, gap_extend_b=gap_extend)


# ── Pure-Python reference SW (linear gap) ────────────────────────────────────

def ref_sw_linear(a, b, matrix, gap_extend):
    """Reference Smith-Waterman with linear gap penalty."""
    m, n = len(a), len(b)
    H = [[0.0] * (n + 1) for _ in range(m + 1)]
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
    ("ACDE",       "NPQR"),
]

@pytest.mark.parametrize("a,b", PAIRS)
def test_matches_reference(a, b, blosum):
    gap = 1.0
    expected = ref_sw_linear(a, b, blosum, gap)
    got = nwgrad.sw_score(a, b, make_params(gap))
    assert got == pytest.approx(expected), (
        f"sw_score({a!r}, {b!r}) = {got}, expected {expected}"
    )


@pytest.mark.parametrize("a,b", PAIRS)
def test_score_nonnegative(a, b):
    assert nwgrad.sw_score(a, b, make_params(1.0)) >= 0.0


@pytest.mark.parametrize("a,b", PAIRS)
def test_symmetric(a, b):
    gap = 1.0
    assert nwgrad.sw_score(a, b, make_params(gap)) == pytest.approx(
        nwgrad.sw_score(b, a, make_params(gap))
    )


@pytest.mark.parametrize("a,b", PAIRS)
def test_local_ge_global(a, b):
    gap = 1.0
    assert nwgrad.sw_score(a, b, make_params(gap)) >= nwgrad.nw_score(a, b, make_params(gap))


def test_identity_score(blosum):
    seq = "ACDEFG"
    AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
    expected = sum(BLOSUM62[AA_ORDER.index(c), AA_ORDER.index(c)] for c in seq)
    assert nwgrad.sw_score(seq, seq, make_params(1.0)) == pytest.approx(expected)


def test_single_char_same():
    assert nwgrad.sw_score("A", "A", make_params(1.0)) == pytest.approx(4.0)


def test_single_char_diff_nonneg():
    assert nwgrad.sw_score("A", "C", make_params(1.0)) == pytest.approx(0.0)


def test_empty_vs_empty():
    assert nwgrad.sw_score("", "", make_params(1.0)) == pytest.approx(0.0)


def test_empty_vs_seq():
    seq = "ACDE"
    assert nwgrad.sw_score("", seq, make_params(1.0)) == pytest.approx(0.0)
    assert nwgrad.sw_score(seq, "", make_params(1.0)) == pytest.approx(0.0)


def test_no_positive_scoring_pairs():
    arr = np.full((20, 20), -10.0)
    assert nwgrad.sw_score("ACDE", "ACDE", make_params(1.0, matrix_arr=arr)) == pytest.approx(0.0)
