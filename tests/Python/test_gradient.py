"""Hard-subgradient tests for nw_grad, sw_grad, nw_affine_grad, sw_affine_grad."""

import numpy as np
import pytest
import nwgrad
from test_subst_matrix import BLOSUM62

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"


def make_params(gap_extend, gap_open=0.0, matrix_arr=None):
    arr = BLOSUM62 if matrix_arr is None else matrix_arr
    return nwgrad.AlignParams(arr, gap_open_a=gap_open, gap_extend_a=gap_extend,
                                   gap_open_b=gap_open, gap_extend_b=gap_extend)


def grad_matrix(g):
    """Return the 20×20 substitution gradient as a numpy array."""
    return g.matrix.to_matrix()


def _aa_idx(c):
    return AA_ORDER.index(c)


# ── Pure-Python reference implementations ────────────────────────────────────

def ref_nw_linear_grad(a, b, matrix, gap_extend):
    m, n = len(a), len(b)
    H = [[0.0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        H[i][0] = -i * gap_extend
    for j in range(n + 1):
        H[0][j] = -j * gap_extend
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            H[i][j] = max(
                H[i-1][j-1] + matrix.score(a[i-1], b[j-1]),
                H[i-1][j]   - gap_extend,
                H[i][j-1]   - gap_extend,
            )
    pairs = {}
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and H[i][j] == H[i-1][j-1] + matrix.score(a[i-1], b[j-1]):
            key = (a[i-1], b[j-1])
            pairs[key] = pairs.get(key, 0) + 1
            i -= 1; j -= 1
        elif i > 0 and H[i][j] == H[i-1][j] - gap_extend:
            i -= 1
        else:
            j -= 1
    return H[m][n], pairs


def ref_sw_linear_grad(a, b, matrix, gap_extend):
    m, n = len(a), len(b)
    H = [[0.0] * (n + 1) for _ in range(m + 1)]
    best_score = 0.0
    best_i = best_j = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            v = max(
                H[i-1][j-1] + matrix.score(a[i-1], b[j-1]),
                H[i-1][j]   - gap_extend,
                H[i][j-1]   - gap_extend,
                0.0,
            )
            H[i][j] = v
            if v > best_score:
                best_score = v
                best_i, best_j = i, j
    pairs = {}
    i, j = best_i, best_j
    while H[i][j] > 0:
        if i > 0 and j > 0 and H[i][j] == H[i-1][j-1] + matrix.score(a[i-1], b[j-1]):
            key = (a[i-1], b[j-1])
            pairs[key] = pairs.get(key, 0) + 1
            i -= 1; j -= 1
        elif i > 0 and H[i][j] == H[i-1][j] - gap_extend:
            i -= 1
        else:
            j -= 1
    return best_score, pairs


def grad_to_dict(g20):
    d = {}
    for i in range(20):
        for j in range(20):
            v = g20[i, j]
            if v != 0:
                d[(AA_ORDER[i], AA_ORDER[j])] = v
    return d


@pytest.fixture
def blosum():
    # Function-scoped on purpose: a module-scoped SubstMatrix would be retained by
    # pytest for the whole session and trip nanobind's leak checker at shutdown.
    return nwgrad.SubstMatrix(BLOSUM62)


PAIRS = [
    ("A",          "A"),
    ("ACDE",       "ACDE"),
    ("ACDE",       "ACDF"),
    ("A",          "AC"),
    ("PLEASANTLY", "MEANLY"),
    ("ACDEFGHIKL", "CDEFGHIKLM"),
    ("MADEEKLF",   "MADEEKLF"),
]

# ── NW linear gradient ───────────────────────────────────────────────────────

@pytest.mark.parametrize("a,b", PAIRS)
def test_nw_grad_score_matches(a, b):
    gap = 1.0
    score_ref = nwgrad.nw_score(a, b, make_params(gap))
    score_g, _ = nwgrad.nw_grad(a, b, make_params(gap))
    assert score_g == pytest.approx(score_ref)


@pytest.mark.parametrize("a,b", PAIRS)
def test_nw_grad_counts_match_reference(a, b, blosum):
    gap = 1.0
    _, g = nwgrad.nw_grad(a, b, make_params(gap))
    ref_score, ref_pairs = ref_nw_linear_grad(a, b, blosum, gap)
    got_pairs = grad_to_dict(grad_matrix(g))
    assert got_pairs == ref_pairs


def test_nw_grad_identity():
    seq = "ACDEF"
    _, g = nwgrad.nw_grad(seq, seq, make_params(1.0))
    gm = grad_matrix(g)
    for c in seq:
        i = _aa_idx(c)
        assert gm[i, i] == pytest.approx(1.0)
    assert gm.sum() == pytest.approx(len(seq))


def test_nw_grad_all_gaps():
    _, g = nwgrad.nw_grad("", "ACDE", make_params(1.0))
    assert grad_matrix(g).sum() == pytest.approx(0.0)


def test_nw_grad_shape():
    _, g = nwgrad.nw_grad("ACDE", "ACDE", make_params(1.0))
    assert grad_matrix(g).shape == (20, 20)


# ── SW linear gradient ───────────────────────────────────────────────────────

@pytest.mark.parametrize("a,b", PAIRS)
def test_sw_grad_score_matches(a, b):
    gap = 1.0
    score_ref = nwgrad.sw_score(a, b, make_params(gap))
    score_g, _ = nwgrad.sw_grad(a, b, make_params(gap))
    assert score_g == pytest.approx(score_ref)


@pytest.mark.parametrize("a,b", PAIRS)
def test_sw_grad_counts_match_reference(a, b, blosum):
    gap = 1.0
    _, g = nwgrad.sw_grad(a, b, make_params(gap))
    _, ref_pairs = ref_sw_linear_grad(a, b, blosum, gap)
    got_pairs = grad_to_dict(grad_matrix(g))
    assert got_pairs == ref_pairs


# ── NW affine gradient ───────────────────────────────────────────────────────

@pytest.mark.parametrize("a,b", PAIRS)
def test_nw_affine_grad_score_matches(a, b):
    score_ref = nwgrad.nw_score_affine(a, b, make_params(1.0, 11.0))
    score_g, _ = nwgrad.nw_affine_grad(a, b, make_params(1.0, 11.0))
    assert score_g == pytest.approx(score_ref)


@pytest.mark.parametrize("a,b", PAIRS)
def test_sw_affine_grad_score_matches(a, b):
    score_ref = nwgrad.sw_score_affine(a, b, make_params(1.0, 11.0))
    score_g, _ = nwgrad.sw_affine_grad(a, b, make_params(1.0, 11.0))
    assert score_g == pytest.approx(score_ref)


# ── Gradient sanity checks ───────────────────────────────────────────────────

def test_nw_affine_grad_identity():
    seq = "ACDEFG"
    _, g = nwgrad.nw_affine_grad(seq, seq, make_params(1.0, 11.0))
    gm = grad_matrix(g)
    for c in seq:
        i = _aa_idx(c)
        assert gm[i, i] == pytest.approx(1.0)
    assert gm.sum() == pytest.approx(len(seq))


# Parametrize over the grad-function name and gap args rather than prebuilt
# AlignParams objects: passing bound instances as parametrize values makes pytest
# retain them for the whole session, tripping nanobind's leak checker at shutdown.
@pytest.mark.parametrize("fn_name,gap_open", [
    ("nw_grad",        0.0),
    ("sw_grad",        0.0),
    ("nw_affine_grad", 11.0),
    ("sw_affine_grad", 11.0),
])
def test_grad_nonnegative(fn_name, gap_open):
    _, g = getattr(nwgrad, fn_name)("PLEASANTLY", "MEANLY", make_params(1.0, gap_open))
    gm = grad_matrix(g)
    assert (gm >= 0).all()
    assert np.allclose(gm, np.round(gm))


@pytest.mark.parametrize("fn_name,gap_open", [
    ("nw_grad",        0.0),
    ("nw_affine_grad", 11.0),
])
def test_grad_sum_le_min_len(fn_name, gap_open):
    a, b = "PLEASANTLY", "MEANLY"
    _, g = getattr(nwgrad, fn_name)(a, b, make_params(1.0, gap_open))
    assert grad_matrix(g).sum() <= min(len(a), len(b))
