"""Step 5: hard-subgradient tests for nw_grad, sw_grad, nw_affine_grad, sw_affine_grad.

Each *_grad function returns (score, grad_20x20) where grad[i,j] counts how
many times AA_ORDER[i] was aligned to AA_ORDER[j] in the optimal traceback.
Gaps do not contribute.
"""

import numpy as np
import pytest
import nwgrad
from test_subst_matrix import BLOSUM62

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"

# ── Pure-Python reference implementations ────────────────────────────────────

def _aa_idx(c):
    return AA_ORDER.index(c)


def ref_nw_linear_grad(a, b, matrix, gap_extend):
    """Returns (score, Counter of (aa_a, aa_b) pairs)."""
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

    # Traceback
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
    """Convert 20×20 numpy grad array to {(aa_a, aa_b): count} skipping zeros."""
    d = {}
    for i in range(20):
        for j in range(20):
            v = g20[i, j]
            if v != 0:
                d[(AA_ORDER[i], AA_ORDER[j])] = v
    return d


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def blosum():
    return nwgrad.SubstMatrix(BLOSUM62)


# ── Test pairs ───────────────────────────────────────────────────────────────

PAIRS = [
    ("A",          "A"),
    ("ACDE",       "ACDE"),
    ("ACDE",       "ACDF"),
    ("A",          "AC"),
    ("PLEASANTLY", "MEANLY"),
    ("ACDEFGHIKL", "CDEFGHIKLM"),
    ("MADEEKLF",   "MADEEKLF"),
]

# ── Step 5a: NW linear gradient ──────────────────────────────────────────────

@pytest.mark.parametrize("a,b", PAIRS)
def test_nw_grad_score_matches(a, b, blosum):
    """Score returned by nw_grad must match nw_score."""
    gap = 1.0
    score_ref = nwgrad.nw_score(a, b, blosum, gap)
    score_g, _ = nwgrad.nw_grad(a, b, blosum, gap)
    assert score_g == pytest.approx(score_ref)


@pytest.mark.parametrize("a,b", PAIRS)
def test_nw_grad_counts_match_reference(a, b, blosum):
    gap = 1.0
    _, g = nwgrad.nw_grad(a, b, blosum, gap)
    ref_score, ref_pairs = ref_nw_linear_grad(a, b, blosum, gap)
    got_pairs = grad_to_dict(np.array(g))
    assert got_pairs == ref_pairs


def test_nw_grad_identity(blosum):
    """Aligning seq with itself: every character is a self-substitution."""
    seq = "ACDEF"
    _, g = nwgrad.nw_grad(seq, seq, blosum, 1.0)
    g = np.array(g)
    for c in seq:
        i = _aa_idx(c)
        assert g[i, i] == pytest.approx(1.0)
    assert g.sum() == pytest.approx(len(seq))


def test_nw_grad_all_gaps(blosum):
    """Aligning "" to a sequence: no substitutions → zero gradient."""
    _, g = nwgrad.nw_grad("", "ACDE", blosum, 1.0)
    assert np.array(g).sum() == pytest.approx(0.0)


def test_nw_grad_shape(blosum):
    _, g = nwgrad.nw_grad("ACDE", "ACDE", blosum, 1.0)
    assert np.array(g).shape == (20, 20)


# ── Step 5b: SW linear gradient ──────────────────────────────────────────────

@pytest.mark.parametrize("a,b", PAIRS)
def test_sw_grad_score_matches(a, b, blosum):
    gap = 1.0
    score_ref = nwgrad.sw_score(a, b, blosum, gap)
    score_g, _ = nwgrad.sw_grad(a, b, blosum, gap)
    assert score_g == pytest.approx(score_ref)


@pytest.mark.parametrize("a,b", PAIRS)
def test_sw_grad_counts_match_reference(a, b, blosum):
    gap = 1.0
    _, g = nwgrad.sw_grad(a, b, blosum, gap)
    _, ref_pairs = ref_sw_linear_grad(a, b, blosum, gap)
    got_pairs = grad_to_dict(np.array(g))
    assert got_pairs == ref_pairs


# ── Step 5c: NW affine gradient ──────────────────────────────────────────────

@pytest.mark.parametrize("a,b", PAIRS)
def test_nw_affine_grad_score_matches(a, b, blosum):
    score_ref = nwgrad.nw_score_affine(a, b, blosum, gap_open=11.0, gap_extend=1.0)
    score_g, _ = nwgrad.nw_affine_grad(a, b, blosum, gap_open=11.0, gap_extend=1.0)
    assert score_g == pytest.approx(score_ref)


@pytest.mark.parametrize("a,b", PAIRS)
def test_sw_affine_grad_score_matches(a, b, blosum):
    score_ref = nwgrad.sw_score_affine(a, b, blosum, gap_open=11.0, gap_extend=1.0)
    score_g, _ = nwgrad.sw_affine_grad(a, b, blosum, gap_open=11.0, gap_extend=1.0)
    assert score_g == pytest.approx(score_ref)


# ── Step 5d: Gradient sanity checks ──────────────────────────────────────────

def test_nw_affine_grad_identity(blosum):
    """Identity alignment: every position should be a self-pair."""
    seq = "ACDEFG"
    _, g = nwgrad.nw_affine_grad(seq, seq, blosum, gap_open=11.0, gap_extend=1.0)
    g = np.array(g)
    for c in seq:
        i = _aa_idx(c)
        assert g[i, i] == pytest.approx(1.0)
    assert g.sum() == pytest.approx(len(seq))


@pytest.mark.parametrize("fn,kw", [
    (nwgrad.nw_grad,        {"gap_extend": 1.0}),
    (nwgrad.sw_grad,        {"gap_extend": 1.0}),
    (nwgrad.nw_affine_grad, {"gap_open": 11.0, "gap_extend": 1.0}),
    (nwgrad.sw_affine_grad, {"gap_open": 11.0, "gap_extend": 1.0}),
])
def test_grad_nonnegative(fn, kw, blosum):
    """Gradient counts must be non-negative integers."""
    _, g = fn("PLEASANTLY", "MEANLY", blosum, **kw)
    g = np.array(g)
    assert (g >= 0).all()
    # counts should be whole numbers
    assert np.allclose(g, np.round(g))


@pytest.mark.parametrize("fn,kw", [
    (nwgrad.nw_grad,        {"gap_extend": 1.0}),
    (nwgrad.nw_affine_grad, {"gap_open": 11.0, "gap_extend": 1.0}),
])
def test_grad_sum_le_min_len(fn, kw, blosum):
    """Number of matched positions ≤ min(len(a), len(b))."""
    a, b = "PLEASANTLY", "MEANLY"
    _, g = fn(a, b, blosum, **kw)
    assert np.array(g).sum() <= min(len(a), len(b))
