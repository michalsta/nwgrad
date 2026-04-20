"""Tests for the soft (differentiable) alignment gradient.

The soft gradient replaces max with log-sum-exp, so:
  1. log_Z >= hard score  (lse >= max)
  2. grad.sum() <= len(a)  (expected counts are fractional, <=1 total per position)
  3. Numerical gradient check: d(log_Z)/d(s(a,b)) == grad[a,b]
     verified via central finite differences on single entries.
  4. As the matrix is scaled by a large factor (low temperature), the soft
     score approaches the hard score and the soft gradient approaches the hard
     subgradient on sequences with a unique optimal alignment.
  5. BatchAligner with grad_mode="soft" aggregates correctly.
"""

import numpy as np
import pytest
import nwgrad
from test_blosum import BLOSUM62

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"

EPS = 1e-5


# ── fixtures & helpers ─────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def blosum():
    return nwgrad.BlosumMatrix(BLOSUM62)


def aa_idx(c):
    return AA_ORDER.index(c)


def make_mat(arr20):
    return nwgrad.BlosumMatrix(np.asarray(arr20, dtype=np.float64))


def perturb_entry(mat20, i, j, eps):
    """Return BlosumMatrix with the entry controlling s(AA[i],AA[j]) perturbed by eps.

    BlosumMatrix uses the lower-triangle element (row >= col) as the canonical
    value; both mat[a][b] and mat[b][a] are set from it.  We therefore perturb
    only m[max(i,j), min(i,j)] to avoid double-counting on the diagonal.
    """
    m = mat20.copy()
    m[max(i, j), min(i, j)] += eps
    return make_mat(m)


# ── parametrised fixture ───────────────────────────────────────────────────────

SOFT_ALIGNERS = [
    ("nw_linear",  nwgrad.nw_soft_grad,        nwgrad.nw_score,
     {"gap_extend": 1.0}),
    ("nw_affine",  nwgrad.nw_affine_soft_grad, nwgrad.nw_score_affine,
     {"gap_open": 11.0, "gap_extend": 1.0}),
    ("sw_linear",  nwgrad.sw_soft_grad,        nwgrad.sw_score,
     {"gap_extend": 1.0}),
    ("sw_affine",  nwgrad.sw_affine_soft_grad, nwgrad.sw_score_affine,
     {"gap_open": 11.0, "gap_extend": 1.0}),
]

PAIRS = [
    ("A",          "A"),
    ("ACDE",       "ACDE"),
    ("PLEASANTLY", "MEANLY"),
    ("MADEEKLF",   "ACDEFGHIKL"),
]


# ── 1. log_Z >= hard score ────────────────────────────────────────────────────

@pytest.mark.parametrize("name,soft_fn,score_fn,kw", SOFT_ALIGNERS,
                          ids=[x[0] for x in SOFT_ALIGNERS])
@pytest.mark.parametrize("a,b", PAIRS)
def test_log_z_geq_hard_score(a, b, name, soft_fn, score_fn, kw, blosum):
    log_z, _ = soft_fn(a, b, blosum, **kw)
    score    = score_fn(a, b, blosum, **kw)
    assert log_z >= score - 1e-9, (
        f"{name} ({a!r},{b!r}): log_Z={log_z:.6g} < score={score:.6g}"
    )


# ── 2. expected counts are non-negative and sum <= min(len(a), len(b)) ────────

@pytest.mark.parametrize("name,soft_fn,score_fn,kw", SOFT_ALIGNERS,
                          ids=[x[0] for x in SOFT_ALIGNERS])
@pytest.mark.parametrize("a,b", PAIRS)
def test_grad_nonneg_and_bounded(a, b, name, soft_fn, score_fn, kw, blosum):
    _, g = soft_fn(a, b, blosum, **kw)
    g = np.array(g)
    assert (g >= -1e-12).all(), f"{name}: negative gradient entry"
    assert g.sum() <= min(len(a), len(b)) + 1e-9, (
        f"{name} ({a!r},{b!r}): grad.sum={g.sum():.6g} > min_len={min(len(a),len(b))}"
    )


# ── 3. numerical gradient check (central finite differences) ─────────────────
# For each nonzero grad entry (i,j), verify:
#   (log_Z(s+eps) - log_Z(s-eps)) / (2*eps)  ≈  grad[i,j] + grad[j,i]
# (symmetry of BlosumMatrix means perturbing (i,j) also perturbs (j,i))

@pytest.mark.parametrize("name,soft_fn,score_fn,kw", SOFT_ALIGNERS,
                          ids=[x[0] for x in SOFT_ALIGNERS])
@pytest.mark.parametrize("a,b", [("ACDE", "ACDE"), ("PLEASANTLY", "MEANLY")])
def test_numerical_gradient(a, b, name, soft_fn, score_fn, kw):
    log_z0, g = soft_fn(a, b, make_mat(BLOSUM62), **kw)
    g = np.array(g)

    for i in range(20):
        for j in range(i, 20):   # upper triangle only (symmetry)
            expected = g[i, j] + (g[j, i] if i != j else 0.0)
            if abs(expected) < 1e-8:
                continue

            mat_p = perturb_entry(BLOSUM62, i, j, +EPS)
            mat_m = perturb_entry(BLOSUM62, i, j, -EPS)
            log_z_p, _ = soft_fn(a, b, mat_p, **kw)
            log_z_m, _ = soft_fn(a, b, mat_m, **kw)
            numerical = (log_z_p - log_z_m) / (2 * EPS)

            assert numerical == pytest.approx(expected, rel=1e-4, abs=1e-6), (
                f"{name} ({a!r},{b!r}) entry ({AA_ORDER[i]},{AA_ORDER[j]}): "
                f"numerical={numerical:.6g}, analytical={expected:.6g}"
            )


# ── 4. high-temperature limit: soft → hard on identity pairs ──────────────────
# Scale BLOSUM62 by a large factor T; log_Z/T → score, soft_grad → hard_grad.

@pytest.mark.parametrize("name,soft_fn,_score_fn,kw", SOFT_ALIGNERS,
                          ids=[x[0] for x in SOFT_ALIGNERS])
@pytest.mark.parametrize("a,b", [("ACDE", "ACDE"), ("MADEEKLF", "MADEEKLF")])
def test_low_temperature_limit(a, b, name, soft_fn, _score_fn, kw):
    T = 1000.0
    scaled_kw = {k: v * T for k, v in kw.items()}
    mat_scaled = make_mat(BLOSUM62 * T)

    log_z, g_soft = soft_fn(a, b, mat_scaled, **scaled_kw)
    g_soft = np.array(g_soft)

    # Hard gradient for comparison
    hard_fn_map = {
        "nw_linear":  nwgrad.nw_grad,
        "nw_affine":  nwgrad.nw_affine_grad,
        "sw_linear":  nwgrad.sw_grad,
        "sw_affine":  nwgrad.sw_affine_grad,
    }
    _, g_hard = hard_fn_map[name](a, b, make_mat(BLOSUM62), **kw)
    g_hard = np.array(g_hard)

    # Soft gradient should be very close to hard gradient at low temperature.
    assert g_soft == pytest.approx(g_hard, abs=1e-3), (
        f"{name} ({a!r},{b!r}): soft grad diverges from hard grad at T={T}"
    )


# ── 5. BatchAligner with grad_mode="soft" ─────────────────────────────────────

def test_batch_soft_grad_matches_single():
    """BatchAligner soft should match summed single-pair soft grads."""
    mat = nwgrad.BlosumMatrix(BLOSUM62)
    pairs = [("ACDE", "ACDF"), ("MADEEKLF", "MADEEKLF"), ("A", "A")]

    ba = nwgrad.BatchAligner(
        matrix=mat, gap_open=11.0, gap_extend=1.0,
        gap_model="affine", mode="global", grad_mode="soft", n_threads=2
    )
    res = ba.align([p[0] for p in pairs], [p[1] for p in pairs])

    # Sum individual soft grads
    expected_grad = np.zeros((20, 20))
    expected_scores = []
    for a, b in pairs:
        log_z, g = nwgrad.nw_affine_soft_grad(
            a, b, mat, gap_open=11.0, gap_extend=1.0
        )
        expected_grad += np.array(g)
        expected_scores.append(log_z)

    assert np.array(res.scores) == pytest.approx(expected_scores, rel=1e-9)
    assert np.array(res.grad) == pytest.approx(expected_grad, rel=1e-9, abs=1e-12)
