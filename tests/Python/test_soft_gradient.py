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
from test_subst_matrix import BLOSUM62

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"

EPS = 1e-5


# ── helpers ─────────────────────────────────────────────────────────────────

def make_params(mat20, gap_extend, gap_open=0.0):
    return nwgrad.AlignParams(np.asarray(mat20, dtype=np.float64),
                               gap_open_a=gap_open, gap_extend_a=gap_extend,
                               gap_open_b=gap_open, gap_extend_b=gap_extend)


def perturb_entry(mat20, i, j, eps, gap_extend, gap_open=0.0):
    """Return AlignParams with mat20 perturbed at (i,j) and (j,i) by eps."""
    m = mat20.copy()
    m[i, j] += eps
    if i != j:
        m[j, i] += eps
    return make_params(m, gap_extend, gap_open)


def aa_idx(c):
    return AA_ORDER.index(c)


# ── parametrised fixture ──────────────────────────────────────────────────────

SOFT_ALIGNERS = [
    ("nw_linear",  nwgrad.nw_soft_grad,        nwgrad.nw_score,        1.0,  0.0),
    ("nw_affine",  nwgrad.nw_affine_soft_grad, nwgrad.nw_score_affine, 1.0, 11.0),
    ("sw_linear",  nwgrad.sw_soft_grad,        nwgrad.sw_score,        1.0,  0.0),
    ("sw_affine",  nwgrad.sw_affine_soft_grad, nwgrad.sw_score_affine, 1.0, 11.0),
]

PAIRS = [
    ("A",          "A"),
    ("ACDE",       "ACDE"),
    ("PLEASANTLY", "MEANLY"),
    ("MADEEKLF",   "ACDEFGHIKL"),
]


# ── 1. log_Z >= hard score ────────────────────────────────────────────────────

@pytest.mark.parametrize("name,soft_fn,score_fn,gap_extend,gap_open", SOFT_ALIGNERS,
                          ids=[x[0] for x in SOFT_ALIGNERS])
@pytest.mark.parametrize("a,b", PAIRS)
def test_log_z_geq_hard_score(a, b, name, soft_fn, score_fn, gap_extend, gap_open):
    p = make_params(BLOSUM62, gap_extend, gap_open)
    log_z, _ = soft_fn(a, b, p)
    score    = score_fn(a, b, p)
    assert log_z >= score - 1e-9, (
        f"{name} ({a!r},{b!r}): log_Z={log_z:.6g} < score={score:.6g}"
    )


# ── 2. expected counts are non-negative and sum <= min(len(a), len(b)) ────────

@pytest.mark.parametrize("name,soft_fn,score_fn,gap_extend,gap_open", SOFT_ALIGNERS,
                          ids=[x[0] for x in SOFT_ALIGNERS])
@pytest.mark.parametrize("a,b", PAIRS)
def test_grad_nonneg_and_bounded(a, b, name, soft_fn, score_fn, gap_extend, gap_open):
    p = make_params(BLOSUM62, gap_extend, gap_open)
    _, g = soft_fn(a, b, p)
    g = g.matrix.to_matrix()
    assert (g >= -1e-12).all(), f"{name}: negative gradient entry"
    assert g.sum() <= min(len(a), len(b)) + 1e-9, (
        f"{name} ({a!r},{b!r}): grad.sum={g.sum():.6g} > min_len={min(len(a),len(b))}"
    )


# ── 3. numerical gradient check (central finite differences) ─────────────────

@pytest.mark.parametrize("name,soft_fn,score_fn,gap_extend,gap_open", SOFT_ALIGNERS,
                          ids=[x[0] for x in SOFT_ALIGNERS])
@pytest.mark.parametrize("a,b", [("ACDE", "ACDE"), ("PLEASANTLY", "MEANLY")])
def test_numerical_gradient(a, b, name, soft_fn, score_fn, gap_extend, gap_open):
    p0 = make_params(BLOSUM62, gap_extend, gap_open)
    log_z0, g = soft_fn(a, b, p0)
    g = g.matrix.to_matrix()

    for i in range(20):
        for j in range(i, 20):
            expected = g[i, j] + (g[j, i] if i != j else 0.0)
            if abs(expected) < 1e-8:
                continue

            mat_p = perturb_entry(BLOSUM62, i, j, +EPS, gap_extend, gap_open)
            mat_m = perturb_entry(BLOSUM62, i, j, -EPS, gap_extend, gap_open)
            log_z_p, _ = soft_fn(a, b, mat_p)
            log_z_m, _ = soft_fn(a, b, mat_m)
            numerical = (log_z_p - log_z_m) / (2 * EPS)

            assert numerical == pytest.approx(expected, rel=1e-4, abs=1e-6), (
                f"{name} ({a!r},{b!r}) entry ({AA_ORDER[i]},{AA_ORDER[j]}): "
                f"numerical={numerical:.6g}, analytical={expected:.6g}"
            )


# ── 4. high-temperature limit: soft → hard on identity pairs ──────────────────

@pytest.mark.parametrize("name,soft_fn,_score_fn,gap_extend,gap_open", SOFT_ALIGNERS,
                          ids=[x[0] for x in SOFT_ALIGNERS])
@pytest.mark.parametrize("a,b", [("ACDE", "ACDE"), ("MADEEKLF", "MADEEKLF")])
def test_low_temperature_limit(a, b, name, soft_fn, _score_fn, gap_extend, gap_open):
    T = 1000.0
    p_scaled = make_params(BLOSUM62 * T, gap_extend * T, gap_open * T)

    log_z, g_soft = soft_fn(a, b, p_scaled)
    g_soft = g_soft.matrix.to_matrix()

    hard_fn_map = {
        "nw_linear":  nwgrad.nw_grad,
        "nw_affine":  nwgrad.nw_affine_grad,
        "sw_linear":  nwgrad.sw_grad,
        "sw_affine":  nwgrad.sw_affine_grad,
    }
    p0 = make_params(BLOSUM62, gap_extend, gap_open)
    _, g_hard = hard_fn_map[name](a, b, p0)
    g_hard = g_hard.matrix.to_matrix()

    assert g_soft == pytest.approx(g_hard, abs=1e-3), (
        f"{name} ({a!r},{b!r}): soft grad diverges from hard grad at T={T}"
    )


# ── 5. BatchAligner with grad_mode="soft" ─────────────────────────────────────

def test_batch_soft_grad_matches_single():
    """BatchAligner soft should match summed single-pair soft grads."""
    params = make_params(BLOSUM62, 1.0, 11.0)
    test_pairs = [("ACDE", "ACDF"), ("MADEEKLF", "MADEEKLF"), ("A", "A")]

    ba = nwgrad.BatchAligner(
        params=params, gap_model="affine", mode="global", grad_mode="soft", n_threads=2
    )
    res = ba.align([pr[0] for pr in test_pairs], [pr[1] for pr in test_pairs])

    expected_grad = np.zeros((20, 20))
    expected_scores = []
    for a, b in test_pairs:
        log_z, g = nwgrad.nw_affine_soft_grad(a, b, params)
        expected_grad += g.matrix.to_matrix()
        expected_scores.append(log_z)

    assert np.array(res.scores) == pytest.approx(expected_scores, rel=1e-9)
    np.testing.assert_allclose(res.grad.matrix.to_matrix(), expected_grad,
                                rtol=1e-9, atol=1e-12)
