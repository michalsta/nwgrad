"""Empirical gradient tests — finite-difference verification of the hard subgradient."""

import numpy as np
import pytest
import nwgrad
from test_subst_matrix import BLOSUM62

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
EPS = 1e-5


def make_params(matrix_arr, gap_extend, gap_open=0.0):
    return nwgrad.AlignParams(matrix_arr,
                               gap_open_a=gap_open, gap_extend_a=gap_extend,
                               gap_open_b=gap_open, gap_extend_b=gap_extend)


def grad_matrix(g):
    return g.matrix.to_matrix()


# Each entry: (name, grad_fn, score_fn, gap_extend, gap_open)
# These are finite-difference gradient checks against a random *double* matrix, so they
# use the _double variants: float32's ~7 significant digits cannot match the numerical
# derivative to the tolerance here (that is a property of finite differencing, not a bug
# in the float32 kernel — its bit-exactness is proven in the C++ suite).
ALIGNERS = [
    ("nw_linear",  nwgrad.nw_grad_double,        nwgrad.nw_score_double,        1.0,  0.0),
    ("sw_linear",  nwgrad.sw_grad_double,        nwgrad.sw_score_double,        1.0,  0.0),
    ("nw_affine",  nwgrad.nw_affine_grad_double, nwgrad.nw_score_affine_double, 1.0, 11.0),
    ("sw_affine",  nwgrad.sw_affine_grad_double, nwgrad.sw_score_affine_double, 1.0, 11.0),
]

IDENTITY_PAIRS = [
    ("A",          "A"),
    ("ACDE",       "ACDE"),
    ("ACDEFG",     "ACDEFG"),
    ("MADEEKLF",   "MADEEKLF"),
    ("PLEASANTLY", "PLEASANTLY"),
]

GENERAL_PAIRS = [
    ("PLEASANTLY", "MEANLY"),
    ("ACDEFGHIKL", "CDEFGHIKLM"),
    ("ACDE",       "ACDF"),
    ("MADEEKLF",   "ACDEFGHIKL"),
]


@pytest.fixture(scope="module")
def blosum():
    return nwgrad.SubstMatrix(BLOSUM62)


def sym(delta):
    return (delta + delta.T) / 2.0


def perturb(delta, ge, go, eps=EPS):
    """Return AlignParams with BLOSUM62 + eps*sym(delta) and given gap penalties."""
    return make_params(BLOSUM62 + eps * sym(delta), ge, go)


def expected_delta(g_arr, delta, eps=EPS):
    return eps * float(np.dot(g_arr.ravel(), sym(delta).ravel()))


# ─── 1. gradient-direction tests ─────────────────────────────────────────────

@pytest.mark.parametrize("name,grad_fn,score_fn,ge,go", ALIGNERS,
                          ids=[a[0] for a in ALIGNERS])
@pytest.mark.parametrize("a,b", IDENTITY_PAIRS)
def test_grad_direction_identity_pairs(a, b, name, grad_fn, score_fn, ge, go):
    p0 = make_params(BLOSUM62, ge, go)
    score0, g = grad_fn(a, b, p0)
    g_arr = grad_matrix(g)
    assert g_arr.sum() == pytest.approx(len(a))

    p1 = perturb(g_arr, ge, go)
    score1 = score_fn(a, b, p1)

    exp = expected_delta(g_arr, g_arr)
    assert score1 - score0 == pytest.approx(exp, abs=1e-10), (
        f"{name} identity ({a!r}): Δ={score1 - score0:.6g}, want {exp:.6g}"
    )


@pytest.mark.parametrize("name,grad_fn,score_fn,ge,go", ALIGNERS,
                          ids=[a[0] for a in ALIGNERS])
@pytest.mark.parametrize("a,b", GENERAL_PAIRS)
def test_grad_direction_general_pairs(a, b, name, grad_fn, score_fn, ge, go):
    p0 = make_params(BLOSUM62, ge, go)
    score0, g = grad_fn(a, b, p0)
    g_arr = grad_matrix(g)
    if g_arr.sum() == 0:
        pytest.skip("zero gradient (all gaps)")

    p1 = perturb(g_arr, ge, go)
    score1 = score_fn(a, b, p1)

    exp = expected_delta(g_arr, g_arr)
    assert score1 - score0 == pytest.approx(exp, abs=1e-10), (
        f"{name} ({a!r},{b!r}): Δ={score1 - score0:.6g}, want {exp:.6g}"
    )


# ─── 2. random-direction tests ────────────────────────────────────────────────

@pytest.mark.parametrize("name,grad_fn,score_fn,ge,go", ALIGNERS,
                          ids=[a[0] for a in ALIGNERS])
@pytest.mark.parametrize("a,b", IDENTITY_PAIRS)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_random_direction_identity_pairs(a, b, seed, name, grad_fn, score_fn, ge, go):
    rng = np.random.default_rng(seed)
    delta = sym(rng.standard_normal((20, 20)))

    p0 = make_params(BLOSUM62, ge, go)
    score0, g = grad_fn(a, b, p0)
    g_arr = grad_matrix(g)

    p1 = perturb(delta, ge, go)
    score1 = score_fn(a, b, p1)

    exp = expected_delta(g_arr, delta)
    assert score1 - score0 == pytest.approx(exp, abs=1e-10), (
        f"{name} identity ({a!r}) seed={seed}: "
        f"Δ={score1-score0:.6g}, want {exp:.6g}"
    )


# ─── 3. per-entry finite difference ──────────────────────────────────────────

@pytest.mark.parametrize("name,grad_fn,score_fn,ge,go", ALIGNERS,
                          ids=[a[0] for a in ALIGNERS])
def test_per_entry_finite_difference(name, grad_fn, score_fn, ge, go):
    a, b = "MADEEKLF", "MADEEKLF"
    p0 = make_params(BLOSUM62, ge, go)
    score0, g = grad_fn(a, b, p0)
    g_arr = grad_matrix(g)

    for i in range(20):
        for j in range(20):
            if g_arr[i, j] == 0 and g_arr[j, i] == 0:
                continue
            delta = np.zeros((20, 20), dtype=np.float64)
            delta[i, j] = 1.0
            delta[j, i] = 1.0

            p1 = perturb(delta, ge, go)
            score1 = score_fn(a, b, p1)

            if i == j:
                exp = EPS * g_arr[i, j]
            else:
                exp = EPS * (g_arr[i, j] + g_arr[j, i])
            assert score1 - score0 == pytest.approx(exp, abs=1e-10), (
                f"{name} entry ({AA_ORDER[i]},{AA_ORDER[j]}): "
                f"Δ={score1-score0:.6g}, want {exp:.6g}"
            )
