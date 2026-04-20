"""Empirical gradient tests — finite-difference verification.

For a hard subgradient G of score s(M) where M is a SYMMETRIC substitution matrix:

    s(M + ε·δ) − s(M)  =  ε · ⟨G, δ⟩

exactly when ε is small enough that the optimal alignment path is unchanged.

SYMMETRY NOTE
-------------
BlosumMatrix stores double[256][256] and enforces symmetry during construction:
for each (i, j) pair it writes both mat[a][b] = v AND mat[b][a] = v.
Iterating i from 0..19 means the (i=row, j=col) pass is later than (i=col, j=row)
for row > col, so the LOWER TRIANGLE of the numpy input determines the final value.

Consequence: if the perturbation δ is asymmetric, BlosumMatrix will silently drop
the asymmetric part and only store the lower-triangle component.  All perturbations
in these tests must therefore be symmetric — i.e. δ[i,j] = δ[j,i].

The effective gradient with respect to a SYMMETRIC matrix has:
  G_eff[i,j] = G[i,j] + G[j,i]   (i ≠ j)
  G_eff[i,i] = G[i,i]             (diagonal unchanged)
i.e. G_eff = G + G.T - diag(G).

For any symmetric δ:  Δs = ε · ⟨G, δ⟩ = ε · ⟨G_eff/2 + diag_correction, δ⟩
                           = ε · np.dot(G.ravel(), δ.ravel())  (since ⟨G, δ⟩ = ⟨G.T, δ⟩ for sym δ)
"""

import numpy as np
import pytest
import nwgrad
from test_blosum import BLOSUM62

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
EPS = 1e-5   # small enough that identity-pair path won't shift

ALIGNERS = [
    ("nw_linear",  nwgrad.nw_grad,        nwgrad.nw_score,
     {"gap_extend": 1.0}),
    ("sw_linear",  nwgrad.sw_grad,        nwgrad.sw_score,
     {"gap_extend": 1.0}),
    ("nw_affine",  nwgrad.nw_affine_grad, nwgrad.nw_score_affine,
     {"gap_open": 11.0, "gap_extend": 1.0}),
    ("sw_affine",  nwgrad.sw_affine_grad, nwgrad.sw_score_affine,
     {"gap_open": 11.0, "gap_extend": 1.0}),
]

# Identity pairs: traceback is the diagonal for any matrix with positive
# self-scores (BLOSUM62 satisfies this), so a small perturbation cannot shift
# the optimal path — making these safe for exact finite-difference tests.
IDENTITY_PAIRS = [
    ("A",          "A"),
    ("ACDE",       "ACDE"),
    ("ACDEFG",     "ACDEFG"),
    ("MADEEKLF",   "MADEEKLF"),
    ("PLEASANTLY", "PLEASANTLY"),
]

# Non-identity pairs where a unique optimal alignment is typical.
# We use these for gradient-direction tests where the perturbed path should
# remain stable with EPS=1e-5.
GENERAL_PAIRS = [
    ("PLEASANTLY", "MEANLY"),
    ("ACDEFGHIKL", "CDEFGHIKLM"),
    ("ACDE",       "ACDF"),
    ("MADEEKLF",   "ACDEFGHIKL"),
]

# ─── helpers ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def blosum():
    return nwgrad.BlosumMatrix(BLOSUM62)


def sym(delta):
    """Return the symmetric part of a 20×20 matrix."""
    return (delta + delta.T) / 2.0


def make_blosum(mat20):
    """Construct a BlosumMatrix from a 20×20 float64 array."""
    return nwgrad.BlosumMatrix(np.asarray(mat20, dtype=np.float64))


def perturb(delta, eps=EPS):
    """Return a new BlosumMatrix with BLOSUM62 + eps * sym(delta).

    Symmetrises delta before adding so the BlosumMatrix round-trip is exact.
    """
    return make_blosum(BLOSUM62 + eps * sym(delta))


def expected_delta(g, delta, eps=EPS):
    """Predicted score change: eps · ⟨G, sym(δ)⟩ = eps · ⟨G, δ⟩ (since δ sym.)"""
    return eps * float(np.dot(g.ravel(), sym(delta).ravel()))


# ─── 1. gradient-direction tests ─────────────────────────────────────────────
# δ = G (symmetrised) → Δscore = ε·⟨G, sym(G)⟩

@pytest.mark.parametrize("name,grad_fn,score_fn,kw", ALIGNERS,
                          ids=[a[0] for a in ALIGNERS])
@pytest.mark.parametrize("a,b", IDENTITY_PAIRS)
def test_grad_direction_identity_pairs(a, b, name, grad_fn, score_fn, kw, blosum):
    """Identity pairs: Δscore = ε·⟨G, sym(G)⟩ exactly."""
    score0, g = grad_fn(a, b, blosum, **kw)
    g = np.array(g, dtype=np.float64)
    assert g.sum() == pytest.approx(len(a))  # sanity: all positions matched

    mat_p = perturb(g)
    score1 = score_fn(a, b, mat_p, **kw)

    exp = expected_delta(g, g)
    assert score1 - score0 == pytest.approx(exp, abs=1e-10), (
        f"{name} identity ({a!r}): Δ={score1 - score0:.6g}, want {exp:.6g}"
    )


@pytest.mark.parametrize("name,grad_fn,score_fn,kw", ALIGNERS,
                          ids=[a[0] for a in ALIGNERS])
@pytest.mark.parametrize("a,b", GENERAL_PAIRS)
def test_grad_direction_general_pairs(a, b, name, grad_fn, score_fn, kw, blosum):
    """General pairs: Δscore = ε·⟨G, sym(G)⟩ when path is stable (EPS=1e-5)."""
    score0, g = grad_fn(a, b, blosum, **kw)
    g = np.array(g, dtype=np.float64)
    if g.sum() == 0:
        pytest.skip("zero gradient (all gaps)")

    mat_p = perturb(g)
    score1 = score_fn(a, b, mat_p, **kw)

    exp = expected_delta(g, g)
    assert score1 - score0 == pytest.approx(exp, abs=1e-10), (
        f"{name} ({a!r},{b!r}): Δ={score1 - score0:.6g}, want {exp:.6g}"
    )


# ─── 2. random-direction tests ────────────────────────────────────────────────
# δ = random symmetric 20×20 → Δscore = ε·⟨G, δ⟩
# Identity pairs guarantee path stability regardless of direction.

@pytest.mark.parametrize("name,grad_fn,score_fn,kw", ALIGNERS,
                          ids=[a[0] for a in ALIGNERS])
@pytest.mark.parametrize("a,b", IDENTITY_PAIRS)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_random_direction_identity_pairs(a, b, seed, name, grad_fn, score_fn, kw, blosum):
    """Identity pairs: Δscore = ε·⟨G, δ⟩ for random symmetric δ."""
    rng = np.random.default_rng(seed)
    delta = sym(rng.standard_normal((20, 20)))  # symmetric

    score0, g = grad_fn(a, b, blosum, **kw)
    g = np.array(g, dtype=np.float64)

    mat_p = perturb(delta)
    score1 = score_fn(a, b, mat_p, **kw)

    exp = expected_delta(g, delta)
    assert score1 - score0 == pytest.approx(exp, abs=1e-10), (
        f"{name} identity ({a!r}) seed={seed}: "
        f"Δ={score1-score0:.6g}, want {exp:.6g}"
    )


# ─── 3. per-entry finite difference ──────────────────────────────────────────
# Perturb a single symmetric entry (i,j) + (j,i) and verify
# Δscore = ε·(G[i,j] + G[j,i])  (or ε·G[i,i] on the diagonal).

@pytest.mark.parametrize("name,grad_fn,score_fn,kw", ALIGNERS,
                          ids=[a[0] for a in ALIGNERS])
def test_per_entry_finite_difference(name, grad_fn, score_fn, kw, blosum):
    """For each nonzero G[i,j]: s(M + ε·e_{ij}^sym) − s(M) = ε·(G[i,j] + G[j,i])."""
    a, b = "MADEEKLF", "MADEEKLF"   # identity: path is always stable
    score0, g = grad_fn(a, b, blosum, **kw)
    g = np.array(g, dtype=np.float64)

    for i in range(20):
        for j in range(20):
            if g[i, j] == 0 and g[j, i] == 0:
                continue
            # symmetric unit perturbation at (i,j) + (j,i)
            delta = np.zeros((20, 20), dtype=np.float64)
            delta[i, j] = 1.0
            delta[j, i] = 1.0   # already handled by sym(), but explicit is clearer

            mat_p = perturb(delta)
            score1 = score_fn(a, b, mat_p, **kw)

            if i == j:
                exp = EPS * g[i, j]
            else:
                exp = EPS * (g[i, j] + g[j, i])
            assert score1 - score0 == pytest.approx(exp, abs=1e-10), (
                f"{name} entry ({AA_ORDER[i]},{AA_ORDER[j]}): "
                f"Δ={score1-score0:.6g}, want {exp:.6g}"
            )
