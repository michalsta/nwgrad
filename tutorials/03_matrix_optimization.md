# Tutorial 3 — Gradient-based Substitution Matrix Optimisation

This tutorial shows how to use nwgrad's gradient computation to optimise a
substitution matrix against a dataset of known homologous sequence pairs.

## Background

A substitution matrix `S` assigns a score `S[a, b]` to aligning amino acid `a`
with amino acid `b`. The BLOSUM matrices were derived from observed substitution
frequencies in aligned blocks, but for a specific task (e.g., a particular protein
family or domain) a task-specific matrix may perform better.

nwgrad makes it possible to optimise `S` directly by gradient descent:

- **Hard gradient**: `∂score/∂S[a,b]` is the count of `(a,b)` pairs in the optimal
  alignment — integer-valued, valid as a subgradient.
- **Soft gradient**: `∂log_Z/∂S[a,b]` is the expected count of `(a,b)` pairs across
  all alignments weighted by their Boltzmann probability — continuous, true gradient.

The soft gradient is the right choice for gradient descent because it is everywhere
differentiable (the hard subgradient is not differentiable at score ties).

## Setup

```python
import numpy as np
import nwgrad

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"

# ... define BLOSUM62 array as in Tutorial 1 ...

# Start from BLOSUM62 as the initial matrix
matrix = BLOSUM62.copy()
```

## Constructing a toy training set

In a real use case you would load aligned homologous pairs from a database such as
SCOP, CATH, or Pfam. Here we use synthetically perturbed sequences for illustration.

```python
rng = np.random.default_rng(0)
alphabet = list(AA_ORDER)

def mutate(seq, rate=0.1):
    """Randomly substitute residues at the given rate."""
    chars = list(seq)
    for i in range(len(chars)):
        if rng.random() < rate:
            chars[i] = rng.choice(alphabet)
    return "".join(chars)

# 500 pairs: a "true" sequence and a mutated version
base_seqs = [
    "".join(rng.choice(alphabet, size=int(rng.integers(20, 60))).tolist())
    for _ in range(500)
]
seqs_a = base_seqs
seqs_b = [mutate(s, rate=0.15) for s in base_seqs]
```

## Objective and gradient

We maximise the sum of alignment log-partition functions over the training set.
This is equivalent to maximising the log-likelihood of the observed pairs under
a model that weights alignments by their Boltzmann probability.

```python
def objective_and_grad(matrix_flat, seqs_a, seqs_b,
                       gap_open=11.0, gap_extend=1.0, n_threads=4):
    """
    Returns (-total_log_z, -gradient) for minimisation with scipy.
    matrix_flat: (400,) float64 — the upper triangle is the free parameters;
    we enforce symmetry explicitly.
    """
    # Reconstruct symmetric 20×20 matrix
    mat20 = matrix_flat.reshape(20, 20)
    mat20 = (mat20 + mat20.T) / 2  # enforce symmetry

    bm = nwgrad.BlosumMatrix(mat20)

    aligner = nwgrad.BatchAligner(
        matrix=bm,
        gap_open=gap_open, gap_extend=gap_extend,
        gap_model="affine", mode="global",
        grad_mode="soft",
        n_threads=n_threads,
    )
    result = aligner.align(seqs_a, seqs_b)

    total_log_z = float(np.array(result.scores).sum())
    grad = np.array(result.grad)  # shape (20, 20), sum over batch

    # Symmetrise gradient (because we symmetrised the matrix)
    grad = (grad + grad.T) / 2

    return -total_log_z, -grad.ravel()
```

## Gradient descent loop

```python
from scipy.optimize import minimize

x0 = BLOSUM62.ravel().copy()

# A few steps of L-BFGS-B
result = minimize(
    fun=objective_and_grad,
    x0=x0,
    args=(seqs_a, seqs_b),
    method="L-BFGS-B",
    jac=True,
    options={"maxiter": 50, "ftol": 1e-10, "gtol": 1e-6, "disp": True},
)

optimised_matrix = result.x.reshape(20, 20)
optimised_matrix = (optimised_matrix + optimised_matrix.T) / 2
print("Optimised matrix diagonal:", np.diag(optimised_matrix))
```

## Manual SGD loop

For full control — useful when the training set is large and you want mini-batches:

```python
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
N_EPOCHS = 5
n_threads = 4

matrix = BLOSUM62.copy()
indices = np.arange(len(seqs_a))

for epoch in range(N_EPOCHS):
    rng.shuffle(indices)
    total_loss = 0.0

    for start in range(0, len(indices), BATCH_SIZE):
        batch = indices[start:start + BATCH_SIZE]
        ba = [seqs_a[i] for i in batch]
        bb = [seqs_b[i] for i in batch]

        bm = nwgrad.BlosumMatrix((matrix + matrix.T) / 2)
        aligner = nwgrad.BatchAligner(
            matrix=bm, gap_open=11.0, gap_extend=1.0,
            gap_model="affine", mode="global",
            grad_mode="soft", n_threads=n_threads,
        )
        result = aligner.align(ba, bb)

        loss = -float(np.array(result.scores).sum())
        grad = -np.array(result.grad)  # shape (20, 20)
        grad = (grad + grad.T) / 2     # symmetrise

        matrix -= LEARNING_RATE * grad
        total_loss += loss

    print(f"Epoch {epoch+1}: loss={total_loss:.2f}")

# Symmetrise and export
final_matrix = (matrix + matrix.T) / 2
print("Final diagonal:", np.diag(final_matrix))
```

## Checking the gradient numerically

Before running optimisation, it is good practice to verify that the soft gradient
is consistent with finite differences on a small example:

```python
a, b = "ACDE", "ACDF"
EPS = 1e-5

log_z, grad = nwgrad.nw_affine_soft_grad(a, b, nwgrad.BlosumMatrix(BLOSUM62),
                                          gap_open=11.0, gap_extend=1.0)
grad = np.array(grad)

# Check entry (0, 0) — A-A substitution
i, j = 0, 0
m_plus  = BLOSUM62.copy(); m_plus[i, j]  += EPS; m_plus[j, i]  += EPS
m_minus = BLOSUM62.copy(); m_minus[i, j] -= EPS; m_minus[j, i] -= EPS

lz_plus,  _ = nwgrad.nw_affine_soft_grad(a, b, nwgrad.BlosumMatrix(m_plus),
                                          gap_open=11.0, gap_extend=1.0)
lz_minus, _ = nwgrad.nw_affine_soft_grad(a, b, nwgrad.BlosumMatrix(m_minus),
                                          gap_open=11.0, gap_extend=1.0)

numerical  = (lz_plus - lz_minus) / (2 * EPS)
analytical = grad[i, j] + grad[j, i]  # both entries updated by the perturbation
print(f"Numerical:  {numerical:.8f}")
print(f"Analytical: {analytical:.8f}")
```

## Using the hard subgradient

The hard subgradient can be used with subgradient methods. The step size must
be annealed because the subgradient is not a descent direction in general.

```python
# Polyak-style step size schedule
def polyak_step(t, initial=1e-2, decay=0.99):
    return initial * (decay ** t)

matrix = BLOSUM62.copy()
aligner_cfg = dict(
    gap_open=11.0, gap_extend=1.0,
    gap_model="affine", mode="global",
    grad_mode="hard", n_threads=4,
)

for t in range(200):
    bm = nwgrad.BlosumMatrix((matrix + matrix.T) / 2)
    aligner = nwgrad.BatchAligner(matrix=bm, **aligner_cfg)
    result = aligner.align(seqs_a, seqs_b)

    # Maximise total score: gradient ascent
    grad = np.array(result.grad)
    grad = (grad + grad.T) / 2

    lr = polyak_step(t)
    matrix += lr * grad

    if t % 20 == 0:
        print(f"Step {t:3d}  lr={lr:.5f}  total_score={np.array(result.scores).sum():.1f}")
```

## Constraints and regularisation

In practice you may want to:

- **Regularise toward BLOSUM62** to prevent degenerate solutions:
  ```python
  reg_loss  = 0.01 * np.sum((matrix - BLOSUM62) ** 2)
  reg_grad  = 0.02 * (matrix - BLOSUM62)
  total_grad = soft_grad + reg_grad
  ```

- **Fix the diagonal** (self-substitution scores) and only optimise off-diagonal:
  ```python
  grad[np.diag_indices(20)] = 0
  ```

- **Project onto the cone** of symmetric positive-semidefinite matrices after each
  step to maintain a valid log-odds interpretation.

## Summary

| Gradient mode | When to use |
|---|---|
| `"soft"` | Gradient descent / L-BFGS — differentiable everywhere |
| `"hard"` | Subgradient methods, counting statistics, interpretability |
| `"none"` | Score-only evaluation (no gradient needed) |

The soft gradient satisfies `log_z ≥ hard_score` and `soft_grad.sum() ≤ min(len(a), len(b))`.
At low temperature (large matrix scale), `log_z / T → score` and `soft_grad → hard_grad`.
