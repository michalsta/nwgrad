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
from nwgrad.matrices import BLOSUM62     # a SubstMatrix carrying its own alphabet

ALPHABET   = BLOSUM62.alphabet           # row/column order of the matrix
N          = BLOSUM62.size
blosum_arr = BLOSUM62.to_matrix()        # (N, N) array, in ALPHABET order

# Helper: build AlignParams from a raw array, always using the matrix's alphabet
# so amino acids map to the right cells.
def make_params(mat, gap_open=11.0, gap_extend=1.0):
    return nwgrad.AlignParams(
        nwgrad.SubstMatrix(mat, ALPHABET),
        gap_open_a=gap_open, gap_extend_a=gap_extend,
        gap_open_b=gap_open, gap_extend_b=gap_extend,
    )

# Start from BLOSUM62 as the initial (learnable) matrix
mat_array = blosum_arr.copy()
```

## Constructing a toy training set

In a real use case you would load aligned homologous pairs from a database such as
SCOP, CATH, or Pfam. Here we use synthetically perturbed sequences for illustration.

```python
rng = np.random.default_rng(0)
alphabet = list("ACDEFGHIKLMNPQRSTVWY")   # draw only canonical residues for the toy data

def mutate(seq, rate=0.1):
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

## Building the batch

`SeqPair` objects are constructed once and reused across all optimisation
iterations. Only the matrix (inside `AlignParams`) changes between iterations —
the sequences and alignment mode are fixed.

```python
params = make_params(mat_array)

batch = nwgrad.SeqPairBatch(n_threads=4)
for a, b in zip(seqs_a, seqs_b):
    batch.add(nwgrad.SeqPair(
        a, b, params,
        gap_model="affine", mode="global",
        grad_mode="soft",
    ))
```

## Gradient descent loop

```python
LEARNING_RATE = 1e-4
BANDWIDTH = 30   # banded DP half-width for iterations 2+

for step in range(50):
    # First step: full DP (no path yet). Subsequent steps: banded DP.
    bw = 0 if step == 0 else BANDWIDTH
    total_log_z = batch.score_and_grad(bandwidth=bw)

    # Summed gradient over all pairs → AlignParams
    grad = batch.compute_grad()

    mat_array += LEARNING_RATE * grad.matrix.to_matrix()   # gradient ascent on log Z
    params = make_params(mat_array)
    batch.set_params(params)

    if step % 10 == 0:
        print(f"step {step:3d}  total log Z = {total_log_z:.2f}")
```

`batch.set_params()` invalidates cached scores and gradients on all pairs but
preserves the alignment paths, so the next `score_and_grad(bandwidth=bw)` uses
banded DP instead of a full re-alignment. The batch keeps `params` alive
automatically.

## Using scipy L-BFGS-B

For second-order methods that call an objective function repeatedly:

```python
from scipy.optimize import minimize

# N = BLOSUM62.size, defined in Setup

def objective_and_grad(matrix_flat):
    p = make_params(matrix_flat.reshape(N, N))
    batch.set_params(p)

    total_log_z = batch.score_and_grad()
    grad = batch.compute_grad()

    return -float(total_log_z), -grad.matrix.to_matrix().ravel()

result = minimize(
    fun=objective_and_grad,
    x0=mat_array.ravel(),
    method="L-BFGS-B",
    jac=True,
    options={"maxiter": 50, "ftol": 1e-10, "gtol": 1e-6, "disp": True},
)

optimised = result.x.reshape(N, N)
print("Optimised matrix diagonal:", np.diag(optimised))
```

Note: L-BFGS-B calls the objective many times per step, so using `bandwidth=0`
(full DP) on every call is safer than banding when the matrix changes
substantially between calls.

## Mini-batch SGD

For large datasets, process pairs in mini-batches. Build a separate
`SeqPairBatch` per mini-batch, or use index slicing with a single large batch:

```python
BATCH_SIZE = 64
LR = 1e-4
all_pairs = list(range(len(seqs_a)))

for epoch in range(5):
    rng.shuffle(all_pairs)
    total_loss = 0.0

    for start in range(0, len(all_pairs), BATCH_SIZE):
        idx = all_pairs[start:start + BATCH_SIZE]
        p = make_params(mat_array)

        mini_batch = nwgrad.SeqPairBatch(n_threads=4)
        for i in idx:
            mini_batch.add(nwgrad.SeqPair(
                seqs_a[i], seqs_b[i], p,
                gap_model="affine", mode="global", grad_mode="soft",
            ))

        loss = -mini_batch.score_and_grad()
        grad = mini_batch.compute_grad()

        mat_array -= LR * grad.matrix.to_matrix()
        total_loss += loss

    print(f"Epoch {epoch+1}: loss={total_loss:.2f}")

print("Final diagonal:", np.diag(mat_array))
```

## Checking the gradient numerically

Before running optimisation, verify that the soft gradient is consistent with
finite differences on a small example:

```python
a, b = "ACDE", "ACDF"
EPS = 1e-5

sp = nwgrad.SeqPair(a, b, make_params(blosum_arr),
                    gap_model="affine", mode="global", grad_mode="soft")
sp.alloc_dp()
sp.align_full()
sp.compute_grad()
log_z = sp.score
grad  = sp.grad.matrix.to_matrix().copy()

# Check entry (0, 0) — A-A substitution (index 0 in ALPHABET)
i, j = 0, 0
m_plus  = blosum_arr.copy(); m_plus[i, j]  += EPS; m_plus[j, i]  += EPS
m_minus = blosum_arr.copy(); m_minus[i, j] -= EPS; m_minus[j, i] -= EPS

p_plus  = make_params(m_plus)
sp_p = nwgrad.SeqPair(a, b, p_plus,
                      gap_model="affine", mode="global", grad_mode="soft")
sp_p.alloc_dp()
sp_p.align_full()

p_minus = make_params(m_minus)
sp_m = nwgrad.SeqPair(a, b, p_minus,
                      gap_model="affine", mode="global", grad_mode="soft")
sp_m.alloc_dp()
sp_m.align_full()

numerical  = (sp_p.score - sp_m.score) / (2 * EPS)
analytical = grad[i, j] + grad[j, i]   # both entries changed by the perturbation
print(f"Numerical:  {numerical:.8f}")
print(f"Analytical: {analytical:.8f}")
```

## Constraints and regularisation

In practice you may want to:

- **Regularise toward BLOSUM62** to prevent degenerate solutions:
  ```python
  reg_loss = 0.01 * np.sum((mat_array - blosum_arr) ** 2)
  reg_grad = 0.02 * (mat_array - blosum_arr)
  grad += reg_grad
  ```

- **Fix the diagonal** (self-substitution scores) and only optimise off-diagonal:
  ```python
  grad_mat = grad.matrix.to_matrix()
  grad_mat[np.diag_indices(grad_mat.shape[0])] = 0
  ```

- **Project onto the cone** of symmetric positive-semidefinite matrices after each
  step to maintain a valid log-odds interpretation.

## Using the hard subgradient

The hard subgradient can be used with subgradient methods. The step size must
be annealed because the subgradient is not a descent direction in general.

```python
params_hard = make_params(mat_array)
batch_hard = nwgrad.SeqPairBatch(n_threads=4)
for a, b in zip(seqs_a, seqs_b):
    batch_hard.add(nwgrad.SeqPair(
        a, b, params_hard,
        gap_model="affine", mode="global", grad_mode="hard",
    ))

for t in range(200):
    lr = 1e-2 * (0.99 ** t)
    total = batch_hard.score_and_grad()
    grad  = batch_hard.compute_grad()

    mat_array += lr * grad.matrix.to_matrix()
    params_hard = make_params(mat_array)
    batch_hard.set_params(params_hard)

    if t % 20 == 0:
        print(f"step {t:3d}  lr={lr:.5f}  total_score={total:.1f}")
```

## Summary

| Gradient mode | When to use |
|---|---|
| `"soft"` | Gradient descent / L-BFGS — differentiable everywhere |
| `"hard"` | Subgradient methods, counting statistics, interpretability |
| `"none"` | Score-only evaluation (no gradient needed) |

The soft gradient satisfies `log_z ≥ hard_score` and `soft_grad.sum() ≤ min(len(a), len(b))`.
At low temperature (large matrix scale), `log_z / T → score` and `soft_grad → hard_grad`.
