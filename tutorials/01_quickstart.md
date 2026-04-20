# Tutorial 1 — Quick Start

This tutorial walks through the core nwgrad API: constructing a substitution matrix,
aligning a single pair of sequences, and reading back the score and gradient.

## Installation

```bash
pip install nwgrad
```

## The substitution matrix

All alignment functions require a `BlosumMatrix` constructed from a `(20, 20)` float64
numpy array. The row/column order is the canonical 20 amino-acid alphabet:

```
ACDEFGHIKLMNPQRSTVWY
```

```python
import numpy as np
import nwgrad

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"

# Full BLOSUM62 in canonical order
BLOSUM62 = np.array([
    # A   C   D   E   F   G   H   I   K   L   M   N   P   Q   R   S   T   V   W   Y
    [ 4,  0, -2, -1, -2,  0, -2, -1, -1, -1, -1, -2, -1, -1, -1,  1,  0,  0, -3, -2],
    [ 0,  9, -3, -4, -2, -3, -3, -1, -3, -1, -1, -3, -3, -3, -3, -1, -1, -1, -2, -2],
    [-2, -3,  6,  2, -3, -1, -1, -3, -1, -4, -3,  1, -1,  0, -2,  0, -1, -3, -4, -3],
    [-1, -4,  2,  5, -3, -2,  0, -3,  1, -3, -2,  0, -1,  2,  0,  0, -1, -2, -3, -2],
    [-2, -2, -3, -3,  6, -3, -1,  0, -3,  0,  0, -3, -4, -3, -3, -2, -2, -1,  1,  3],
    [ 0, -3, -1, -2, -3,  6, -2, -4, -2, -4, -3,  0, -2, -2, -2,  0, -2, -3, -2, -3],
    [-2, -3, -1,  0, -1, -2,  8, -3, -1, -3, -2,  1, -2,  0,  0, -1, -2, -3, -2,  2],
    [-1, -1, -3, -3,  0, -4, -3,  4, -3,  2,  1, -3, -3, -3, -3, -2, -1,  3, -3, -1],
    [-1, -3, -1,  1, -3, -2, -1, -3,  5, -2, -1,  0, -1,  1,  2,  0, -1, -2, -3, -2],
    [-1, -1, -4, -3,  0, -4, -3,  2, -2,  4,  2, -3, -3, -2, -2, -2, -1,  1, -2, -1],
    [-1, -1, -3, -2,  0, -3, -2,  1, -1,  2,  5, -2, -2,  0, -1, -1, -1,  1, -1, -1],
    [-2, -3,  1,  0, -3,  0,  1, -3,  0, -3, -2,  6, -2,  0,  0,  1,  0, -3, -4, -2],
    [-1, -3, -1, -1, -4, -2, -2, -3, -1, -3, -2, -2,  7, -1, -2, -1, -1, -2, -4, -3],
    [-1, -3,  0,  2, -3, -2,  0, -3,  1, -2,  0,  0, -1,  5,  1,  0, -1, -2, -2, -1],
    [-1, -3, -2,  0, -3, -2,  0, -3,  2, -2, -1,  0, -2,  1,  5, -1, -1, -3, -3, -2],
    [ 1, -1,  0,  0, -2,  0, -1, -2,  0, -2, -1,  1, -1,  0, -1,  4,  1, -2, -3, -2],
    [ 0, -1, -1, -1, -2, -2, -2, -1, -1, -1, -1,  0, -1, -1, -1,  1,  5,  0, -2, -2],
    [ 0, -1, -3, -2, -1, -3, -3,  3, -2,  1,  1, -3, -2, -2, -3, -2,  0,  4, -3, -1],
    [-3, -2, -4, -3,  1, -2, -2, -3, -3, -2, -1, -4, -4, -2, -3, -3, -2, -3, 11,  2],
    [-2, -2, -3, -2,  3, -3,  2, -1, -2, -1, -1, -2, -3, -1, -2, -2, -2, -1,  2,  7],
], dtype=np.float64)

blosum62 = nwgrad.BlosumMatrix(BLOSUM62)
```

The matrix is copied into an internal 256×256 ASCII-indexed table on construction.
Subsequent modifications to `BLOSUM62` have no effect on `blosum62`.

```python
# Individual score lookup
print(blosum62.score('A', 'A'))   # 4.0
print(blosum62.score('W', 'W'))   # 11.0
print(blosum62.score('D', 'E'))   # 2.0

# Round-trip back to numpy
recovered = blosum62.to_matrix()  # shape (20, 20), dtype float64
```

## Global alignment (Needleman-Wunsch)

### Affine gap penalty

The affine gap model charges `gap_open + gap_extend * k` for a gap of length `k`.
This is the standard model for protein alignment.

```python
a = "PLEASANTLY"
b = "MEANLY"

score = nwgrad.nw_score_affine(a, b, blosum62, gap_open=11.0, gap_extend=1.0)
print(f"NW affine score: {score}")  # 8.0
```

### Linear gap penalty

The linear gap model charges `gap_extend * k` for a gap of length `k`.

```python
score = nwgrad.nw_score(a, b, blosum62, gap_extend=1.0)
print(f"NW linear score: {score}")
```

## Local alignment (Smith-Waterman)

```python
score = nwgrad.sw_score_affine(a, b, blosum62, gap_open=11.0, gap_extend=1.0)
print(f"SW affine score: {score}")

score = nwgrad.sw_score(a, b, blosum62, gap_extend=1.0)
print(f"SW linear score: {score}")
```

Local alignment scores are always ≥ 0 (a zero-length local alignment is valid).

## Computing the hard subgradient

The gradient functions return `(score, grad)` where `grad` is a `(20, 20)` numpy array.

```python
score, grad = nwgrad.nw_affine_grad("PLEASANTLY", "MEANLY", blosum62,
                                    gap_open=11.0, gap_extend=1.0)

print(f"Score: {score}")
print(f"Grad shape: {grad.shape}")   # (20, 20)
print(f"Grad dtype: {grad.dtype}")   # float64
print(f"Grad sum: {grad.sum()}")     # number of matched positions
```

`grad[i, j]` counts how many times amino acid `AA_ORDER[i]` is aligned to
`AA_ORDER[j]` in the optimal alignment traceback. This is the subgradient of
`score` with respect to `matrix[i, j]`.

To inspect which substitutions occur:

```python
for i, a in enumerate(AA_ORDER):
    for j, b in enumerate(AA_ORDER):
        if grad[i, j] > 0:
            print(f"  {a}-{b}: {int(grad[i, j])}")
```

## Computing the soft (differentiable) gradient

The soft gradient replaces `max` with `log-sum-exp` in the DP recurrence.
It returns the log-partition function `log Z = log Σ exp(score(alignment))` and
the expected substitution counts under the Boltzmann distribution over alignments.

```python
log_z, soft_grad = nwgrad.nw_affine_soft_grad("PLEASANTLY", "MEANLY", blosum62,
                                               gap_open=11.0, gap_extend=1.0)

print(f"log Z: {log_z}")            # always >= hard score
print(f"soft_grad sum: {soft_grad.sum()}")  # fractional, <= min(len(a), len(b))
```

The soft gradient is the true gradient of `log_z` with respect to the substitution
matrix — verified by central finite differences in the test suite. It is suitable
for gradient-based optimisation of the substitution matrix.

## Summary of single-pair functions

| Function | Returns | Gap model |
|---|---|---|
| `nw_score(a, b, mat, gap_extend)` | `float` | Linear, global |
| `sw_score(a, b, mat, gap_extend)` | `float` | Linear, local |
| `nw_score_affine(a, b, mat, gap_open, gap_extend)` | `float` | Affine, global |
| `sw_score_affine(a, b, mat, gap_open, gap_extend)` | `float` | Affine, local |
| `nw_grad(a, b, mat, gap_extend)` | `(float, grad)` | Linear, global |
| `sw_grad(a, b, mat, gap_extend)` | `(float, grad)` | Linear, local |
| `nw_affine_grad(a, b, mat, gap_open, gap_extend)` | `(float, grad)` | Affine, global |
| `sw_affine_grad(a, b, mat, gap_open, gap_extend)` | `(float, grad)` | Affine, local |
| `nw_soft_grad(a, b, mat, gap_extend)` | `(log_z, grad)` | Linear, global |
| `sw_soft_grad(a, b, mat, gap_extend)` | `(log_z, grad)` | Linear, local |
| `nw_affine_soft_grad(a, b, mat, gap_open, gap_extend)` | `(log_z, grad)` | Affine, global |
| `sw_affine_soft_grad(a, b, mat, gap_open, gap_extend)` | `(log_z, grad)` | Affine, local |

Next: [Tutorial 2 — Batch alignment and multithreading](02_batch_alignment.md)
