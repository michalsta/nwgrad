# Tutorial 1 — Quick Start

This tutorial walks through the core nwgrad API: constructing a substitution matrix,
aligning a single pair of sequences, and reading back the score and gradient.

## Installation

```bash
pip install nwgrad
```

## The substitution matrix

All alignment functions require a `SubstMatrix` constructed from an `(N, N)` float64
numpy array and an alphabet string of length `N`. The default alphabet is the canonical
20 amino-acid order:

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

blosum62 = nwgrad.SubstMatrix(BLOSUM62)
# Equivalent explicit form:
# blosum62 = nwgrad.SubstMatrix(BLOSUM62, alphabet="ACDEFGHIKLMNPQRSTVWY")
```

The matrix is copied into an internal 256×256 ASCII-indexed table on construction.
Subsequent modifications to `BLOSUM62` have no effect on `blosum62`.

```python
# Individual score lookup
print(blosum62.score('A', 'A'))   # 4.0
print(blosum62.score('W', 'W'))   # 11.0
print(blosum62.score('D', 'E'))   # 2.0

# Inspect alphabet
print(blosum62.size)       # 20
print(blosum62.alphabet)   # "ACDEFGHIKLMNPQRSTVWY"

# Round-trip back to numpy — shape is (N, N) = (20, 20) for the default alphabet
recovered = blosum62.to_matrix()
```

### Custom alphabets

Pass any square numpy array and a matching alphabet string to work with non-standard
residues, DNA, RNA, or codon tables:

```python
# 4×4 DNA matrix: match=2, mismatch=-1
DNA_ORDER = "ACGT"
dna_matrix = np.eye(4, dtype=np.float64) * 3 - 1   # diagonal 2, off-diagonal -1
dna_sm = nwgrad.SubstMatrix(dna_matrix, alphabet=DNA_ORDER)

print(dna_sm.size)       # 4
print(dna_sm.alphabet)   # "ACGT"
print(dna_sm.score('A', 'A'))   # 2.0
print(dna_sm.score('A', 'C'))   # -1.0

# Align DNA sequences with a linear-gap model
params = nwgrad.AlignParams(dna_matrix, alphabet=DNA_ORDER,
                             gap_extend_a=2.0, gap_extend_b=2.0)
score, grad = nwgrad.nw_grad("ACGTACGT", "ACGTACGT", params)
print(score)                         # 16.0 (8 perfect matches × 2)
print(grad.matrix.to_matrix())       # 4×4 identity (one count per match position)
print(grad.matrix.alphabet)          # "ACGT"
```

The gradient matrix shape always matches the alphabet: `to_matrix()` returns `(N, N)`.

## Aligning a sequence pair with `SeqPair`

`SeqPair` is the primary interface for single-pair alignment. It holds the
sequences, alignment parameters, and cached alignment state, and can be reused
efficiently across multiple matrix updates.

`SeqPair` takes an `AlignParams` object that bundles the substitution matrix
with gap penalties. Construct one before creating the pair:

```python
params = nwgrad.AlignParams(
    BLOSUM62,
    gap_open_a=11.0, gap_extend_a=1.0,   # gap penalties for sequence A
    gap_open_b=11.0, gap_extend_b=1.0,   # gap penalties for sequence B
)

sp = nwgrad.SeqPair(
    "PLEASANTLY", "MEANLY", params,
    gap_model="affine",   # "linear" | "affine"
    mode="global",        # "global" (NW) | "local" (SW)
    grad_mode="hard",     # "hard" | "soft" | "none"
)
```

`SeqPair` stores a reference to the `AlignParams` object — keep it alive for
the lifetime of the pair.

### Computing the score

```python
sp.alloc_dp()     # allocate own DP buffer before the first alignment
sp.align_full()
print(sp.score)   # -3.0
```

`alloc_dp()` must be called once before `align_full()`. It is not needed when
using `SeqPairBatch.score_and_grad()`, which allocates per-thread buffers
instead.

`align_full()` runs the full O(m×n) DP and caches the alignment path (`guide_j`).

### Computing the hard subgradient

```python
sp.compute_grad()
grad = sp.grad     # AlignParams object
```

`grad.matrix.to_matrix()[i, j]` counts how many times `alphabet[i]` is aligned to
`alphabet[j]` in the optimal traceback. This is the subgradient of the score
with respect to `matrix[i, j]`.

```python
for i, a in enumerate(AA_ORDER):
    for j, b in enumerate(AA_ORDER):
        if grad[i, j] > 0:
            print(f"  {a}-{b}: {int(grad[i, j])}")
```

### Computing the soft (differentiable) gradient

Construct with `grad_mode="soft"` to use the forward-backward algorithm instead
of Viterbi traceback. The score becomes `log Z = log Σ exp(score(alignment))`
summed over all alignments, and the gradient is the expected substitution counts
under the Boltzmann distribution — the true gradient of `log Z` with respect to
the matrix.

```python
sp_soft = nwgrad.SeqPair(
    "PLEASANTLY", "MEANLY", params,
    gap_model="affine", mode="global",
    grad_mode="soft",
)
sp_soft.alloc_dp()
sp_soft.align_full()
print(sp_soft.score)    # log Z — always >= hard score

sp_soft.compute_grad()
soft_grad = sp_soft.grad                         # AlignParams object
soft_mat  = sp_soft.grad.matrix.to_matrix()      # fractional, shape (N, N)
print(f"soft_grad sum: {soft_mat.sum()}")  # <= min(len(a), len(b))
```

The soft gradient is the right choice for gradient-based optimisation — it is
everywhere differentiable (the hard subgradient is not differentiable at score ties).

### All four alignment modes

```python
for gap_model in ("linear", "affine"):
    for mode in ("global", "local"):
        kw = dict(gap_extend_a=1.0, gap_extend_b=1.0)
        if gap_model == "affine":
            kw["gap_open_a"] = kw["gap_open_b"] = 11.0
        p = nwgrad.AlignParams(BLOSUM62, **kw)
        sp = nwgrad.SeqPair("PLEASANTLY", "MEANLY", p,
                            gap_model=gap_model, mode=mode)
        sp.alloc_dp()
        sp.align_full()
        print(f"{gap_model:6s} {mode:6s}  score={sp.score:.1f}")
```

Local alignment scores are always ≥ 0 (a zero-length local alignment is valid).

## Updating the matrix

After a gradient step the matrix changes but the alignment path is still a
reasonable guide. `set_params()` swaps the parameters and preserves the cached
path, so `realign_banded()` can re-score cheaply around it instead of running
the full O(m×n) DP again.

```python
new_mat_array = BLOSUM62 + 0.1 * grad.matrix.to_matrix()
new_params = nwgrad.AlignParams(new_mat_array,
                                gap_open_a=11.0, gap_extend_a=1.0,
                                gap_open_b=11.0, gap_extend_b=1.0)

sp.set_params(new_params)      # clears score/grad; path_valid stays True
sp.realign_banded(bandwidth=20)
sp.compute_grad()
print(sp.score, sp.grad.matrix.to_matrix().sum())
```

`new_params` must remain alive for as long as `sp` uses it.

The `bandwidth` parameter is the half-width of the band in cells. If the true
optimal path lies outside the band, the result is silently sub-optimal.

## State flags

```python
sp.path_valid   # guide_j is usable (realign_banded is callable)
sp.score_valid  # score matches current matrix and path
sp.grad_valid   # grad is populated
sp.dp_valid     # DP tables are still in memory (compute_grad is callable)
```

`compute_grad()` requires both `score_valid` and `dp_valid`. After
`score_and_grad()` via `SeqPairBatch` (see Tutorial 2), `dp_valid` is `False`
but `grad_valid` is already `True`.

## Summary

| `grad_mode` | `score` | `grad` |
|---|---|---|
| `"hard"` | Viterbi alignment score | Substitution-pair counts (integer-valued) |
| `"soft"` | Log-partition function `log Z` | Expected counts (differentiable) |
| `"none"` | Viterbi alignment score | `compute_grad()` throws |

The soft gradient satisfies `log_z ≥ hard_score`. At high matrix scale (low temperature), `log_z → score` and `soft_grad → hard_grad`.

Next: [Tutorial 2 — Batch alignment with SeqPairBatch](02_batch_alignment.md)
