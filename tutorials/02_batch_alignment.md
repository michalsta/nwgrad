# Tutorial 2 — Batch Alignment with SeqPairBatch

This tutorial covers `SeqPairBatch`: aligning many sequence pairs in parallel
and collecting scores and gradients efficiently.

## Why `SeqPairBatch`?

`SeqPairBatch` is designed for the workload that appears in substitution matrix
optimisation: the same set of sequence pairs is aligned repeatedly as the matrix
changes. Compared to looping over `SeqPair` objects in Python:

1. **Thread-level parallelism** — pairs are distributed across CPU cores via a
   lock-free work queue, and short pairs are aligned several per vector instruction
   (`fill="interpair"`, the default).
2. **Memory efficiency** — a pair costs a few hundred bytes in the batch's own arrays
   (its encoded sequences, score, alignment path and gradient), and the DP runs on one
   buffer per thread, reused across pairs: peak DP memory is `n_threads × max(m×n)`,
   not `N × max(m×n)`.
3. **Path reuse** — each pair's alignment path (`guide_j`) is cached. After
   `set_params()`, `banded_grad(bw)` runs a cheap banded DP around the old path
   rather than a full DP.

## Basic usage

A batch holds pairs of **one problem type**: one gap model, one alignment mode and
one gradient mode, given to the constructor.

```python
import numpy as np
import nwgrad
from nwgrad.matrices import BLOSUM62   # a SubstMatrix, NCBI 23-symbol alphabet

seqs_a = ["PLEASANTLY", "ACDEFGHIKL", "MADEEKLF", "ACDE"]
seqs_b = ["MEANLY",     "CDEFGHIKLM", "MADEEKLF", "ACDF"]

# AlignParams bundles the matrix with gap penalties.  Pass the SubstMatrix
# directly — its alphabet travels with it.
params = nwgrad.AlignParams(BLOSUM62,
                            gap_open_a=11.0, gap_extend_a=1.0,
                            gap_open_b=11.0, gap_extend_b=1.0)

batch = nwgrad.SeqPairBatch(n_threads=4,           # 0 = physical core count
                            gap_model="affine", mode="global", grad_mode="soft")
batch.add_many(seqs_a, seqs_b, params)              # encoded once, in parallel
print(len(batch))                                   # 4
```

The batch keeps `params` alive. `add_many()` may be called again to append more pairs,
with the same or other parameters (over the same alphabet).

## `score_and_grad()`

The primary batch operation. Aligns all pairs in parallel and caches each pair's score,
gradient and alignment path.

```python
total_log_z = batch.score_and_grad()
print(f"Sum of log Z: {total_log_z:.3f}")

# Per-pair results: as arrays, or one pair at a time
print(batch.scores())                     # float64 array, pair order
for i in range(len(batch)):
    sp = batch[i]                         # a SeqPair view of pair i
    print(f"  score={sp.score:.2f}  grad_valid={sp.grad_valid}")

# Sum gradients across all pairs → AlignParams
grad = batch.compute_grad()
print(f"Gradient matrix shape: {grad.matrix.to_matrix().shape}")   # (N, N)
```

`compute_grad()` sums the cached per-pair gradients; it runs no DP.

## Updating the matrix and banded re-alignment

After a gradient step, call `set_params()` on the batch to push the new parameters
to all pairs at once. The alignment paths are kept, so `banded_grad(bw)` can re-align
around them instead of running the full DP again:

```python
alphabet  = BLOSUM62.alphabet
mat_array = BLOSUM62.to_matrix()        # learnable array, in `alphabet` order

batch.score_and_grad()                  # full DP once: every pair gets a path
for step in range(20):
    grad = batch.compute_grad()
    mat_array += 0.01 * grad.matrix.to_matrix()   # grad is in the same alphabet order
    params = nwgrad.AlignParams(nwgrad.SubstMatrix(mat_array, alphabet),
                                gap_open_a=11.0, gap_extend_a=1.0,
                                gap_open_b=11.0, gap_extend_b=1.0)
    batch.set_params(params)
    total = batch.banded_grad(30)       # banded DP around the previous paths
    print(f"step {step:2d}  total log Z = {total:.2f}")
```

The bandwidth is the half-width of the band in DP cells. If the true optimal path
under the new matrix lies outside the band, the score is silently sub-optimal. Wider
bands are safer but slower; re-run the full `score_and_grad()` now and then to
re-centre the paths.

## Gradient modes

| `grad_mode` | `score` per pair | `compute_grad()` result |
|---|---|---|
| `"soft"` | log-partition `log Z` | Expected substitution counts (differentiable) |
| `"hard"` | Viterbi alignment score | Substitution-pair counts (integer-valued) |
| `"none"` | Viterbi alignment score | Raises |

Use `"none"` when you only need scores — it skips the gradient. It does not produce a
zero gradient: `compute_grad()` on a `"none"` batch raises, rather than handing back a
zero that would quietly cancel out of a sum.

## All four alignment modes

`SeqPairBatch` supports all combinations of gap model and mode — one per batch. To
align the same pairs under several, use one batch for each:

```python
params_lin = nwgrad.AlignParams(BLOSUM62, gap_extend_a=1.0, gap_extend_b=1.0)
batch_lin_global = nwgrad.SeqPairBatch(n_threads=4, gap_model="linear",
                                       mode="global", grad_mode="hard")
batch_lin_global.add_many(seqs_a, seqs_b, params_lin)

total = batch_lin_global.score_and_grad()
grad  = batch_lin_global.compute_grad()
```

## Thread count and reproducibility

Results are bit-for-bit independent of the thread count: every pair is computed the
same way whichever thread takes it, and the sums (`score_and_grad()`'s total,
`compute_grad()`, `weighted_grad()`) run in a fixed order.

```python
ref = None
for n in [1, 2, 4, 8]:
    b = nwgrad.SeqPairBatch(n_threads=n, gap_model="affine", mode="global",
                            grad_mode="soft")
    b.add_many(seqs_a, seqs_b, params)
    total = b.score_and_grad()
    g = b.compute_grad().matrix.to_matrix()
    if ref is None:
        ref = (total, g)
    assert total == ref[0] and np.array_equal(g, ref[1])
```

`n_threads=0` (the default) uses the number of physical cores (SMT siblings
contend in this DP, so the logical count is slower). The thread count is clamped to
the amount of work.

## All-vs-all pairs

Batch input is a flat list of pairs. To align every sequence against every other:

```python
sequences = ["ACDEFG", "MADEEKLF", "PLEASANTLY", "ACDE", "CDEFGHIKLM"]
ii, jj = np.triu_indices(len(sequences), k=1)

params_ava = nwgrad.AlignParams(BLOSUM62,
                                gap_open_a=11.0, gap_extend_a=1.0,
                                gap_open_b=11.0, gap_extend_b=1.0)
batch = nwgrad.SeqPairBatch(n_threads=4, gap_model="affine", mode="global",
                            grad_mode="soft")
batch.add_many([sequences[i] for i in ii], [sequences[j] for j in jj], params_ava)

total_log_z = batch.score_and_grad()
grad = batch.compute_grad()
```

## Reading the alignments

`score_and_grad()` keeps scores, gradients and paths (`guide_j`), but not the
alignments themselves. Ask for them with `keep_paths=True` — a few bytes per alignment
column:

```python
batch.score_and_grad(keep_paths=True)
a_al, b_al = batch[0].aligned()
print(batch[0].formatted())
print(batch[0].coordinates())        # Biopython-style, for Bio.Align.Alignment
batch.drop_paths()                   # free them; scores and gradients stay
```

## Scoring without keeping the pairs

When you only need the scores and the summed gradient of a set of pairs once — not
their individual results, and not again — `align()` aligns them without adding them,
so its memory stays proportional to the thread count:

```python
result = batch.align(seqs_a, seqs_b, params)
print(result.scores, result.grad.matrix.to_matrix().sum())
```

It uses the batch's problem type, gradient mode and settings.

Next: [Tutorial 3 — Gradient-based substitution matrix optimisation](03_matrix_optimization.md)
