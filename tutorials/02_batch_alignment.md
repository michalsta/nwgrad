# Tutorial 2 — Batch Alignment with SeqPairBatch

This tutorial covers `SeqPairBatch`: aligning many sequence pairs in parallel
and collecting scores and gradients efficiently.

## Why `SeqPairBatch`?

`SeqPairBatch` is designed for the workload that appears in substitution matrix
optimisation: the same set of sequence pairs is aligned repeatedly as the matrix
changes. Compared to looping over `SeqPair` objects in Python:

1. **Thread-level parallelism** — pairs are distributed across CPU cores via a
   lock-free work queue.
2. **Memory efficiency** — `score_and_grad()` allocates one DP buffer per thread
   (sized to the largest pair) and reuses it across all assigned pairs. Peak
   memory is `n_threads × max(m×n)`, not `N × max(m×n)`.
3. **Path reuse** — alignment paths (guide_j) are cached on each `SeqPair`.
   After `set_matrix()`, `score_and_grad(bandwidth=bw)` runs a cheap banded DP
   around the old path rather than a full DP.

## Basic usage

```python
import numpy as np
import nwgrad

# ... construct blosum62 and BLOSUM62 array as in Tutorial 1 ...

seqs_a = ["PLEASANTLY", "ACDEFGHIKL", "MADEEKLF", "ACDE"]
seqs_b = ["MEANLY",     "CDEFGHIKLM", "MADEEKLF", "ACDF"]

# AlignParams bundles the matrix with gap penalties.
params = nwgrad.AlignParams(BLOSUM62,
                             gap_open_a=11.0, gap_extend_a=1.0,
                             gap_open_b=11.0, gap_extend_b=1.0)

# Build SeqPair objects once; they will be reused across iterations.
# SeqPair stores a reference to params — keep it alive.
pairs = [
    nwgrad.SeqPair(a, b, params,
                   gap_model="affine",
                   mode="global",
                   grad_mode="soft")
    for a, b in zip(seqs_a, seqs_b)
]

batch = nwgrad.SeqPairBatch(n_threads=4)   # 0 = hardware_concurrency
for sp in pairs:
    batch.add(sp)
```

`SeqPairBatch` holds non-owning references to `SeqPair` objects — each `SeqPair`
must remain alive for the lifetime of the batch.

## `score_and_grad()`

The primary batch operation. Aligns all pairs in parallel using per-thread DP
buffers, then caches the score, gradient, and alignment path on each `SeqPair`.

```python
total_log_z = batch.score_and_grad()
print(f"Sum of log Z: {total_log_z:.3f}")

# Access per-pair results
for sp in pairs:
    print(f"  score={sp.score:.2f}  grad_valid={sp.grad_valid}")

# Sum gradients across all pairs → AlignParams
grad = batch.compute_grad()
print(f"Gradient matrix shape: {grad.matrix.to_matrix().shape}")   # (N, N)
```

`compute_grad()` reads the cached per-pair gradients. No DP work is done if all
pairs already have `grad_valid == True`.

## Updating the matrix and banded re-alignment

After a gradient step, call `set_matrix()` on the batch to push the new matrix
to all pairs at once. The alignment paths are preserved so the next call to
`score_and_grad(bandwidth=bw)` can use banded DP instead of full DP:

```python
mat_array = BLOSUM62.copy()

for step in range(20):
    total = batch.score_and_grad(bandwidth=30 if step > 0 else 0)
    grad  = batch.compute_grad()

    mat_array += 0.01 * grad.matrix.to_matrix()
    params = nwgrad.AlignParams(mat_array,
                                 gap_open_a=11.0, gap_extend_a=1.0,
                                 gap_open_b=11.0, gap_extend_b=1.0)
    batch.set_params(params)

    print(f"step {step:2d}  total log Z = {total:.2f}")
```

On the first iteration `bandwidth=0` forces a full DP (no path exists yet).
Subsequent iterations use banded DP around the previous alignment path.

The `bandwidth` is the half-width of the band in DP cells. If the true optimal
path under the new matrix lies outside the band, the score is silently
sub-optimal. Wider bands are safer but slower.

## Gradient modes

| `grad_mode` | `score` per pair | `compute_grad()` result |
|---|---|---|
| `"soft"` | log-partition `log Z` | Expected substitution counts (differentiable) |
| `"hard"` | Viterbi alignment score | Substitution-pair counts (integer-valued) |
| `"none"` | Viterbi alignment score | Zero matrix |

Use `"none"` when you only need scores — it skips the traceback/backward pass.

## All four alignment modes

`SeqPairBatch` supports all combinations; the mode is fixed per `SeqPair` at
construction time. You can mix modes in the same batch if needed.

```python
params_lin = nwgrad.AlignParams(BLOSUM62, gap_extend_a=1.0, gap_extend_b=1.0)
batch_lin_global = nwgrad.SeqPairBatch(n_threads=4)
for a, b in zip(seqs_a, seqs_b):
    batch_lin_global.add(nwgrad.SeqPair(
        a, b, params_lin,
        gap_model="linear", mode="global", grad_mode="hard",
    ))

total = batch_lin_global.score_and_grad()
grad  = batch_lin_global.compute_grad()
```

## Thread count and reproducibility

Results are identical regardless of thread count — per-thread gradient
accumulations are merged under a mutex at join time, and floating-point addition
order is deterministic within each thread.

```python
params_orig = nwgrad.AlignParams(BLOSUM62,
                                  gap_open_a=11.0, gap_extend_a=1.0,
                                  gap_open_b=11.0, gap_extend_b=1.0)
ref_total = None
for n in [1, 2, 4, 8]:
    b = nwgrad.SeqPairBatch(n_threads=n)
    for sp in pairs:
        b.add(sp)
    # Re-run from a clean state
    for sp in pairs:
        sp.set_params(params_orig)
    total = b.score_and_grad()
    grad  = b.compute_grad()
    if ref_total is None:
        ref_total = total
    assert abs(total - ref_total) < 1e-10
```

`n_threads=0` (the default) uses `hardware_concurrency`. Thread count is
automatically clamped to the number of pairs.

## All-vs-all pairs

Batch input is a flat list — not necessarily same-length lists. To align every
sequence against every other:

```python
sequences = ["ACDEFG", "MADEEKLF", "PLEASANTLY", "ACDE", "CDEFGHIKLM"]

params_ava = nwgrad.AlignParams(BLOSUM62,
                                 gap_open_a=11.0, gap_extend_a=1.0,
                                 gap_open_b=11.0, gap_extend_b=1.0)
batch = nwgrad.SeqPairBatch(n_threads=4)
for i in range(len(sequences)):
    for j in range(i + 1, len(sequences)):
        batch.add(nwgrad.SeqPair(
            sequences[i], sequences[j], params_ava,
            gap_model="affine", mode="global", grad_mode="soft",
        ))

total_log_z = batch.score_and_grad()
grad = batch.compute_grad()
```

## Memory management

`score_and_grad()` uses per-thread DP buffers and never allocates pair-owned DP
tables (`dp_valid` remains `False`). The gradient is cached on each `SeqPair`.

If you need pair-owned DP tables (e.g., to call `compute_grad()` after
`align_full()` / `realign_banded()` individually), call `alloc_dp()` first:

```python
batch.alloc_dp()         # allocates O(mn) per pair — can be large
batch.align_full()
# compute_grad() on individual pairs is now possible
batch.compute_grad()     # still works as batch operation
batch.drop_dp()          # free the pair-owned tables; cached results survive
```

In practice, `score_and_grad()` is preferred because it avoids allocating N
full DP tables simultaneously.

Next: [Tutorial 3 — Gradient-based substitution matrix optimisation](03_matrix_optimization.md)
