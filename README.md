# nwgrad

High-performance C++ sequence alignment library with **gradient computation** for substitution matrix optimization — exposed to Python via [nanobind](https://github.com/wjakob/nanobind).

nwgrad implements Needleman-Wunsch (global) and Smith-Waterman (local) alignment with both linear and affine gap penalties. Its primary novel feature is computing the **gradient of the alignment score with respect to substitution matrix entries**, enabling substitution matrix optimization in gradient-based ML pipelines.

## Features

- **Global (NW) and local (SW) alignment** — shared template core, selected at compile time
- **Linear and affine gap models** — single gap penalty or gap-open + gap-extend
- **Hard subgradient** — substitution-pair counts along the optimal traceback
- **Soft (differentiable) gradient** — forward-backward in log-space; log-partition function and expected substitution counts
- **Multithreaded batch processing** — lock-free work dispatch, per-thread gradient accumulation
- **Guide-banded DP** — cheap re-alignment under a new matrix around a cached alignment path
- **Double precision throughout** — no float truncation in the hot path
- **Zero-copy Python interface** — nanobind buffer protocol; no unnecessary array copies

## Installation

```bash
pip install nwgrad
```

Or from source:

```bash
git clone https://github.com/michalsta/nwgrad
cd nwgrad
pip install .
```

Requires Python ≥ 3.8 and a C++20 compiler (GCC 11+ or Clang 13+).

## Quick Start

```python
import numpy as np
import nwgrad

# Predefined matrices ship as ready-to-use SubstMatrix objects (they carry
# their own alphabet, so you never have to specify the symbol order yourself):
from nwgrad.matrices import BLOSUM62        # SubstMatrix, NCBI 23-symbol alphabet
print(BLOSUM62.alphabet)                    # 'ARNDCQEGHILKMFPSTWYVBZX'

# Or build one from your own (N, N) array.  A SubstMatrix always pairs a matrix
# with the alphabet that indexes its rows/columns:
dna_mat = nwgrad.SubstMatrix(np.eye(4) * 2 - 1, alphabet="ACGT")
```

### Single pair — `SeqPair`

`SeqPair` takes an `AlignParams` object that bundles the substitution matrix
with gap penalties.  Pass a `SubstMatrix` directly — its alphabet travels with
it, so amino acids map to matrix cells unambiguously:

```python
params = nwgrad.AlignParams(BLOSUM62,
                            gap_open_a=11.0, gap_extend_a=1.0,
                            gap_open_b=11.0, gap_extend_b=1.0)

sp = nwgrad.SeqPair(
    "PLEASANTLY", "MEANLY", params,
    gap_model="affine",   # "linear" | "affine"
    mode="global",        # "global" | "local"
    grad_mode="hard",     # "hard" | "soft" | "none"
)

# Align, then read the score and a printable alignment:
score, grad = sp.score_and_grad()   # one call: allocates DP, aligns, computes grad
print(score)
print(sp.formatted())               # PLEASANTLY / match line / -MEAN---LY
print(sp.aligned())                 # ('PLEASANTLY', '-MEAN---LY')

# `grad` is an AlignParams; inspect it as a dict:
g = grad.to_dict()
print(g["matrix"].shape, g["gap_open_a"], g["gap_extend_b"])
```

`align_full()` (the DP) and `compute_grad()` (reading the gradient off the
cached DP tables) are exposed separately so you can re-align many times and only
pay for the gradient when you need it; `score_and_grad()` above just fuses the
common case.  The hard gradient is a subgradient read from the Viterbi path; the
soft gradient (`grad_mode="soft"`) is the exact gradient of the log-partition
function via forward–backward.

After updating the matrix, `realign_banded()` re-scores the pair cheaply around the existing alignment path:

```python
new_params = nwgrad.AlignParams(new_subst_matrix,
                                gap_open_a=11.0, gap_extend_a=1.0,
                                gap_open_b=11.0, gap_extend_b=1.0)
sp.set_params(new_params)
sp.realign_banded(bandwidth=20)
sp.compute_grad()
```

### Batch — `SeqPairBatch`

```python
from nwgrad.matrices import BLOSUM62

params = nwgrad.AlignParams(BLOSUM62,
                            gap_open_a=11.0, gap_extend_a=1.0,
                            gap_open_b=11.0, gap_extend_b=1.0)
pairs = [
    nwgrad.SeqPair(a, b, params,
                   gap_model="affine", mode="global", grad_mode="soft")
    for a, b in zip(seqs_a, seqs_b)
]

batch = nwgrad.SeqPairBatch(n_threads=8)
for sp in pairs:
    batch.add(sp)

# Align all pairs in parallel; score and grad are cached on each SeqPair
total_log_z = batch.score_and_grad()

# Sum gradients across all pairs → AlignParams
grad = batch.compute_grad()

# Update matrix and re-run with banded DP around the existing paths
new_params = nwgrad.AlignParams(new_subst_matrix,
                                gap_open_a=11.0, gap_extend_a=1.0,
                                gap_open_b=11.0, gap_extend_b=1.0)
batch.set_params(new_params)
total_log_z = batch.score_and_grad(bandwidth=20)
grad = batch.compute_grad()
```

`SeqPair` objects are constructed once and reused across optimization iterations. Only the `AlignParams` changes; the sequence data and, when applicable, the alignment path are preserved.

## API Reference

### `SubstMatrix`

```python
nwgrad.SubstMatrix(matrix: np.ndarray, alphabet: str = "ACDEFGHIKLMNPQRSTVWY")
```

Constructs a substitution matrix from an `(N, N)` float64 numpy array where `N = len(alphabet)`. The default alphabet is the 20 canonical amino acids (`ACDEFGHIKLMNPQRSTVWY`), giving the same 20×20 behaviour as before. Any square matrix with a matching alphabet string is accepted — including DNA (`"ACGT"`), extended amino acids, or any other symbol set.

Stored internally as a `double[256][256]` ASCII-indexed table for O(1) lookup without char-to-index mapping. Asymmetric matrices are fully supported.

| Member | Description |
|---|---|
| `score(a, b)` | Substitution score for single characters `a`, `b` |
| `to_matrix()` | Export as `(N, N)` float64 numpy array in alphabet order |
| `size` | Alphabet size `N` |
| `alphabet` | The alphabet string (length `N`) |

**Predefined matrices** are available from `nwgrad.matrices` as ready-to-use
`SubstMatrix` objects (each already carrying the correct alphabet):
`BLOSUM45/50/62/80/90`, `PAM30/70/250`, `VTML40/80/160/200`, and `NUC44` (DNA).

```python
from nwgrad.matrices import BLOSUM62
BLOSUM62.alphabet            # 'ARNDCQEGHILKMFPSTWYVBZX'
BLOSUM62.to_matrix()         # (23, 23) float64 array
```

---

### `AlignParams`

```python
nwgrad.AlignParams(matrix, gap_open_a=0.0, gap_extend_a=0.0,
                           gap_open_b=0.0, gap_extend_b=0.0)
# or, from a raw array + explicit alphabet:
nwgrad.AlignParams(array, alphabet="ACDEFGHIKLMNPQRSTVWY", gap_open_a=0.0, ...)
```

Bundles a substitution matrix with (possibly asymmetric) gap penalties. The
first form takes a `SubstMatrix` directly and is preferred — the alphabet
travels with the matrix, so there is no risk of mismatching the symbol order.
Suffix `_a` penalties apply to gaps in sequence A, `_b` to gaps in sequence B.

| Member | Description |
|---|---|
| `matrix` | The `SubstMatrix` (read/write) |
| `gap_open_a` / `gap_extend_a` / `gap_open_b` / `gap_extend_b` | Gap penalties (read/write) |
| `to_dict()` | Return `{"matrix": (N,N) array, "alphabet": str, "gap_open_a": …, …}` |
| `+ - * ` | Element-wise arithmetic over all fields (for gradient-descent updates) |

`AlignParams` is also the gradient type returned by `SeqPair.grad` /
`compute_grad()`; `to_dict()` is the easy way to inspect a computed gradient.

---

### `SeqPair`

```python
nwgrad.SeqPair(
    seq_a, seq_b, params,   # params: AlignParams
    gap_model="affine",     # "linear" | "affine"
    mode="global",          # "global" | "local"
    grad_mode="hard",       # "hard" | "soft" | "none"
)
```

Persistent sequence-pair object. Sequences and alignment mode are fixed at
construction; the alignment parameters can be swapped cheaply via `set_params()`.
The pair holds its `AlignParams` alive automatically, so you don't need to keep
a separate reference to it.

**Methods:**

| Method | Description |
|---|---|
| `alloc_dp()` | Pre-allocate own DP tables (needed before `align_full()` / `realign_banded()`; not needed if using `SeqPairBatch.score_and_grad()`). |
| `align_full()` | Full DP alignment. Sets `score` and `guide_j`; clears `grad`. |
| `realign_banded(bandwidth)` | Banded DP around the current path. Requires prior `align_full()`. Sets `score`; clears `grad`. |
| `compute_grad()` | Compute and cache the gradient from the current alignment. Requires `align_full()` or `realign_banded()` to have been called first. |
| `score_and_grad()` | Convenience: `alloc_dp()` + `align_full()` + `compute_grad()` in one call. Returns `(score, grad)`. |
| `aligned()` | Return the alignment as a pair of gapped strings `(seq_a, seq_b)`. Requires the DP tables (call before `drop_dp()`). |
| `formatted(width=60)` | Pretty-printed alignment block (seq A / match line / seq B), wrapped at `width` columns (`0` = no wrap). |
| `set_params(params)` | Swap alignment parameters. Clears `score` and `grad`; preserves `guide_j`. |
| `drop_dp()` | Free O(m×n) DP table memory. Cached `score`, `grad`, and `guide_j` survive (but `aligned()` / `compute_grad()` then need a re-align). |

**Properties:**

| Property | Type | Description |
|---|---|---|
| `score` | `float \| None` | Alignment score (or `log Z` for soft), or `None` if not computed |
| `grad` | `AlignParams \| None` | Gradient, or `None` if not computed |
| `guide_j` | `list[int] \| None` | Alignment path (length m+1), or `None` if not computed |
| `seq_a`, `seq_b` | `str` | The fixed sequences |
| `path_valid` | `bool` | `guide_j` is usable as a banding guide |
| `score_valid` | `bool` | `score` matches current matrix and path |
| `grad_valid` | `bool` | `grad` is populated |
| `dp_valid` | `bool` | DP tables are in memory (`compute_grad()` is callable) |

**Gradient modes:**

| `grad_mode` | `score` | `grad` |
|---|---|---|
| `"hard"` | Viterbi alignment score | Substitution-pair counts (integer-valued subgradient) |
| `"soft"` | Log-partition function `log Z` | Expected substitution counts (true gradient of `log Z`) |
| `"none"` | Viterbi alignment score | `compute_grad()` throws |

---

### `SeqPairBatch`

```python
nwgrad.SeqPairBatch(n_threads=0)
```

`n_threads=0` (default) uses `hardware_concurrency`. `add()` keeps each `SeqPair` (and, transitively, its `AlignParams`) alive for the lifetime of the batch.

**Methods:**

| Method | Returns | Description |
|---|---|---|
| `add(seq_pair)` | — | Append a `SeqPair` |
| `set_params(params)` | — | Call `set_params()` on all pairs |
| `score_and_grad(bandwidth=0)` | `float` (sum of scores) | Full-pipeline parallel alignment. Uses per-thread DP buffers (pair-owned tables are never allocated). If `bandwidth > 0`, runs a full DP for the guide path then a banded DP. Results are cached on each `SeqPair`. |
| `compute_grad()` | `AlignParams` | Sum cached per-pair gradients. No DP work if all `grad_valid` are already true. |
| `align_full()` | `float` (sum of scores) | Full DP on all pairs in parallel using pair-owned buffers. Call `alloc_dp()` first. |
| `realign_banded(bandwidth)` | `float` (sum of scores) | Banded DP on all pairs in parallel using pair-owned buffers. |
| `alloc_dp()` | — | Pre-allocate pair-owned DP tables in parallel. |
| `drop_dp()` | — | Free pair-owned DP tables in parallel. |

**Properties:**

| Property | Type | Description |
|---|---|---|
| `n_threads` | `int` | Thread count |

**Typical optimization loop:**

```python
from nwgrad.matrices import BLOSUM62

# Learnable matrix starts from BLOSUM62; keep its alphabet to stay consistent.
alphabet = BLOSUM62.alphabet
mat_array = BLOSUM62.to_matrix()

# Build pairs once
params = nwgrad.AlignParams(nwgrad.SubstMatrix(mat_array, alphabet),
                            gap_open_a=11.0, gap_extend_a=1.0,
                            gap_open_b=11.0, gap_extend_b=1.0)
batch = nwgrad.SeqPairBatch(n_threads=8)
for a, b in zip(seqs_a, seqs_b):
    batch.add(nwgrad.SeqPair(a, b, params,
                             gap_model="affine", mode="global", grad_mode="soft"))

for step in range(n_steps):
    total_log_z = batch.score_and_grad(bandwidth=bw if step > 0 else 0)
    grad = batch.compute_grad()
    mat_array += lr * grad.matrix.to_matrix()   # both in `alphabet` order
    params = nwgrad.AlignParams(nwgrad.SubstMatrix(mat_array, alphabet),
                                gap_open_a=11.0, gap_extend_a=1.0,
                                gap_open_b=11.0, gap_extend_b=1.0)
    batch.set_params(params)
```

---

### `BatchAligner`

Stateless batch alignment: constructs a fresh DP buffer per call and does not preserve alignment paths across calls. Simpler API when you do not need to reuse paths for banded re-alignment.

```python
nwgrad.BatchAligner(
    matrix,
    gap_open=11.0,
    gap_extend=1.0,
    gap_model="affine",   # "linear" | "affine"
    mode="global",        # "global" | "local"
    grad_mode="hard",     # "hard" | "soft" | "none"
    n_threads=1,
)
```

**`.align(sequences_a, sequences_b) -> BatchResult`**

Aligns each pair `(sequences_a[i], sequences_b[i])`.

### `BatchResult`

| Attribute | Type | Description |
|---|---|---|
| `scores` | `np.ndarray[N]` float64 | Score (or log-partition for soft) per pair |
| `grad` | `AlignParams` | Gradient summed over all pairs |

---

### Single-pair convenience functions

Stateless functions that create and destroy their DP tables on every call. Useful for one-off alignments; prefer `SeqPair` / `SeqPairBatch` for loops over many pairs or iterative optimization.

**Score only** — return `float`:

| Function | Gap model | Alignment |
|---|---|---|
| `nw_score(a, b, matrix, gap_extend)` | Linear | Global (NW) |
| `sw_score(a, b, matrix, gap_extend)` | Linear | Local (SW) |
| `nw_score_affine(a, b, matrix, gap_open, gap_extend)` | Affine | Global (NW) |
| `sw_score_affine(a, b, matrix, gap_open, gap_extend)` | Affine | Local (SW) |

**Hard gradient** — return `(score: float, grad: AlignParams)`:

| Function | Gap model | Alignment |
|---|---|---|
| `nw_grad(a, b, matrix, gap_extend)` | Linear | Global |
| `sw_grad(a, b, matrix, gap_extend)` | Linear | Local |
| `nw_affine_grad(a, b, matrix, gap_open, gap_extend)` | Affine | Global |
| `sw_affine_grad(a, b, matrix, gap_open, gap_extend)` | Affine | Local |

**Soft gradient** — return `(log_z: float, grad: AlignParams)`:

| Function | Gap model | Alignment |
|---|---|---|
| `nw_soft_grad(a, b, matrix, gap_extend)` | Linear | Global |
| `sw_soft_grad(a, b, matrix, gap_extend)` | Linear | Local |
| `nw_affine_soft_grad(a, b, matrix, gap_open, gap_extend)` | Affine | Global |
| `sw_affine_soft_grad(a, b, matrix, gap_open, gap_extend)` | Affine | Local |

---

## Gap penalty conventions

For **linear** gap model, a gap of length `k` costs `gap_extend * k`.

For **affine** gap model, a gap of length `k` costs `gap_open + gap_extend * k`.

This matches BioPython's convention where `open` is charged once per gap regardless of length.

## Performance notes

- Full O(m×n) DP tables are retained for gradient computation — no Hirschberg-style memory optimisation.
- `SeqPairBatch.score_and_grad()` allocates one `DpBuffer` per thread (sized to the largest sequence pair in the batch) and reuses it across all assigned pairs. Pair-owned DP tables are never allocated, keeping peak memory at `n_threads × max(m×n)` rather than `N × max(m×n)`.
- Work is distributed via a shared `std::atomic` counter — no per-task mutex, no work-stealing queue.
- Gradient accumulation is per-thread; a single mutex is taken once at join to merge partial gradients.

## C++ header-only library

The C++ implementation is header-only (`src/nwgrad/cpp/nwgrad/`). To use it directly from C++:

```bash
python -m nwgrad --include
# prints: /path/to/nwgrad/cpp
```

Add that path to your include path and `#include "nwgrad/aligner.hpp"` etc.

## Building from source

```bash
pip install scikit-build-core nanobind
pip install -e ".[dev]"
pytest tests/Python/
```

C++ unit tests (requires CMake):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
ctest --test-dir build
```

## License

MIT — see [LICENSE](LICENSE).
