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
- **float32 by default, float64 on request** — the plain names (`SeqPair`, `nw_score`, …) run the Viterbi DP in float32; `SeqPairDouble`, `nw_score_double`, … run it in float64 (see [Precision](#precision))
- **Linear-space alignment** — Hirschberg (Myers-Miller) traceback is the default for affine global alignment, so long pairs no longer need O(m×n) tables
- **Runtime ISA dispatch** — one binary carries SSE2 / AVX2 / AVX-512 (x86) or NEON (ARM) kernels and picks the best one at load; every level is bit-exact with the scalar kernel
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

Prebuilt wheels cover CPython 3.9–3.14 on Linux x86_64 / aarch64 and macOS arm64.
Anywhere else pip builds from source, which needs a C++20 compiler with libstdc++'s
`<experimental/simd>`: GCC 11+, or Clang 13+ built against libstdc++. Apple's
system clang (libc++) and MSVC lack that header — on macOS use Homebrew `gcc`;
Windows is not supported.

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

# Or weight each pair's gradient, e.g. by a per-pair loss derivative computed
# from the scores.  Reads the cached gradients; no alignment is run.
scores = batch.scores()                     # float64 array, pair order
grad = batch.weighted_grad(weights)         # sum_i weights[i] * grad_i

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

### `Alphabet`

```python
nwgrad.Alphabet.get(symbols: str) -> Alphabet
```

The character↔index mapping that a matrix, its gradients, and its sequences all share. Alphabets are **interned**: `Alphabet.get("ACGT") is nwgrad.DNA`, so two matrices are compatible exactly when their alphabets are the same object.

Named alphabets:

| Constant | Symbols | N |
|---|---|---|
| `nwgrad.DNA` | `ACGT` | 4 |
| `nwgrad.DNA_N` | `ACGTN` | 5 |
| `nwgrad.RNA` | `ACGU` | 4 |
| `nwgrad.RNA_N` | `ACGUN` | 5 |
| `nwgrad.PROTEIN` | the canonical 20 | 20 |
| `nwgrad.PROTEIN_X` | + `X` (unknown) | 21 |
| `nwgrad.PROTEIN_UO` | + `U` (selenocysteine), `O` (pyrrolysine) | 22 |
| `nwgrad.PROTEIN_UOX` | + `U`, `O`, `X` | 23 |
| `nwgrad.NCBI_PROTEIN` | `ARNDCQEGHILKMFPSTWYVBZX` | 23 |
| `nwgrad.IUPAC_DNA` | `ATGCSWRYKMBVHDN` | 15 |

Extensions **append at the end**, so the canonical 20 keep indices 0–19 and a 20×20 matrix *in that order* embeds as the top-left block of any extended one.

**The matrices in `nwgrad.matrices` use a different ordering.** BLOSUM, PAM and VTML are all in NCBI column order (`NCBI_PROTEIN`), and `NUC44` is in IUPAC order (`IUPAC_DNA`). These are *not* extensions of `PROTEIN` / `DNA` — BLOSUM62 does not embed in `PROTEIN_X`. Combining a matrix over one alphabet with a gradient over another raises, rather than silently misreading the columns.

An `Alphabet` governs legality and ordering only — scoring is entirely the matrix's job, including for `X` and `N`.

For anything not listed, build your own: `nwgrad.Alphabet.get("ACGTRYSWKM")`.

| Member | Description |
|---|---|
| `symbols` | The symbol string, in index order |
| `size`, `len(a)` | Number of symbols `N` |
| `index_of(c)` | Index of `c`, or `-1` if absent |
| `contains(c)` | Whether `c` is in the alphabet |
| `encode(s)` | Validate and encode to indices; raises `ValueError` on any foreign character |
| `decode(indices)` | Inverse of `encode` |

### `SubstMatrix`

```python
nwgrad.SubstMatrix(matrix: np.ndarray, alphabet: str = "ACDEFGHIKLMNPQRSTVWY")
```

Constructs a substitution matrix from an `(N, N)` float64 numpy array where `N = len(alphabet)`. The default alphabet is the 20 canonical amino acids (`ACDEFGHIKLMNPQRSTVWY`). Any square matrix with a matching alphabet is accepted — including DNA (`"ACGT"`), extended amino acids, or any other symbol set. The alphabet is given as a string; if you are holding an `Alphabet` object, pass its `.symbols`.

Stored as a dense `N × N` block indexed by alphabet position — 16 doubles for DNA. Sequences are validated and encoded to indices once, at the boundary, so the DP inner loop is a single O(1) lookup and never touches a character. Asymmetric matrices are fully supported.

Characters outside the alphabet **raise `ValueError`**; they are not silently scored as zero. Case is significant — `'d'` is not `'D'`, so soft-masked FASTA must be upper-cased by the caller.

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

**Sign convention.** Every field of a gradient — the matrix entries *and* the four
gap fields — is the derivative of the score with respect to that field. The score
*subtracts* gap penalties, so a gradient's gap fields come out non-positive while
its matrix fields come out non-negative. That is what lets one update rule move
the whole struct in the ascent direction:

```python
params = params + lr * grad    # element-wise over matrix and all four gap fields
```

---

### `SeqPair`

```python
nwgrad.SeqPair(
    seq_a, seq_b, params,   # params: AlignParams
    gap_model="affine",     # "linear" | "affine"
    mode="global",          # "global" | "local"
    grad_mode="hard",       # "hard" | "soft" | "none"
    kernel="auto",          # Viterbi backend, see "Kernel selection"
    traceback="auto",       # see "Traceback modes"
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
| `traceback` | `str` | The traceback mode this pair *resolved* to (never `"auto"`) |
| `hb_cutoff` | `int` | Hirschberg base-case size in rows (settable; default 512) |

**Gradient modes:**

| `grad_mode` | `score` | `grad` |
|---|---|---|
| `"hard"` | Viterbi alignment score | Substitution-pair counts (integer-valued subgradient) |
| `"soft"` | Log-partition function `log Z` | Expected substitution counts (true gradient of `log Z`) |
| `"none"` | Viterbi alignment score | `compute_grad()` throws |

---

### `SeqPairBatch`

```python
nwgrad.SeqPairBatch(n_threads=0, traceback="auto")
```

`n_threads=0` (default) uses the number of *physical* cores (falling back to
`hardware_concurrency` where that cannot be determined): the DP is stall-bound, so
SMT siblings contend and the logical count measured up to 1.44× slower. `add()` keeps each `SeqPair` (and, transitively, its `AlignParams`) alive for the lifetime of the batch.

**Methods:**

| Method | Returns | Description |
|---|---|---|
| `add(seq_pair)` | — | Append a `SeqPair` |
| `add_many(seqs_a, seqs_b, params, gap_model="affine", mode="global", grad_mode="hard", kernel="auto")` | — | Build N `SeqPair`s in C++ and append them — much faster than N `add()` calls. The pairs take the batch's `traceback` and `hb_cutoff`. |
| `set_params(params)` | — | Call `set_params()` on all pairs |
| `score_and_grad(bandwidth=0)` | `float` (sum of scores) | Full-pipeline parallel alignment. Uses per-thread DP buffers (pair-owned tables are never allocated). If `bandwidth > 0`, runs a full DP for the guide path then a banded DP. Results are cached on each `SeqPair`. |
| `compute_grad()` | `AlignParams` | Sum cached per-pair gradients. No DP work if all `grad_valid` are already true. |
| `scores()` | `numpy.ndarray` (float64) | The cached per-pair scores, in pair order. Runs no DP; raises if any pair has no valid score. |
| `weighted_grad(weights)` | `AlignParams` | `sum_i weights[i] * grad_i` over the cached per-pair gradients. `weights` is a 1-D numeric array with one entry per pair. Runs no DP; raises if any pair has no valid gradient. Summed in pair order, so the result does not depend on `n_threads`. |
| `align_full()` | `float` (sum of scores) | Full DP on all pairs in parallel using pair-owned buffers. Call `alloc_dp()` first. |
| `realign_banded(bandwidth)` | `float` (sum of scores) | Banded DP on all pairs in parallel using pair-owned buffers. |
| `banded_grad(bandwidth)` | `float` (sum of scores) | Banded re-align + gradient around each pair's cached path, using per-thread buffers. Run `score_and_grad()` once to establish the paths, then `set_params()` + `banded_grad(bw)` after each update. |
| `alloc_dp()` | — | Pre-allocate pair-owned DP tables in parallel. |
| `drop_dp()` | — | Free pair-owned DP tables in parallel. |

The batch is also a sequence: `len(batch)` is the number of pairs and `batch[i]`
returns the `i`-th `SeqPair` (negative indices allowed), so per-pair results can be
read back without keeping a separate list.

**Properties:**

| Property | Type | Description |
|---|---|---|
| `n_threads` | `int` | Thread count |
| `traceback` | `str` | The batch's traceback mode as given (`"auto"` resolves per pair) |
| `hb_cutoff` | `int` | Hirschberg base-case size applied by `add_many()` (default 512) |
| `schedule` | `str` | `"dynamic"` (default; atomic counter) or `"sorted"` (length-sorted equal-work chunks — bounds peak DP memory). Results are identical either way. |

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
    params,               # AlignParams — matrix and gap penalties together
    band=0,               # 0 = full DP; >0 = banded half-width
    gap_model="affine",   # "linear" | "affine"
    mode="global",        # "global" | "local"
    grad_mode="hard",     # "hard" | "soft" | "none"
    n_threads=1,
    kernel="auto",        # Viterbi backend, see "Kernel selection"
)
```

**`.align(sequences_a, sequences_b, aligned_a=[], aligned_b=[]) -> BatchResult`**

Aligns each pair `(sequences_a[i], sequences_b[i])`. `aligned_a` / `aligned_b`, if
given, are gapped alignment strings (one per pair) used as banding guides.

### `BatchResult`

| Attribute | Type | Description |
|---|---|---|
| `scores` | `np.ndarray[N]` float64 | Score (or log-partition for soft) per pair |
| `grad` | `AlignParams` | Gradient summed over all pairs |

---

### Single-pair convenience functions

Stateless functions that create and destroy their DP tables on every call. Useful for one-off alignments; prefer `SeqPair` / `SeqPairBatch` for loops over many pairs or iterative optimization.

All twelve share the same signature — the gap penalties live in the `AlignParams`,
not in the argument list, so the linear and affine variants differ only in which of
its gap fields they read:

```python
f(seq_a, seq_b, params, band=0, aligned_a="", aligned_b="", kernel="auto")
```

Each also exists with a `_double` suffix (`nw_score_double`, `nw_affine_grad_double`,
…) that runs the DP in float64 — see [Precision](#precision).

`band > 0` (or a non-empty `aligned_a` / `aligned_b` guide pair) runs a banded DP
instead of the full one. The sequences are plain `str` and are validated and
encoded against `params`'s alphabet on the way in.

**Score only** — return `float`:

| Function | Gap model | Alignment |
|---|---|---|
| `nw_score` | Linear | Global (NW) |
| `sw_score` | Linear | Local (SW) |
| `nw_score_affine` | Affine | Global (NW) |
| `sw_score_affine` | Affine | Local (SW) |

**Hard gradient** — return `(score: float, grad: AlignParams)`:

| Function | Gap model | Alignment |
|---|---|---|
| `nw_grad` | Linear | Global |
| `sw_grad` | Linear | Local |
| `nw_affine_grad` | Affine | Global |
| `sw_affine_grad` | Affine | Local |

**Soft gradient** — return `(log_z: float, grad: AlignParams)`:

| Function | Gap model | Alignment |
|---|---|---|
| `nw_soft_grad` | Linear | Global |
| `sw_soft_grad` | Linear | Local |
| `nw_affine_soft_grad` | Affine | Global |
| `sw_affine_soft_grad` | Affine | Local |

(Note the naming is irregular: the affine *score* functions suffix `_affine`, while
the affine *gradient* functions infix it.)

```python
score, grad = nwgrad.nw_affine_grad("PLEASANTLY", "MEANLY", params)
```

---

## Precision

The plain names — `SeqPair`, `SeqPairBatch`, `BatchAligner` and the twelve
convenience functions — run the Viterbi DP in **float32**. The `*Double` classes
(`SeqPairDouble`, `SeqPairBatchDouble`, `BatchAlignerDouble`) and `_double`
functions run it in **float64**. Inputs and outputs are float64 either way
(`AlignParams`, scores, gradients); only the DP arithmetic differs. Over 400
protein pairs measured against a float64 reference, float32 scores deviated by up
to 8.4e-3 (mean 1.6e-3). The soft-gradient (forward-backward) path is float64 in both,
so the `_double` soft functions are identical to the plain ones.

The two precisions also have different traceback defaults (next section), so a
float32 result and a float64 result are not bit-comparable even where the
arithmetic would agree.

## Traceback modes

`SeqPair` and `SeqPairBatch` take `traceback=`, fixed at construction. It decides
what the DP retains in order to recover the alignment path:

| `traceback` | Memory | Notes |
|---|---|---|
| `"pointers"` | 3 B/cell | Records a predecessor byte per cell per state. |
| `"scores"` | 12 B/cell | Keeps the score tables and re-derives the path. Bit-identical to `"pointers"`, slower; for table introspection. |
| `"hirschberg"` | O(m+n) | Linear-space divide-and-conquer. Pairs no longer than `hb_cutoff` (default 512) never split and are bit-exact with `"pointers"`; longer pairs get an optimal path, but where alignments tie it may be a *different* optimal path (a different valid subgradient). Affine + full DP only (global or local). |
| `"hirschberg_pmax"` | O(m+n) | As `"hirschberg"`, with the gap carry computed as a closed-form prefix max: 1.5–3.9× faster on related sequences. The **only mode that can return a slightly suboptimal path** — measured worst case 4.9e-4 at float32, below float32's own error. Bit-identical across ISA levels. |
| `"auto"` (default) | — | Affine + global + full DP: `"hirschberg_pmax"` at float32, `"hirschberg"` at float64. Everything else: `"pointers"`. |

Asking explicitly for a Hirschberg mode on a linear-gap or banded problem raises;
`"auto"` falls back to `"pointers"` there instead. Local alignment supports
Hirschberg but `"auto"` keeps `"pointers"` for it, because local Hirschberg wins on
memory, not speed: select `traceback="hirschberg"` explicitly for very long local
pairs (e.g. >10k residues), where the pointer tables would not fit.

## Kernel selection

`kernel=` picks the Viterbi backend: `"auto"` (default — the strongest SIMD level
the CPU runs), `"scalar_fallback"`, or a named level: `"sse2"`, `"avx2"`, `"avx512"`
(x86) or `"neon"` (ARM). Every level is **bit-exact** with `"scalar_fallback"`
(same tables, paths and gradients), so this is purely a speed knob. An unknown
name, or a level this CPU cannot run, raises. It affects the Viterbi (hard / none)
path only; the linear gap model has no SIMD kernel, so any choice is a no-op there.

Module-level controls: `nwgrad.simd_isa()` reports the active level,
`nwgrad.available_isa_levels()` lists what this CPU runs, and
`nwgrad.set_isa_level(name)` (for testing; not thread-safe while work is in flight)
or the `NWGRAD_ISA` environment variable force the level that `"auto"` dispatches to. `nwgrad.compiled_with()` names the compiler the extension was built
with.

## Gap penalty conventions

For **linear** gap model, a gap of length `k` costs `gap_extend * k`.

For **affine** gap model, a gap of length `k` costs `gap_open + gap_extend * k`.

This matches BioPython's convention where `open` is charged once per gap regardless of length.

## Performance notes

- With `traceback="pointers"`, a hard-gradient alignment keeps 3 bytes per DP cell; `"scores"` keeps 12. The default for affine global alignment is Hirschberg, which keeps O(m+n) — so the per-thread buffer described below applies to the pointer modes and to the Hirschberg base case (≤ `hb_cutoff` rows).
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

C++ unit tests (requires CMake). The extension target does `find_package(nanobind)`,
so even a tests-only configure needs nanobind discoverable:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug \
      -Dnanobind_DIR="$(python -m nanobind --cmake_dir)"
cmake --build build
ctest --test-dir build
```

`-DNWGRAD_SANITIZE=ON` builds the C++ tests with AddressSanitizer and
UndefinedBehaviorSanitizer (`-fno-sanitize-recover=all`, so the first violation
fails the run rather than printing and carrying on). CI runs this under both GCC
and Clang.

## License

MIT — see [LICENSE](LICENSE).
