# nwgrad Python API reference

Every public class and function, with its arguments and contracts. For what the
library is and a quick start, see the [README](../README.md); for choosing precision,
traceback mode and SIMD kernel, see [tuning.md](tuning.md); for the C++ headers and
building from source, see [cpp.md](cpp.md).

## Contents

- [`Alphabet`](#alphabet) · [`SubstMatrix`](#substmatrix) · [`AlignParams`](#alignparams)
  (with [gap penalty conventions](#gap-penalty-conventions))
- [`SeqPair`](#seqpair) · [`SeqPairBatch`](#seqpairbatch)
- [`BatchAligner`](#batchaligner) · [`BatchResult`](#batchresult)
- [Single-pair convenience functions](#single-pair-convenience-functions)

## `Alphabet`

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

## `SubstMatrix`

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

## `AlignParams`

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


### Gap penalty conventions

For **linear** gap model, a gap of length `k` costs `gap_extend * k`.

For **affine** gap model, a gap of length `k` costs `gap_open + gap_extend * k`.

This matches BioPython's convention where `open` is charged once per gap regardless of length.

---

## `SeqPair`

```python
nwgrad.SeqPair(
    seq_a, seq_b, params,   # params: AlignParams
    gap_model="affine",     # "linear" | "affine"
    mode="global",          # "global" | "local"
    grad_mode="hard",       # "hard" | "soft" | "none"
    kernel="auto",          # Viterbi backend, see tuning.md#kernel-selection
    traceback="auto",       # see tuning.md#traceback-modes
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
| `aligned()` | Return the alignment as a pair of gapped strings `(seq_a, seq_b)`. Requires the DP tables of the current params: raises `RuntimeError` after `drop_dp()` or `set_params()` until the next `align_full()` / `realign_banded()`. With `grad_mode="soft"` it is the Viterbi alignment. |
| `formatted(width=60)` | Pretty-printed alignment block (seq A / match line / seq B), wrapped at `width` columns (`0` = no wrap). Same availability as `aligned()`. |
| `set_params(params)` | Swap alignment parameters. Clears `score` and `grad` and invalidates the retained DP tables (`dp_valid` becomes `False`, so `aligned()` raises until you re-align: the tables belong to the replaced params, which may already be freed). Preserves `guide_j`, so `realign_banded()` works straight after. |
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

## `SeqPairBatch`

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
| `weighted_grad(weights)` | `AlignParams` | `sum_i weights[i] * grad_i` over the cached per-pair gradients. `weights` is a 1-D numeric array with one entry per pair. Runs no DP; raises if any pair has no valid gradient. Summed in fixed blocks of 4096 pairs (each in pair order, the blocks in parallel), then over the blocks in order, so the result is bit-reproducible and does not depend on `n_threads`. |
| `grads()` | `(numpy.ndarray, numpy.ndarray)` (float64) | The cached per-pair gradients as two arrays, in pair order: `matrices` of shape `(N, n, n)`, rows and columns in the order of `alphabet`, and `gaps` of shape `(N, 4)` with columns `gap_open_a`, `gap_extend_a`, `gap_open_b`, `gap_extend_b`. The same numbers as `batch[i].grad`, without one `AlignParams` object per pair. Runs no DP; raises on an empty batch and if any pair has no valid gradient. |
| `align_full()` | `float` (sum of scores) | Full DP on all pairs in parallel using pair-owned buffers. Call `alloc_dp()` first. |
| `realign_banded(bandwidth)` | `float` (sum of scores) | Banded DP on all pairs in parallel using pair-owned buffers. |
| `banded_grad(bandwidth)` | `float` (sum of scores) | Banded re-align + gradient around each pair's cached path, using per-thread buffers. Run `score_and_grad()` once to establish the paths, then `set_params()` + `banded_grad(bw)` after each update. |
| `alloc_dp()` | — | Pre-allocate pair-owned DP tables in parallel. |
| `drop_dp()` | — | Free pair-owned DP tables in parallel. |

The batch is also a sequence: `len(batch)` is the number of pairs and `batch[i]`
returns the `i`-th `SeqPair` (negative indices allowed), so per-pair results can be
read back without keeping a separate list. A pair taken out this way keeps its batch alive, so it
stays usable after the batch itself is dropped.

Each `SeqPair` may appear in a batch **once**: adding the same object twice raises
`ValueError` at the next batch operation (the workers would otherwise align it
concurrently). The same pair may belong to several batches.

**Properties:**

| Property | Type | Description |
|---|---|---|
| `n_threads` | `int` | Thread count |
| `alphabet` | `str` | The symbols of the alphabet every pair shares (the row and column order of `grads()`). Raises on an empty batch. |
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

## `BatchAligner`

Stateless batch alignment: constructs a fresh DP buffer per call and does not preserve alignment paths across calls. Simpler API when you do not need to reuse paths for banded re-alignment.

```python
nwgrad.BatchAligner(
    params,               # AlignParams — matrix and gap penalties together
    band=0,               # 0 = full DP; >0 = banded half-width
    gap_model="affine",   # "linear" | "affine"
    mode="global",        # "global" | "local"
    grad_mode="hard",     # "hard" | "soft" | "none"
    n_threads=1,
    kernel="auto",        # Viterbi backend, see tuning.md#kernel-selection
)
```

**`.align(sequences_a, sequences_b, aligned_a=[], aligned_b=[]) -> BatchResult`**

Aligns each pair `(sequences_a[i], sequences_b[i])`. `aligned_a` / `aligned_b`, if
given, are gapped alignment strings (one per pair) used as banding guides. Each guide
must describe its own pair — see [guide validation](#guide-validation).

## `BatchResult`

| Attribute | Type | Description |
|---|---|---|
| `scores` | `np.ndarray[N]` float64 | Score (or log-partition for soft) per pair |
| `grad` | `AlignParams` | Gradient summed over all pairs |

---

## Single-pair convenience functions

Stateless functions that create and destroy their DP tables on every call. Useful for one-off alignments; prefer `SeqPair` / `SeqPairBatch` for loops over many pairs or iterative optimization.

All twelve share the same signature — the gap penalties live in the `AlignParams`,
not in the argument list, so the linear and affine variants differ only in which of
its gap fields they read:

```python
f(seq_a, seq_b, params, band=0, aligned_a="", aligned_b="", kernel="auto")
```

Each also exists with a `_double` suffix (`nw_score_double`, `nw_affine_grad_double`,
…) that runs the DP in float64 — see [Precision](tuning.md#precision).

`band > 0` (or a non-empty `aligned_a` / `aligned_b` guide pair) runs a banded DP
instead of the full one. The sequences are plain `str` and are validated and
encoded against `params`'s alphabet on the way in.

<a id="guide-validation"></a>**Guide validation.** A guide built from `aligned_a` /
`aligned_b` must describe the input pair, not merely be well formed: it needs one
entry per residue of `seq_a` plus one (i.e. the gapped `aligned_a` spells `seq_a`),
every column in `[0, len(seq_b)]`, never decreasing. Anything else raises
`ValueError` before the DP runs, as does a negative `band`. The last guide entry need
not equal `len(seq_b)`: B residues after the last A residue (trailing gaps in A) do not
add an entry.

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
