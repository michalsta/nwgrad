# nwgrad Python API reference

Every public class and function, with its arguments and contracts. For what the
library is and a quick start, see the [README](../README.md); for choosing precision,
traceback mode and SIMD kernel, see [tuning.md](tuning.md); for the C++ headers and
building from source, see [cpp.md](cpp.md).

## Contents

- [`Alphabet`](#alphabet) · [`SubstMatrix`](#substmatrix) · [`AlignParams`](#alignparams)
  (with [gap penalty conventions](#gap-penalty-conventions))
- [`SeqPair`](#seqpair) · [`SeqPairBatch`](#seqpairbatch) · [`nwgrad.logistic`](#nwgradlogistic)
  · [`BatchResult`](#batchresult)
- [Migrating from 0.5](#migrating-from-05)
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

**Pairs and batches copy their params.** `SeqPair(...)`, `add_many()` and
`set_params()` take a copy, so an `AlignParams` is never shared with the pairs built
from it: updating it in place (`+=`, `*=`, the setters) changes nothing they compute,
and their cached scores stay valid. Hand the new values over explicitly:

```python
params += lr * grad
batch.set_params(params)       # the batch sees the update from here on
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

One sequence pair, aligned on the calling thread. Sequences and problem type are fixed
at construction; the parameters can be swapped cheaply with `set_params()`. The pair
holds its own **copy** of the `AlignParams`: you don't need to keep a reference, and
changing the object afterwards (`p *= 2`, `p.gap_open_a = …`) does not affect the pair
until you pass it to `set_params()`.

A `SeqPair` is also what `batch[i]` returns: a **view** of pair `i` of a
[`SeqPairBatch`](#seqpairbatch), with the same methods and properties. A view's
results are the batch's (a view never copies), and its settings — params, `fill`, the
soft options, `hb_cutoff` — belong to the batch: setting them on a view raises
`RuntimeError`; set them on the batch.

**Methods:**

| Method | Description |
|---|---|
| `align_full()` | Full DP: sets `score`, `guide_j` and the stored alignment path. The gradient is computed too but **held**: `grad` stays `None` until `compute_grad()`. |
| `realign_banded(bandwidth)` | Banded DP around the current path, under the current params. Requires a prior `align_full()` (or a batch `score_and_grad()`). Same holding of the gradient. |
| `compute_grad()` | Release the gradient of the last `align_full()` / `realign_banded()`. Raises if there is none (no alignment yet, `set_params()` or `drop_dp()` since, or `grad_mode="none"`). |
| `score_and_grad()` | `align_full()` + `compute_grad()`. Returns `(score, grad)`; raises if `grad_mode` is `"none"`. |
| `aligned()` | The alignment as a pair of gapped strings `(seq_a, seq_b)`. Needs a stored path: raises after `drop_dp()` or `set_params()` until the next align. With `grad_mode="soft"` it is the Viterbi alignment. |
| `coordinates()` | The alignment as Biopython-style coordinates: an `int64` array of shape `(2, k)`, row 0 positions in `seq_a`, row 1 in `seq_b`, with a column at the start, at each change between aligned and gap columns, and at the end. `Bio.Align.Alignment([seq_a, seq_b], coordinates)` rebuilds it. A local alignment starts where its path starts. Same availability as `aligned()`. |
| `formatted(width=60)` | Pretty-printed alignment block (seq A / match line / seq B), wrapped at `width` columns (`0` = no wrap). Same availability as `aligned()`. |
| `set_params(params)` | (Standalone pairs.) Swap alignment parameters. Clears `score`, `grad` and the stored path; preserves `guide_j`, so `realign_banded()` works straight after. |
| `drop_dp()` | Free the stored path. `score`, a released `grad` and `guide_j` survive. |
| `alloc_dp()` | **Deprecated** no-op (there are no per-pair DP tables any more). |

**Properties:**

| Property | Type | Description |
|---|---|---|
| `score` | `float \| None` | Alignment score (or `log Z` for soft), or `None` if not computed |
| `grad` | `AlignParams \| None` | Gradient, or `None` if not computed (or held) |
| `guide_j` | `list[int] \| None` | Alignment path (length m+1), or `None` if not computed |
| `seq_a`, `seq_b` | `str` | The fixed sequences |
| `gap_model`, `mode`, `grad_mode` | `str` | The problem type |
| `path_valid` | `bool` | `guide_j` is usable as a banding guide |
| `score_valid` | `bool` | `score` matches the current params |
| `grad_valid` | `bool` | `grad` is populated |
| `dp_valid` | `bool` | A stored path is available (`aligned()`, `coordinates()`) |
| `traceback` | `str` | The traceback mode the pair *resolved* to (never `"auto"`) |
| `hb_cutoff` | `int` | Hirschberg base-case size in rows (settable on a standalone pair; default 512) |
| `fill` | `str` | Full-DP simd fill at double precision: `"striped"` (default) or `"rowwise"` (settable on a standalone pair; see `SeqPairBatch.fill`) |
| `soft_impl`, `soft_temperature` | | As on `SeqPairBatch` (settable on a standalone pair) |

**Gradient modes:**

| `grad_mode` | `score` | `grad` |
|---|---|---|
| `"hard"` | Viterbi alignment score | Substitution-pair counts (integer-valued subgradient) |
| `"soft"` | Log-partition function `log Z` | Expected substitution counts (true gradient of `log Z`) |
| `"none"` | Optimal alignment score, **score only** (see below) | `compute_grad()` raises |

<a id="score-only"></a>**`grad_mode="none"` is score only.** No DP table is kept and no
traceback runs: rolling rows (O(n) memory) through a SIMD kernel per pair, and the
inter-pair kernel for short pairs (one pair per vector lane, one row in place). The
score is the exact optimum — bit-identical to a `traceback="pointers"` pair's at every
`kernel=` level — so it does not depend on `traceback`: under float32's `auto` default
(`hirschberg_pmax`), the hard and soft modes report the score replayed along the
prefix-max path instead, which can sit a hair below. A batch's `score_and_grad()` in this
mode defers the guides `banded_grad()` bands around: each is computed on first use
(`banded_grad()`, `batch[i].guide_j`) under the params that were scored, so it is
exactly the guide the traceback would have stored, even after `set_params()`.
`keep_paths=True` runs the traceback as before. Measured against 0.6's table-filling
path on single pairs (AVX2, 1 thread): 1.2–3.1× affine Global, 3–6.5× affine Local,
1.5–14.5× linear; it keeps that lead with threads, having no table to stream.

---

## `SeqPairBatch`

```python
nwgrad.SeqPairBatch(
    n_threads=0, traceback="auto", *,
    gap_model="affine",     # "linear" | "affine"
    mode="global",          # "global" | "local"
    grad_mode="hard",       # "hard" | "soft" | "none"
)
```

Many sequence pairs of **one problem type** — one gap model, alignment mode and grad
mode — aligned in parallel. Each pair's state (its encoded sequences, score, guide and
gradient: a few hundred bytes) lives in the batch's own arrays, and the DP runs on
per-thread buffers. Naming any of `gap_model` / `mode` / `grad_mode` fixes the type
(the others take their defaults); naming none is the deprecated pre-0.6 form, where the
first `add_many()` fixes it (see [Migrating from 0.5](#migrating-from-05)).

`n_threads=0` (default) uses the number of *physical* cores among the CPUs the
process may run on — its affinity mask, so `taskset` and container CPU pinning are
respected (a cgroup CPU *quota* is not) — falling back to the allowed logical count
where the topology cannot be read: the DP is stall-bound, so SMT siblings contend and
the logical count measured up to 1.44× slower.

**Methods:**

| Method | Returns | Description |
|---|---|---|
| `add_many(seqs_a, seqs_b, params, kernel="auto")` | — | Append the pairs `(seqs_a[i], seqs_b[i])` under `params`: one **segment**. Params may differ between calls (one alphabet); `set_params()` replaces them all. Sequences are validated and encoded in parallel; a bad character raises and adds nothing. The batch keeps a **copy** of `params` (calls with equal params share one); later changes to the object do not reach the batch — use `set_params()`. |
| `set_params(params)` | — | Give every pair a copy of `params` (same alphabet; a mismatch raises without changing anything). Clears cached scores, gradients and stored paths; keeps the guides, for `banded_grad()`. |
| `score_and_grad(keep_paths=False)` | `float` (sum of scores) | Full DP on every pair: score, guide and (unless `grad_mode="none"`) gradient, cached per pair. With `grad_mode="none"` and no `keep_paths`: [score only](#score-only), guides deferred. `keep_paths=True` also stores each alignment path, so `batch[i].aligned()` / `.coordinates()` work (a few bytes per alignment column). |
| `banded_grad(bandwidth, keep_paths=False)` | `float` (sum of scores) | Banded re-align + gradient around each pair's cached guide. Run `score_and_grad()` once to establish the guides, then `set_params()` + `banded_grad(bw)` after each update; no full DP is run. |
| `align(seqs_a, seqs_b, params, band=0, aligned_a=[], aligned_b=[], kernel="auto")` | [`BatchResult`](#batchresult) | Align pairs **without adding them** (what `BatchAligner` was): scores plus the gradient summed over the pairs, nothing kept per pair, so memory stays O(threads). Uses the batch's type, grad mode, threads and settings. `band > 0` bands every pair (`aligned_a` / `aligned_b`, one gapped string per pair, give the guides; the diagonal otherwise); `band == 0` with guides bands at width 0, so all pairs or none must carry one. Soft: forward-backward only (no Viterbi). None, unbanded: [score only](#score-only). |
| `compute_grad()` | `AlignParams` | The cached gradients summed over all pairs. Summed in fixed blocks, so bit-reproducible whatever `n_threads`. Raises on an empty batch. |
| `scores()` | `numpy.ndarray` (float64) | The cached per-pair scores, in pair order. Runs no DP; raises if any pair has no valid score. |
| `weighted_grad(weights)` | `AlignParams` | `sum_i weights[i] * grad_i` over the cached per-pair gradients. `weights` is a 1-D numeric array with one entry per pair. Runs no DP; raises if any pair has no valid gradient. Summed in fixed blocks of 4096 pairs (each in pair order, the blocks in parallel), then over the blocks in order, so the result is bit-reproducible and does not depend on `n_threads`. |
| `grads()` | `(numpy.ndarray, numpy.ndarray)` (float64) | The cached per-pair gradients as two arrays, in pair order: `matrices` of shape `(N, n, n)`, rows and columns in the order of `alphabet`, and `gaps` of shape `(N, 4)` with columns `gap_open_a`, `gap_extend_a`, `gap_open_b`, `gap_extend_b`. Runs no DP; raises on an empty batch and if any pair has no valid gradient. |
| `drop_paths()` | — | Free the stored alignment paths. |

The batch is also a sequence: `len(batch)` is the number of pairs and `batch[i]`
returns pair `i` as a [`SeqPair`](#seqpair) view (negative indices allowed). A view
keeps its batch alive, so it stays usable after the batch itself is dropped.

**Properties:**

| Property | Type | Description |
|---|---|---|
| `n_threads` | `int` | Thread count |
| `gap_model`, `mode`, `grad_mode` | `str \| None` | The batch's problem type (`None` while a deprecated untyped batch has no pairs) |
| `alphabet` | `str` | The symbols of the alphabet every pair shares (the row and column order of `grads()`). Raises on an empty batch. |
| `traceback` | `str` | The traceback mode as constructed (`"auto"` reported as such; `batch[i].traceback` shows what it resolves to) |
| `hb_cutoff` | `int` | Hirschberg base-case size in rows (default 512), for every pair |
| `schedule` | `str` | `"dynamic"` (default; atomic counter) or `"sorted"` (length-sorted equal-work chunks — bounds peak DP memory) for the per-pair fills. Results are identical either way. |
| `fill` | `str` | How `score_and_grad()` / `banded_grad()` / `align()` vectorize the DP: `"interpair"` (default), `"striped"` or `"rowwise"`. Hard results are bit-identical whichever fill runs. See below. |
| `soft_impl` | `str` | Soft-path evaluation: `"scaled"` (default; scaled probability space, raises `ValueError` for a pair out of its range), `"scaled_or_log"` (falls back to log space per pair, silently) or `"log"` (the log-space recurrences). |
| `soft_temperature` | `float` | Soft score `T·log Z(θ/T)`; the gradient is the expected counts under `θ/T`. Default 1. |
| `soft_guide` | `str` | When soft pairs get the guide path that `banded_grad()` bands around: `"eager"` (default; `score_and_grad()` runs the guide Viterbi), `"lazy"` (computed on first use, under the params current then) or `"posterior"` (no Viterbi: per row of A the column where the posterior path most likely leaves the row, made non-decreasing — a band centre from the soft pass itself, under the params scored). |

**Choosing a fill.** All three fills compute bit-identical tables, so scores, paths and
hard gradients do not depend on the choice; only speed and memory do (soft results are
tolerance-equal, as the soft path always is). A pair that a fill cannot take, or would
slow down, silently runs its own fill instead, with the same results.

| `fill` | How | Use for |
|---|---|---|
| `"interpair"` (default) | `score_and_grad()` and `banded_grad()` align W pairs at once, one per vector lane (W = 2 double / 4 float32 on SSE2/NEON, 4 / 8 on AVX2, 8 / 16 on AVX-512), grouping pairs by length | Batches of short pairs, e.g. miRNA × target site, peptides |
| `"striped"` | Every pair its own fill: striped vectors within one pair (Farrar's layout, lazy-F gap correction) for affine double, the float32 / linear fills otherwise | Long pairs only — which `"interpair"` already sends there |
| `"rowwise"` | As `"striped"`, but the affine double fill goes one row of one pair at a time; no lazy-F fixpoint, whose cost grows when gaps are cheap | Short pairs on a single `SeqPairDouble` |

Which pairs each fill applies to:

- **`"interpair"`** (`SeqPairBatch` / `SeqPairBatchDouble`, including `align()`): both precisions, affine and linear gaps, any alphabet (over 8
  letters the substitution scores are gathered per row), hard and soft pairs — the soft
  forward-backward is shared too, and a soft pair's guide Viterbi rides the shared fill —
  full DP and, for hard affine pairs, `banded_grad()`. Pairs in one group must share
  their parameters; B lengths in one group may differ by up to 1.25× + 4 (each lane is
  padded to the longest). Run their own fill instead: pairs whose group tables would
  leave L2 (1.25 MiB; 512 KiB for the soft pass — long pairs, where the per-pair fills
  win), hard linear Global pairs below 4 lanes or with alphabets over 8 letters (their
  own fill is cheaper), empty sequences, Hirschberg pairs longer than `hb_cutoff`, and
  `kernel="scalar_fallback"`.
- **`"rowwise"`**: affine gaps at double precision on a simd kernel. Float32, linear
  gaps, `kernel="scalar_fallback"` and Hirschberg pairs longer than `hb_cutoff` ignore
  it.

Measured on an i5-12500 (AVX2), `score_and_grad()`: on miRNA × target-site pairs
(A ~22, B = 50) `"interpair"` takes 0.18–0.29× `"striped"`'s time at float32 and
0.30–0.39× at double (one thread; 0.33–0.48× / 0.38–0.58× at 12 threads), the first
call's pair grouping included; on 10–120-residue protein pairs 0.21–0.50× (affine).
For affine gaps `"rowwise"` and `"interpair"` keep three score tables per thread (per
group of pairs for `"interpair"`) even with `traceback="pointers"`. The `"interpair"`
grouping is built once and reused until pairs are added.

**Typical optimization loop:**

```python
import numpy as np
import nwgrad
from nwgrad.matrices import BLOSUM62

# The learnable matrix starts from BLOSUM62; keep its alphabet to stay consistent.
alphabet = BLOSUM62.alphabet
mat_array = BLOSUM62.to_matrix()

def make_params(m):
    return nwgrad.AlignParams(nwgrad.SubstMatrix(m, alphabet),
                              gap_open_a=11.0, gap_extend_a=1.0,
                              gap_open_b=11.0, gap_extend_b=1.0)

batch = nwgrad.SeqPairBatch(n_threads=8, gap_model="affine", mode="global",
                            grad_mode="soft")
batch.add_many(seqs_a, seqs_b, make_params(mat_array))    # encoded once

batch.score_and_grad()                  # full DP: establishes every pair's guide
for step in range(n_steps):
    grad = batch.compute_grad()
    mat_array += lr * grad.matrix.to_matrix()             # both in `alphabet` order
    batch.set_params(make_params(mat_array))
    total_log_z = batch.banded_grad(bw)  # re-align around the cached guides only
```

---

## `nwgrad.logistic`

A binary logistic link over per-pair alignment scores, P(y = 1) = expit(α + score): the
likelihood [DiscrimAlign](https://github.com/BioGeMT/DiscrimAlign) maximises. It lives in
its own module (`src/nwgrad/cpp/nwgrad/logistic/`) and uses only `SeqPairBatch`'s public
interface. Every sum is taken over fixed blocks of 4096 elements and the block sums are
added in order, so results do not depend on `n_threads` (0 = the default thread count).
`labels` are float64 arrays with each label in [0, 1]: 0/1 class labels, or soft labels
(target probabilities, for which the likelihood is the Bernoulli cross-entropy). They must
not be all 0 or all 1, or the likelihood has no finite maximum in α. **Labels are not
checked**: they are the caller's responsibility, and labels that break these conditions
give a meaningless `alpha` instead of an error. The log-likelihood itself is not provided (`log_likelihood()` and
`Step.loglik_at_alpha0` were removed in 0.6.0: they clipped probabilities to
[ε, 1 − ε]). Evaluate it from the logits z = α + score as Σ y·z − Σ log(1 + eᶻ), e.g.
`np.dot(labels, z) - np.logaddexp(0, z).sum()`, which needs no clipping.

| Function | Returns | Description |
|---|---|---|
| `step(batch, labels, alpha0)` | `Step` | One optimisation iteration's logistic work after `batch.score_and_grad()`: the fitted `alpha` (as `fit_alpha`) and `grad = Σᵢ (labels[i] − expit(alpha + scoreᵢ)) gradᵢ` as an `AlignParams` (as `weighted_grad` returns it). Uses the batch's `n_threads`. `SeqPairBatch` and `SeqPairBatchDouble`. |
| `fit_alpha(scores, labels, alpha0, n_threads=0, tol=1e-12, max_newton=8, maxiter=200)` | `float` | The intercept maximising the likelihood: the root of dL/dα = Σ(y − p), which is strictly decreasing in α. Plain Newton steps while \|step\| ≤ 1, otherwise a bracketed safeguarded Newton (Numerical Recipes' rtsafe). Exact to rounding from any start. A NaN or infinite score (any non-finite derivative), a non-finite `alpha0` or a negative `tol` raises `ValueError`. |
| `probabilities(scores, alpha, n_threads=0)` | `numpy.ndarray` | expit(α + scores), bit-identical to `scipy.special.expit`. |

## `BatchResult`

Returned by `SeqPairBatch.align()`.

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
`ValueError` before the DP runs, as does a negative `band` or giving only one of
`aligned_a` / `aligned_b`. A `band` of `len(seq_b)` or more covers whole rows, i.e. it
is the full DP. The last guide entry need not equal `len(seq_b)`: B residues after the
last A residue (trailing gaps in A) do not add an entry.

**Score only** — return `float`. They run the [score-only](#score-only) kernels (no
table, no traceback; `band > 0` runs the banded fill): the exact optimum, bit-identical to
a `traceback="pointers"` pair's.

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

## Migrating from 0.5

0.6 restructured the batch layer: a `SeqPairBatch` holds pairs of **one problem type**
in its own arrays (about 10× less memory per pair, faster batch-wide steps), and
`BatchAligner` became `SeqPairBatch.align()`. Old calls either still work with a
`DeprecationWarning`, or raise with a message that names the replacement:

| 0.5 | 0.6 | |
|---|---|---|
| `SeqPairBatch(n)` + `add_many(A, B, p, gap_model=…, mode=…, grad_mode=…)` | `SeqPairBatch(n, gap_model=…, mode=…, grad_mode=…)` + `add_many(A, B, p)` | deprecated, works: the first `add_many()` fixes the type (missing names take the old defaults `"affine"` / `"global"` / `"hard"`); later calls must agree |
| one batch mixing gap models, modes or grad modes | one batch per problem type | raises `ValueError` |
| `batch.add(seq_pair)` | `batch.add_many([...], [...], params)` | removed |
| `BatchAligner(p, band, gap_model, mode, grad_mode, n_threads, kernel).align(A, B, ga, gb)` | `SeqPairBatch(n_threads, gap_model=…, mode=…, grad_mode=…).align(A, B, p, band, ga, gb, kernel)` | removed; same results |
| `batch.alloc_dp(); batch.align_full()` | `batch.score_and_grad(keep_paths=True)` | deprecated, works (the gradient is held until `compute_grad()`, as before) |
| `batch.realign_banded(bw)` | `batch.banded_grad(bw, keep_paths=True)` | deprecated, works |
| `batch.drop_dp()` | `batch.drop_paths()` | deprecated, works |
| `pair.alloc_dp()` | (nothing) | deprecated no-op |
| `batch[i].set_params(p)`, `batch[i].hb_cutoff = …`, `.fill`, `.soft_impl`, `.soft_temperature` | set them on the batch | raises: a pair in a batch is a view, its settings are the batch's (`hb_cutoff` now applies to every pair) |
| `batch[i] is batch[i]` | — | `batch[i]` returns a new view each time |

Other behaviour changes since 0.5.2:

- **Params are copied, not borrowed — check training loops.** `SeqPair`, `add_many()`
  and `set_params()` keep a copy, so an in-place update of an `AlignParams` (`+=`, `*=`,
  the gap / matrix setters) no longer reaches existing pairs. In 0.5 the pairs held a
  reference, so a loop like `params += lr * g; batch.score_and_grad()` picked up the new
  values (while results cached before the update stayed marked valid). In 0.6 **that loop
  silently keeps scoring the original params** — nothing raises. Call
  `batch.set_params(params)` after each update.
- **Input checks that were missing.** A negative `band` raises everywhere (it silently
  ran the full DP when no guide was given); giving only one of `aligned_a` /
  `aligned_b` raises (it silently used the diagonal guide); `logistic.fit_alpha` raises
  on non-finite scores (NaN used to "converge" to a plausible intercept).
- **`n_threads=0` respects the CPU affinity mask** (`taskset`, cpusets): under
  `taskset -c 0` it is now 1, not the host's core count.
- **Score only.** `grad_mode="none"` batches (`score_and_grad()`, `align()`) and the
  `nw_score*` / `sw_score*` functions keep no table and run no traceback, and return the
  exact optimum. For float32 pairs longer than `hb_cutoff` under `traceback="auto"`, 0.5's
  score (replayed along the `hirschberg_pmax` path) could sit a hair below it, so such a
  score may now come out slightly higher. A none-mode `score_and_grad()` defers the guides
  (same guides, computed on first use).
- `logistic.log_likelihood()` and `Step.loglik_at_alpha0` were removed (see
  [`nwgrad.logistic`](#nwgradlogistic)).
- **`logistic` labels are no longer checked**, and soft labels in [0, 1] are accepted.
  `step()` and `fit_alpha()` no longer raise `ValueError` for labels other than 0/1 or
  of a single class; the caller must check them (see
  [`nwgrad.logistic`](#nwgradlogistic)).

Results are unchanged: scores, alignments and hard gradients are bit-identical to
0.5.2's, and soft per-pair results too. Two sums differ in the last bits for soft
pairs only, because they are now reproducible: `compute_grad()` (0.5 merged per-thread
sums in completion order) and the scores of a deprecated `align_full()` under
`fill="interpair"` (0.5 ran each pair's own forward-backward; the shared soft pass
agrees within the soft path's usual tolerance).
