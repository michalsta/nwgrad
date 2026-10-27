# Precision, traceback modes and SIMD kernels

Three independent choices, all with safe defaults: the arithmetic precision of the
DP, what the DP keeps in memory to recover the alignment path, and which vector
instruction set runs it. See the [API reference](api.md) for where each is set.

## Precision

The plain names — `SeqPair`, `SeqPairBatch` and the twelve
convenience functions — run the Viterbi DP in **float32**. The `*Double` classes
(`SeqPairDouble`, `SeqPairBatchDouble`) and `_double`
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
| `"pointers"` | 1 B/cell | Records each cell's predecessors in one byte (affine: the M, X and Y states' choices packed together). |
| `"scores"` | 12 B/cell (affine, float32) | Keeps the score tables and re-derives the path. Bit-identical to `"pointers"`; for table introspection. |
| `"hirschberg"` | O(m+n) | Linear-space divide-and-conquer, affine or linear gaps, full DP (global or local). Pairs no longer than `hb_cutoff` (default 512) never split and are bit-exact with `"pointers"`; longer pairs get an optimal path, but where alignments tie it may be a *different* optimal path (a different valid subgradient). |
| `"hirschberg_pmax"` | O(m+n) | As `"hirschberg"`, with the affine gap carry computed as a closed-form prefix max — in the recursion and, for local alignment, in the endpoint scans: 1.2–1.5× faster than `"hirschberg"` on related sequences (1.5–3.9× before the shortened gap carry of 0.8.x, which sped up the exact sweep more). The **only mode that can return a slightly suboptimal path** — measured worst case 4.9e-4 at float32, below float32's own error. Bit-identical across ISA levels. For linear gaps it is `"hirschberg"` (there is no gap-open chain to replace). |
| `"grid"` | O(k·(m+n)) | **Bit-identical to `"pointers"`** (score, path, guide, gradient, ties included) at about the cost of a score-only pass: one forward pass saves a grid of rows and columns, then the walk back re-fills only the blocks the path crosses. Affine global, full DP, float32/float64, non-negative gap opens, on a SIMD kernel, pairs longer than `hb_cutoff`; everything else runs as `"pointers"` (the same result by definition). Measured (AVX2, 1 thread, protein, × score-only): 1.15–1.44× at L=1000, 0.88–1.06× at L=3000–10000, against 2.1–3.1× for the defaults. Memory ~32·(m+n)·3 values (≈15 MB for 10 000 aa at float64). |
| `"auto"` (default) | — | Affine full DP, global, float32 and float64: `"grid"` (Pointers' result; since 0.8.x). Where grid does not apply — tracks, `kernel="scalar_fallback"`, negative gap opens — the pair takes the previous default: `"hirschberg_pmax"` at float32; at float64, pairs of up to 2²⁶ cells (≈ 8200 × 8200) run as `"pointers"` and larger ones as `"hirschberg"`. Affine full DP, local: `"hirschberg_pmax"` at float32 and the integers, with pairs of up to 2²⁵ cells (≈ 5800 × 5800) run as `"pointers"`; `"pointers"` at float64. Integers, global: `"hirschberg_pmax"` (int64 pairs of up to 2²⁵ cells as `"pointers"`). Linear: `"pointers"` (see below). Banded: score tables. |

Asking explicitly for a Hirschberg mode on a banded problem raises; `"auto"` never
selects one there. With one byte per cell of direction codes, `"pointers"` is faster
than the Hirschberg recursions for most pairs that fit a per-thread budget (measured
on AVX2, AVX-512, NEON and a 64-core NUMA host, 2026-10-25): 1.2–2× against exact
`"hirschberg"` for float64 global alignment, 1.5–3× against `"hirschberg_pmax"` for
local alignment of related sequences (on unrelated sequences, where the local
alignment is short, the recursion stays 0–37 % faster). So `"auto"` runs those pairs as
pointers up to 2²⁵ cells — 2²⁶ for float64 global, where pointers kept winning to
L8000 once the codes went dense (see below) — and switches to the
linear-memory recursion above that. Global float32 and int16/int32 keep
`"hirschberg_pmax"`, which wins there on related sequences. Local float64 keeps
`"pointers"` at every size — select `traceback="hirschberg"` explicitly for very long
local pairs (e.g. >10k residues), where the pointer table would not fit.

Past a size, `"pointers"` records **dense** direction codes — 2 bits per code, four
cells per byte for linear gaps and four cells in three bytes for affine — where the
smaller table outweighs the extra packing work. That size depends on the SIMD level,
gap model, mode and number type. Dense codes are used on AVX2 and AVX-512, and on
NEON for linear float32 / int32 and affine int32, float32 and local float64. SSE2 /
SSE4 and the scalar fill always use one byte per cell. The path is bit-identical
either way. `NWGRAD_DENSE_CODES=0` / `=1` forces the format (for A/B timing).

Recursive Hirschberg (pairs with `len(seq_a) > hb_cutoff`) requires non-negative
gap penalties; linear gaps ignore the gap-open fields. An explicit
`"hirschberg"` or `"hirschberg_pmax"` request raises `ValueError` otherwise.
`"auto"` falls back to `"pointers"`, preserving the optimum but increasing DP
memory from O(m+n) to O(m×n). This can matter in training loops whose updates
move a gap cost below zero. Pairs at or below the cutoff still use pointers.

For linear gaps `"auto"` resolves to `"pointers"`, but a pair whose score table would
be at most 2 MiB keeps the table instead of direction bytes: the table fill is ~1.35×
faster per thread, while past L2 the table's memory traffic costs up to 3.5× at 12
threads. The path is the same either way. An explicit `"pointers"` always records
bytes.

## Kernel selection

`kernel=` picks the Viterbi backend: `"auto"` (default — the strongest SIMD level
the CPU runs), `"scalar_fallback"`, or a named level: `"sse2"`, `"sse4"` (x86-64-v2:
SSE4.2 + POPCNT), `"avx2"`, `"avx512"` (x86) or `"neon"` (ARM). Every level is **bit-exact** with `"scalar_fallback"`
(same tables, paths and gradients), so this is purely a speed knob. An unknown
name, or a level this CPU cannot run, raises. It affects the Viterbi (hard / none)
path and the inter-pair fills; a single linear-gap pair has no SIMD kernel (only the
batch `fill="interpair"` vectorizes linear gaps, across pairs), so any choice is a
no-op for it.

Module-level controls: `nwgrad.simd_isa()` reports the active level,
`nwgrad.available_isa_levels()` lists what this CPU runs, and
`nwgrad.set_isa_level(name)` (for testing; not thread-safe while work is in flight)
or the `NWGRAD_ISA` environment variable force the level that `"auto"` dispatches to. `nwgrad.compiled_with()` names the compiler the extension was built
with.

## Fill gates

Which fill a pair runs (shared inter-pair groups or its own fill, striped or scalar
linear kernel) is decided by thresholds measured on one host per ISA level. Results
never depend on them, since every fill is bit-identical; only speed and memory do.
For measuring them on other hardware, `nwgrad._tuning` overrides them at runtime,
process-wide. It is **unstable**, not part of the API:

```python
from nwgrad import _tuning
_tuning.GATES                     # the gate names
_tuning.defaults()                # compiled defaults (None = no cap)
with _tuning.override(inter_score_gather_cells=None):    # gate off
    batch.score_and_grad()
_tuning.set_gate("inter_group_bytes", 2 << 20); _tuning.reset_gates()
batch._inter_plan()               # [pairs in shared groups, pairs on their own fill, groups]
```

An `int` is the threshold, `None` turns the gate off, and `"default"` removes the
override. A batch regroups on its next call after any change.

## Performance notes

- With `traceback="pointers"`, a hard-gradient alignment keeps 1 byte per DP cell; `"scores"` keeps 12. Under `"auto"`, pairs up to 2²⁵ cells (2²⁶ for float64 global) run as pointers (≤ 64 MiB per thread) and larger ones take a Hirschberg mode, which keeps O(m+n) — so the per-thread buffer described below applies to the pointer modes and to the Hirschberg base case (≤ `hb_cutoff` rows).
- `SeqPairBatch.score_and_grad()` uses one `DpBuffer` per thread (sized to the largest sequence pair in the batch) and reuses it across all assigned pairs — and, since 0.8.x, across calls: the batch keeps up to `n_threads` of them between calls, so a training loop does not re-allocate (and first-touch) its DP tables every step. `drop_paths()` frees them. Pair-owned DP tables are never allocated, keeping peak memory at `n_threads × max(m×n)` rather than `N × max(m×n)`.
- Transparent huge pages for the DP tables are opt-in: `NWGRAD_HUGEPAGE=1` (they cost Pointers 1.3–1.9× at 12 threads on a THP-`madvise` host and gained nothing at 1 thread).
- Work is distributed via a shared `std::atomic` counter — no per-task mutex, no work-stealing queue.
- Stored-batch gradients are summed in fixed blocks of 4096 pairs, with each block
  accumulated in pair order and the block sums merged in order. Results are
  reproducible across thread counts. Streaming `align()` uses per-thread sums.
