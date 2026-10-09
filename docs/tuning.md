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
| `"pointers"` | 3 B/cell (affine), 1 B/cell (linear) | Records a predecessor byte per cell (per state). |
| `"scores"` | 12 B/cell (affine, float32) | Keeps the score tables and re-derives the path. Bit-identical to `"pointers"`; for table introspection. |
| `"hirschberg"` | O(m+n) | Linear-space divide-and-conquer, affine or linear gaps, full DP (global or local). Pairs no longer than `hb_cutoff` (default 512) never split and are bit-exact with `"pointers"`; longer pairs get an optimal path, but where alignments tie it may be a *different* optimal path (a different valid subgradient). |
| `"hirschberg_pmax"` | O(m+n) | As `"hirschberg"`, with the affine gap carry computed as a closed-form prefix max — in the recursion and, for local alignment, in the endpoint scans: 1.5–3.9× faster than `"hirschberg"` on related sequences. The **only mode that can return a slightly suboptimal path** — measured worst case 4.9e-4 at float32, below float32's own error. Bit-identical across ISA levels. For linear gaps it is `"hirschberg"` (there is no gap-open chain to replace). |
| `"auto"` (default) | — | Affine full DP at float32: `"hirschberg_pmax"`, global and local. Affine full DP at float64: `"hirschberg"` for global, `"pointers"` for local. Linear: `"pointers"` (see below). Banded: score tables. |

Asking explicitly for a Hirschberg mode on a banded problem raises; `"auto"` never
selects one there. Local float32 defaults to `"hirschberg_pmax"` because, with the
prefix-max carry in its endpoint scans, it is 1.5–5× faster than `"pointers"` on an
AVX2 host (homologous or unrelated pairs, 1–12 threads); local float64 keeps
`"pointers"`, where exact local Hirschberg wins on memory only — select
`traceback="hirschberg"` explicitly for very long local pairs (e.g. >10k residues),
where the pointer tables would not fit.

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
the CPU runs), `"scalar_fallback"`, or a named level: `"sse2"`, `"avx2"`, `"avx512"`
(x86) or `"neon"` (ARM). Every level is **bit-exact** with `"scalar_fallback"`
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

## Performance notes

- With `traceback="pointers"`, a hard-gradient alignment keeps 3 bytes per DP cell; `"scores"` keeps 12. The default for affine global alignment is Hirschberg, which keeps O(m+n) — so the per-thread buffer described below applies to the pointer modes and to the Hirschberg base case (≤ `hb_cutoff` rows).
- `SeqPairBatch.score_and_grad()` allocates one `DpBuffer` per thread (sized to the largest sequence pair in the batch) and reuses it across all assigned pairs. Pair-owned DP tables are never allocated, keeping peak memory at `n_threads × max(m×n)` rather than `N × max(m×n)`.
- Work is distributed via a shared `std::atomic` counter — no per-task mutex, no work-stealing queue.
- Stored-batch gradients are summed in fixed blocks of 4096 pairs, with each block
  accumulated in pair order and the block sums merged in order. Results are
  reproducible across thread counts. Streaming `align()` uses per-thread sums.
