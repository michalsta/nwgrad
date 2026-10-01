# Precision, traceback modes and SIMD kernels

Three independent choices, all with safe defaults: the arithmetic precision of the
DP, what the DP keeps in memory to recover the alignment path, and which vector
instruction set runs it. See the [API reference](api.md) for where each is set.

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

## Performance notes

- With `traceback="pointers"`, a hard-gradient alignment keeps 3 bytes per DP cell; `"scores"` keeps 12. The default for affine global alignment is Hirschberg, which keeps O(m+n) — so the per-thread buffer described below applies to the pointer modes and to the Hirschberg base case (≤ `hb_cutoff` rows).
- `SeqPairBatch.score_and_grad()` allocates one `DpBuffer` per thread (sized to the largest sequence pair in the batch) and reuses it across all assigned pairs. Pair-owned DP tables are never allocated, keeping peak memory at `n_threads × max(m×n)` rather than `N × max(m×n)`.
- Work is distributed via a shared `std::atomic` counter — no per-task mutex, no work-stealing queue.
- Gradient accumulation is per-thread; a single mutex is taken once at join to merge partial gradients.
