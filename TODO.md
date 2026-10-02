# TODO

## Batch-wide walks over SeqPair objects are memory-bound and do not scale

Measured 2026-10-02 on nighthaven, 2.5M Manakov pairs: `set_params()` 56 ms at 12
threads vs 61 ms at 1; `scores()` 25 vs 33 ms; `weighted_grad()` ~50 ms. perf puts
the time in the worker loop itself: each pair is a separate ~1 KB heap object, so
touching a few fields per pair is close to one DRAM miss per pair, and more threads
do not help. With `fill="interpair"` these walks are ~14% of a DiscrimAlign
iteration (0.13 of 0.94 s). The fix is structural: keep the hot per-pair state
(score, validity flags, params pointer, the gradient's counts) in contiguous
batch-owned arrays, so these become streaming passes.

## `banded_grad()` in global mode is not deterministic across threads

Found 2026-10-02, present on `main` (`eb08bb6`). After `score_and_grad()` and
`set_params()`, `banded_grad(2)` on 120 global affine DNA pairs (A 10-30, B 45)
gives different scores/gradients from run to run at `n_threads=3`: 20 of 30 runs
differ from the first on `main`, 30 of 30 on `perf-dp`; at `n_threads=1`, 0 of
30. The likely cause is a read of DP cells outside the band that were never
initialised for this pair, so the value depends on what the thread's buffer held
before (the GuideBanded global border/band init in `viterbi_affine_simd`). Local
mode did not show it in the same test. Reproducer: `banded_nondet.py` pattern in
`tests/Python/test_fill.py::test_guides_match_striped` with more than one thread.

## Default thread count (`n_threads=0`): physical cores are wrong for short pairs

`default_thread_count()` (`parallel.hpp`) resolves `n_threads=0` to the number
of physical cores. Its comment justifies that with the human proteome (23.9
Gcells): the DP is stall-bound once the tables stop fitting in cache, and two
SMT siblings halve each other's L1/L2. On that workload, logical cores cost up to
1.44× (nighthaven).

On short pairs the opposite holds. The tables stay in L1/L2, per-pair overhead
(setup, profile, traceback, gradient) dominates, and SMT hides its latency.
Measured 2026-10-01 through DiscrimAlign's iteration loop (`e2f847a`, double,
pointer traceback, local/affine/general; seconds per iteration, median of 5):

| host (physical / logical) | workload | auto (physical) | all logical | logical gain |
|---|---|---|---|---|
| nighthaven, i5-12500 (6 / 12) | 50k pairs, 22×50 nt | 0.080 | 0.060 | 25% |
| solace, i7-11850H (8 / 16) | 50k pairs, 22×50 nt | 0.069 | 0.052 | 25% |
| nighthaven | 2000 pairs, 300×300 aa | 0.114 | 0.092 | 19% |
| solace | 2000 pairs, 300×300 aa | 0.091 | 0.076 | 16% |
| wloczykij, Opteron 6380 (?, 64) | 50k pairs, 22×50 nt | 0.129 | 0.100 (64) | 22% |

(wloczykij: presumably its module pairs share a `core_id`, so auto picked 32.
Not verified.)

### Proposed heuristic (not implemented)

Choose by whether the batch's DP working set is cache-resident, computed per
call from the batch's actual contents:

1. Per-pair working set for the traceback mode in use: about 3 B/cell × m·n
   (pointers), 12/24 B/cell (scores), O(n) rows (Hirschberg); soft mode holds
   both directions' tables.
2. Weight by work (cells): the fraction of the batch's total cells in pairs
   whose working set fits in **half the per-core L2** (two siblings share it).
3. If most of the work (≥ 50%, to be tuned) is resident, use the logical count;
   otherwise use physical cores, as now.
4. L2 per core from `/sys/devices/system/cpu/cpu0/cache/index2/size` (Linux) or
   `hw.perflevel0.l2cachesize` / `hw.l2cachesize` (macOS), clamped to
   [256 KB, 2 MB], defaulting to 1 MB. VMs report fictional caches (skynet: a
   960 MiB "L3").
5. Resolve per call, not once per process as now (`static const` in
   `default_thread_count()`).

It agrees with every measurement above (22×50: ~3 KB per pair; 300×300: ~270 KB
per pair, i.e. 540 KB for two siblings, under 1.25 MB of L2) and with the
proteome (work dominated by multi-MB tables). Where physical == logical (VMs,
no-SMT hosts) it cannot do harm.

### Open questions before implementing

- **Which traceback mode the proteome measurement used.** If it was Hirschberg
  (O(n) memory, nearly always resident), this model predicts "logical" for it and
  would be wrong. The SMT loss would then come from something else, e.g. DRAM
  bandwidth.
- **The DRAM-bandwidth ceiling** (nighthaven's proteome optimum was 4 of 6 cores)
  is not modelled, as it is not now.
- **Tune the 50% threshold** with a sweep (short vs long pairs) on nighthaven,
  solace and wloczykij.

Until then, callers with short pairs should pass an explicit `n_threads`.
DiscrimAlign does: it passes the logical core count.

## Per-pair gradients stored contiguously (weighted_grad, grads)

`weighted_grad()` and `grads()` read every pair's cached gradient from its own
heap objects: `pairs[i]` points to a `SeqPair` (about 2.8 KB per pair with its
DP machinery), whose `grad_` holds a `SubstMatrix` with its values in yet
another heap block. The cost is the walk, not the arithmetic. Measured
2026-10-02 on skynet, 2.5M random 22×50 nt pairs, DNA, double:

| operation | time |
|---|---|
| serial `weighted_grad()` (0.5.0) | 779 ms (312 ns/pair) |
| `grads()`, the same walk copying instead of multiplying | 591 ms |
| numpy `w @ X` on the same numbers in one contiguous (N, 20) array, 1 thread | 160 ms |
| `weighted_grad()` in fixed parallel blocks (current), 1 / 4 / 16 / 60 threads | 708 / 304 / 118 / 63 ms |

The block-parallel sum (fixed blocks of `WEIGHTED_GRAD_BLOCK` pairs, block
sums added in order) hides the latency across cores and stays bit-identical
for any thread count; it flattens above ~16 threads.

### Proposal (not implemented)

Let `score_and_grad()` write each pair's gradient into one batch-owned
(N, n²+4) array instead of, or besides, `SeqPair::grad_`. Then
`weighted_grad()` is a streaming pass (memory bandwidth, not latency) and
`grads()` is a copy or a view. Worth it for few-core machines and as part of
slimming the per-pair objects (2.8 KB/pair is what made DiscrimAlign's
2.5M-pair Manakov fit peak at 10 GB); not worth it on its own while the
block-parallel sum is already ~2% of a DiscrimAlign iteration. Keep the
block order (and the no-FMA rule) so results do not change.
