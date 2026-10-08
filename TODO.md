# TODO

## Batch-wide walks over SeqPair objects were memory-bound — DONE in the 0.6 restructure

Measured 2026-10-02 on nighthaven, 2.5M Manakov pairs: `set_params()` 56 ms at 12
threads vs 61 ms at 1; `scores()` 25 vs 33 ms; `weighted_grad()` ~50 ms — one ~1 KB heap
object per pair, close to one DRAM miss per pair. The 0.6 restructure (branch
`restructure`, 2026-10-07) did the structural fix proposed here: per-pair state in flat
batch-owned arrays (`batch_engine.hpp`). Measured with `tools/bench_ab.py` (100k pairs,
nighthaven, 12 threads): a whole hard training step (set_params + score_and_grad +
scores + weighted_grad) 0.76x the old time, a banded step 0.62x, resident memory
385 -> 34 MB. See AGENTS.md "batch_engine.hpp".

## Drop the deprecated 0.5 batch API once DiscrimAlign is ported

Kept at the maintainer's request (2026-10-07) so DiscrimAlign runs unchanged: untyped
`SeqPairBatch(...)` + `add_many(gap_model=, mode=, grad_mode=)`, and the batch's
`alloc_dp` / `align_full` / `realign_banded` / `drop_dp` (plus `SeqPair.alloc_dp`).
DiscrimAlign's call sites: `src/nwgrad_engine.py` 121-126 and 156-169, `src/adaptive.py`
143-159. Port them to `SeqPairBatch(n, traceback, gap_model=, mode=, grad_mode=)`,
`add_many(A, B, params)` and `score_and_grad(keep_paths=True)` + `batch[i].coordinates()`,
check its acceptance fits stay bit-identical, then delete the wrappers, the `kHeld` /
`hold_grads` machinery that exists only for them, and their tests.

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

## Package-scan follow-ups (scan of 2026-10-07; correctness items fixed 2026-10-08)

The scan's correctness and maintenance findings (B1–B8, S1–S4) are closed: fixed on
branch `scan-fixes` or already gone with the 0.6 restructure. What remains is
performance work, none of it measured yet; the cost notes are estimates.

- **O1 — score-only: DONE 2026-10-08** (`compute_score`, `score_kernel_impl.inl`, inter-pair
  `inter_score_*`; see AGENTS.md "Score only").  Left: (a) the per-pair striped kernel's
  two-row form (D = max(M,X,Y) and X instead of M/X/Y, as `inter_score_affine_pos` — exact
  for non-negative opens; fewer loads and maxes per cell; est. 15-25 %); (b) float32 on
  AVX-512 (W=16): lazy-F rounds make long Global pairs 0.89x `hirschberg_pmax` — consider a
  W=8 (256-bit) float kernel there, or an opt-in prefix-max score; (c) the score kernels on
  NEON/AVX-512 were verified before negative penalties entered the tests — rerun
  `NWGRAD_SCORE_FUZZ=20000 nwgrad_tests "[score]"` on spot/solace; (d) inter-pair
  eligibility for score only still uses the table-size cap (`inter_pair_fits`), which
  does not apply to one row — re-derive the crossover against the per-pair kernel.
- **float32 Local `hirschberg_pmax` short of the optimum by 0.5** on a random DNA pair
  (penalties 3/0.3/2/0.2, integer matrix; 153.1 vs Pointers 153.6) — beyond the
  measured <= 4.9e-4 (those were Global fixtures).  The Local endpoint scans took the
  prefix-max carry on 2026-10-05; characterize (tools/char_hb_pmax.py on Local) before
  trusting the float32 Local `auto` default.
- **Hirschberg with negative gap penalties** is guarded (auto -> Pointers, explicit raises);
  a real fix would make the join and the Local endpoint scans handle gaps that gain score.
- **O3 — release the GIL in long batch calls.** Same item as AGENTS.md's Open TODO.
  ~0.1 µs per call: release around batch calls only, not the µs-scale single-pair
  functions. Blocked on a stated contract against two Python threads mutating one batch.
- **O4 — logistic fitting starts threads per evaluation.** `detail::for_blocks`
  (`logistic.hpp`) → `run_workers_guarded` spawns and joins workers on every Newton /
  bracketing / bisection evaluation, and their `thread_local` scratch dies with them.
  One-block inputs already run on the calling thread. Try a call-scoped pool (spawn once
  per `fit_alpha`); the fixed block order must survive so results stay bit-identical.
- **O5 — per-problem parameter setup is redone for every pair.** `Aligner::set_problem`
  converts the substitution block to float32 per problem (`blkT_storage_`), and the
  scaled soft path exponentiates its weights per problem. The batch now owns
  deduplicated params copies per segment (`params_own_`, scan fix B3) — the natural
  place to cache the converted block / weights, invalidated by `set_params()`. Matters
  for many short protein pairs; measure the setup share first.
- **O2 remainder — cgroup CPU quotas.** `default_thread_count()` now respects the
  affinity mask (`allowed_cpus()`), but not a container's `cpu.max` quota. One file read
  per process; mind cgroup v1 vs v2.

Also open from the same work:
- **Verify the unified affine walk (`walk_affine`, S3) on AVX-512 and NEON.** It was
  differentially checked against the six old walkers on scalar/sse2/avx2 only (solace
  and spot were off). The harness compiles the old `aligner.hpp` (from `5d97c38`) inside
  a namespace next to the new one — copy `aligner_simd.hpp` alongside it under another
  name with different content, or GCC's `#pragma once` (content + mtime) skips it.
- **Release notes: B3 changes behaviour.** Batches and pairs own a copy of their params:
  an in-place change (`*=`, the gap/matrix setters) no longer reaches them — call
  `set_params()` — and the Python side no longer keeps the params object alive.
  DiscrimAlign is unaffected (it builds fresh params and calls `set_params()`).
