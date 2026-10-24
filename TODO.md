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

## Remove the 0.6.0- and 0.7.0-deprecated APIs — DONE

0.7 removed the 0.6.0-deprecated batch API (untyped batches, per-add problem types, batch
`alloc_dp` / `align_full` / `realign_banded` / `drop_dp`, `SeqPair.alloc_dp`).

0.8 (2026-10-20) removed what 0.7.0 deprecated, after DiscrimAlign's port (BioGeMT
PR #18) was merged:
- `SeqPairBatch.add_many()` (Python and C++; `BatchEngine::add_many()` became the
  engine's pair constructor) and the 0.7 pair lifecycle `align_full()` /
  `realign_banded()` / `compute_grad()` / `drop_dp()` / `dp_valid`, with
  `pair_api_compat.hpp` and the batch `compute_grad()` bridge (it now checks every
  gradient is published, as `weighted_grad()` does).
- The engine: pairs set once, so no append bookkeeping (offsets from N0, the rollback
  after a failed encode, incremental store growth, the alphabet / K checks against
  existing pairs, `make_tracked_()`, `generation_`); no segments (`Segment`, `segs_`,
  the per-pair `seg_`, `set_params_segment()`): one params (`params_`, an
  `OwnedParams` holding the tracked prep too) and one kernel; no params store
  (`own_params_` dedup, `params_idx_`, `params_same_`, `drop_unused_params_`, `prep_`);
  the inter-pair plan has one backend for every group and no "same params" lane checks.
- Tests: the append / segment / compatibility cases deleted, each one's non-append
  coverage kept or ported (`test_pair_lifecycle.py`, `test_batch_constructor.py`
  against streaming `align()`); `tools/fuzz_fill.py` builds several params as K stacked
  sets with constant tracks, `tools/fuzz_ab.py` one params per case.

Not measured: no speed claim (no DP kernel changed).

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

## Integer scalar types: what is left (branch `integer`, 2026-10-22)

- **Linear kernels: what is left** (the leveled striped linear fill, linear Hirschberg, the
  NEON float inter-pair cap and the AVX-512 gate all landed and were measured on AVX2,
  AVX-512 and NEON, 2026-10-23: AGENTS "Versus parasail"). Global linear protein pairs now join the
  inter-pair fill (done 2026-10-23; caps confirmed on AVX2, AVX-512 and NEON — on NEON
  they forgo int L80-100 and float32 L130 wins of 3-23 %, accepted). The per-level thresholds come from one host per ISA — re-derive
  them on another AVX-512 CPU (Zen 4 splits 512-bit ops) before trusting them there.
- **The integer score kernel's working set: DONE for the score kernel** (2026-10-24, AGENTS
  "Versus parasail": the carry fused into the row pass, 0.45–0.9 of the separate carry's
  time on every host and type measured). Left: the same separate carry (opens / flags / Q
  rows) in the Pointers, Scores and hb_base kernels, which write tables anyway — less to
  gain, but check their 12-thread L1000 drop on nighthaven; the Pointers kernel needs the
  open codes, so the fusion does not transfer one to one.
- **4 KB set aliasing of the rolling rows on 4-way L1Ds** (found 2026-10-24, skynet;
  investigation on wloczykij in progress): rows of ~4000–4080 B are allocated exactly 4096 B
  apart and thrash a 16 KB 4-way L1D (int32 L1000 4–9× slower than L960). Remedy candidates:
  carve the rolling rows from one block with a row stride that is not a multiple of 4096
  (pad by a cache line where it is), so the streams fall in distinct sets.
- **int16 saturating arithmetic, with a number**: at L1000 BLOSUM62 nwgrad refuses int16
  and runs int32 (261 us a pair, score-only, AVX2) where parasail's saturating scan16 takes
  145 us — the one fixture in the comparison parasail wins.
- The streaming `align()` validates integer params per pair (the aligner's `nw_int_prepare`
  on the caller's params, O(K * nalpha^2) a pair); prepare once per call as the batch does.
- `score()` / `scores()` stay double: exact below 2^53, so an int64 score above that
  (parameters ~2^40) loses bits in the report.
- int16 with saturating arithmetic would remove the (m + n + 66) * step bound (the sentinel
  then self-preserving under `psubsw`); std::simd has no saturating ops, so intrinsics per
  level. Not started; measure int16 first.
- AVX-512: verified and measured 2026-10-22 (solace under WSL, gcc 15.2 / clang 21.1; AGENTS
  "Integer scalar types"). gcc's AVX-512 integer Pointers kernels: after the always_inline
  round and the row-argmax broadcast fix (both 2026-10-23) gcc is at 1.10-1.23x clang on
  them (int64 Local 49.4 vs 42.8 us a pair, Global 38.8 vs 31.6; int16 / int32 1.1-1.16x),
  the striped loops' scheduling residue (AGENTS "Toolchain"); the score kernels tie.
  Nothing specific left to read there; perf setup on solace: RelWithDebInfo venvs under
  ~/cc/bench-int-inline, `perf record -e cpu-clock` works under WSL.
- spot (NEON): the final state is verified (ctest 17/17, Python suite, `bench_int_cc.sh`
  at neon, 2026-10-22; built with `CMAKE_BUILD_PARALLEL_LEVEL=3` after the first attempt
  took the 16 GB laptop down). Note for long jobs there: take a `systemd-inhibit` (see
  the fleet notes) — the laptop slept after this run finished.
- gcc vs clang on the integer striped kernels, final state: clang still 4-20 % faster
  (nighthaven protein int16/int32: gcc at 0.80-0.96 of clang's time at avx2, 0.78-0.94 at
  sse4; Piledriver int16 score 0.70). What is left is clang's x2 unrolling and interleaved
  scheduling of the segment loops, the same residue the float kernels show at W=2
  (AGENTS "Toolchain"); `#pragma GCC unroll` did not close it there either. Not pursued.
- int64 at AVX2 costs 1.5-2.6x float32 (no packed 64-bit max: `vpcmpgtq` + `vpblendvb`,
  the same 4 lanes as double) and 4-5x at sse2 (no 64-bit compare at all). It is headroom
  for huge parameters, not a speed type; nothing cheap left to do for it.

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
  `inter_score_*`; see AGENTS.md "Score only").  (a) DONE 2026-10-08: the per-pair striped
  kernel's D/X form (`score_affine_striped_pos`, non-negative opens): nighthaven AVX2 vs the
  three-state fused form 1.0-1.5x at 1 thread, 1.0-1.6x at 12; float32 Global ~neutral
  (0.95-1.16, lazy-F also updates D).  Left: (b) float32 on
  AVX-512 (W=16): lazy-F rounds make long Global pairs 0.89x `hirschberg_pmax` — consider a
  W=8 (256-bit) float kernel there, or an opt-in prefix-max score; (c) rerun
  `NWGRAD_SCORE_FUZZ=20000 nwgrad_tests "[score]"` on solace (AVX-512) with negative
  penalties. NEON is verified on spot, 2026-10-09, candidate `e0e434b`: 800,392
  assertions passed; see [release validation](docs/release-validation.md). (d) inter-pair
  eligibility for score only still uses the table-size cap (`inter_pair_fits`), which
  does not apply to one row — re-derive the crossover against the per-pair kernel.
- **Hirschberg boundary states — FIXED 2026-10-09.**  Not float-specific: when the join
  lets a vertical gap run straddle the cut, the halves must CONTINUE it (lower half leaves
  its origin vertically, upper half enters its corner vertically), but the sweeps seeded
  the X state at the origin / corner, which also let it feed a diagonal or horizontal move.
  A half was promised more than its solver (base case: strictly ends in X) could deliver,
  compounding with depth: random pairs at small hb_cutoff lost up to 4.8 (float32) / 3.5
  (double); default cutoff 512 measured fine before and after.  Fix: origin / corner
  unreachable, the first swept row's column 0 seeded 0 - ge_b (scalar hb_fwd / hb_rev /
  hb_base, the vector exact and prefix-max sweeps and hb_base_striped, the test model).
  After: 0 of 36k scalar and 2.4k vector random alignments beyond rounding.  Regression
  tests: test_hirschberg.py::test_inherited_gap_run_keeps_the_optimum,
  ::test_boundary_states_double_case.  (A first attempt that changed the seed VALUE only
  moved the failures around — any constant seed is a uniform shift inside a block.)
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
- **Verify the unified affine walk (`walk_affine`, S3) on AVX-512.** NEON is done:
  spot, 2026-10-09, candidate `e0e434b`, 1.6M scalar/NEON comparison cases against
  the six old walkers, zero differences. Solace is still unreachable. The harness
  compiles the old `aligner.hpp` (from `5d97c38`) inside
  a namespace next to the new one — copy `aligner_simd.hpp` alongside it under another
  name with different content, or GCC's `#pragma once` (content + mtime) skips it.
- **Release notes: B3 changes behaviour — DOCUMENTED in `docs/api.md`.** Batches and
  pairs own a copy of their params:
  an in-place change (`*=`, the gap/matrix setters) no longer reaches them — call
  `set_params()` — and the Python side no longer keeps the params object alive.
  DiscrimAlign is unaffected (it builds fresh params and calls `set_params()`).

## Tracks (parameter sets per row of A): what is left

Tracks landed on branch `tracks` (2026-10-11; see AGENTS.md, "Tracks").  Deferred:

- **Glocal mode** (global in A, local in B: free leading and trailing gaps along B) —
  PosimiR's second DP mode.  A third `AlignMode`, or row-0 / row-m horizontal costs of
  0 via the gap track (that needs a "free" set whose counts reach no gradient).  Not
  started; the gap track already prices rows 0 and m on their own, so the hooks exist.
- **Fleet runs not yet done: spot (neon) and solace (avx512).**  Correctness
  (`ctest`, `NWGRAD_FORCE_KERNEL=simd pytest tests/Python`) and `tools/bench_tracks.py`.
  Everything on the branch was verified on sse2 (skynet, wloczykij) and avx2 (nighthaven).
- **Python track conversion** (`to_tracks` in py_exports.cpp) casts each pair's list
  separately: ~1-2 µs per pair, seconds for millions.  Accepting one flat int array plus
  offsets would remove it.
- **Track storage is per pair** (2 B per residue, both tracks).  PosimiR's tracks are a
  function of the miRNA's length (0, 1, 2, ...): interning identical tracks would make
  them free.  Not needed yet (≈ 90 B per Manakov pair).
- **Tracked soft batches store a per-row record per pair**: `len(A) * n + 4 * (len(A)+1)`
  doubles (≈ 1.4 KB per Manakov pair, ≈ 3.6 GB for 2.5 M).  Inherent when every row has
  its own set; `align()` (streaming) stores nothing.  Hard gradients are paths (≈ 30 B).
- **Tracked affine global is 1.16-1.27x untracked, soft 1.18-1.34x** (100k Manakov
  pairs, K = 25; profile, what was done and tried: AGENTS.md, "Tracks", Performance).
- **Stored counts (`track_storage="counts"`) cost memory**: 120-370 B a pair after a fill
  (100k Manakov pairs: 12-37 MB) against ~35 B for a path; for 2.5 M pairs, 0.3-0.9 GB.
  The buffers grow by doubling (the 12-thread figures are ~1.5x the 1-thread ones):
  sizing each writer from the last generation's use, or one shrink after a fill, would
  cut that.  Keys and counts fit 16 bits each for every PosimiR shape (one word an
  entry already).  Paths stay the default; counts win 5-15% on linear, are +-4% on affine.
- **`linear-guide-walk` (from `main`)**: the untracked linear batch walked each
  traceback twice (gradient, then guide); one walk: -11 to -37% on linear sg,
  bit-identical.  Also there: inter-pair rows built in registers (only the vectors
  `inter_sub` reads).  To merge into `main` separately; `tracks` then rebases.
- **Untracked guide-banded iterations at 1 thread are 3-5% slower than on `main`** (+1%
  instructions, kernels identical; spread over lane setup and `set_problem`, layout
  sensitive).  12 threads: 1.01.
- **Tracked Hirschberg builds each row's profile as it sweeps** (no shared profile): one
  gather per cell.  A compact profile per block would help when rows share sets.
