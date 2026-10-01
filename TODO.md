# TODO

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
