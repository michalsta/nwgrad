# Benchmarks against other aligners

nwgrad measured against every pairwise aligner we could find with a C, C++ or Python
interface, on the same problems, on five machines (AVX2, AVX-512, NEON, SSE4), single- and
multithreaded. Measured 2026-10-27 with nwgrad from GitHub `main` (23ca19b);
the harness is [`tools/fleetbench/`](../tools/fleetbench/), the raw per-arm results are in
[`tools/fleetbench/results/`](../tools/fleetbench/results/).

**In short.** Over 39 000 timed arms on five machines (and 23 000 more re-measuring PR #12),
every nwgrad result matched the float64 reference exactly (float32 within rounding), in
every number type, gap model, mode, traceback, fill, kernel, compiler and API.

- **Score + alignment and gradients — nwgrad's purpose — it is the fastest at almost every
  number type and length**: 1.0–2× parasail, 1–6× SeqAn3 / SeqAn 2, 2.5–8× Biopython and
  scikit-bio at float64; the hard gradient has no competitor, and DeepBLAST's soft gradient
  was stopped at 10× nwgrad's time in all but two cells (5.9× slower there, on
  Piledriver). At float32 / float64 the only other SIMD aligners (SeqAn3, SeqAn 2) run
  scalar and trail 2–7×.
- **Where others win:**
  - *Unit-cost / penalty DNA, global*: WFA2 and edlib are 1.4–50× faster — their cost grows
    with the number of differences, not the DP area. A different algorithmic class; it does
    not apply to the learned real-valued scores nwgrad exists for.
  - *Integer score-only on long sequences*: parasail (16/32-bit `scan`, `striped`) is up to
    ~2× faster at int32 from L1000 up, most with linear gaps.
  - *Short integer pairs* (miRNA, L100): SeqAn3 / SeqAn 2's inter-sequence vectorisation is
    up to 3.4× faster, most on AVX-512.
- **nwgrad's own defaults leave speed on the table** (see *nwgrad's variants*): local affine
  traceback at L10000 runs 2.0–2.6× faster with `traceback="pointers"` than with `auto`'s
  linear-memory choice; on AVX-512, `kernel="avx2"` is up to 2.9× faster than the default
  for short float32 pairs; `traceback="scores"` beats `auto` in 20–70 cells per host.
- **gcc vs clang**: within ±4 % on x86 except int16 on Piledriver (clang 0.89×); on Apple
  M1 clang is ~10 % slower at float. The causes are traced and fixed in PR #12, which
  re-measures at +2–5 % for gcc and +4–11 % for clang (below). The C++ and Python APIs
  perform the same (≤2 %).

## What was compared

Comparisons are strictly **within one cell**: the same fixture, alignment mode (global /
local), gap model (affine / linear), output (score only / score + alignment / hard gradient /
soft gradient) and number type. A tool that has no implementation for a cell shows `–`
there; a cell nwgrad itself cannot run (no float32 soft path, an int16 overflow bound) is
left out entirely. In every cell each tool is represented by **its best implementation** —
SIMD where it has SIMD, its fastest variant among those that returned correct scores.

| tool | version | how it was built | what was run |
|---|---|---|---|
| **nwgrad** | `main` 23ca19b | from source, gcc **and** clang, release-wheel flags (below); Python batch API and the C++ API | everything; default plus variants |
| [parasail](https://github.com/jeffdaily/parasail) | 2.6.1 (Python 1.3.4) | **binary**: the `libparasail.so` its PyPI wheel ships, called from a C driver (no Python per call) | `striped`, `scan`, `diag` × 16/32/64-bit and `sat` (8→16→32 on overflow); score and traceback (CIGAR) |
| [SeqAn3](https://github.com/seqan/seqan3) | 3.4.2 | header-only: `-O3 -march=native`, C++23 | `align_pairwise`, `vectorised` (integer scores) and scalar; native `parallel` |
| [SeqAn 2](https://github.com/seqan/seqan) | 2.5.3 | header-only: `-O3 -march=native`, C++20 | `ExecutionPolicy<Serial/Parallel, Vectorial>` (16/32-bit on x86) and the scalar per-pair API |
| [SSW](https://github.com/mengyao/Complete-Striped-Smith-Waterman-Library) | `a66636b` | source: `-O3 -march=native` (sse2neon on ARM) | striped local alignment, 8→16-bit, score and CIGAR |
| [KSW2](https://github.com/lh3/ksw2) | `289609b` | source: `-O3 -march=native` (sse2neon on ARM) | `ksw_extz2_sse`, global, full width, generic matrix |
| [WFA2-lib](https://github.com/smarco/WFA2-lib) | 2.3.6 | source: **clang** `-O3 -march=native` (see below) | exact WFA (heuristics off); memory modes high/med/low/ultralow (BiWFA); edit metric |
| [edlib](https://github.com/Martinsos/edlib) | 1.3.9.post1 | **binary** (PyPI wheel) | unit-cost global (NW), distance and path |
| [Biopython](https://biopython.org) `PairwiseAligner` | 1.88 | **binary** (PyPI wheel) | Gotoh / NWSW, score and first alignment |
| [scikit-bio](https://scikit.bio) `pair_align` | 0.7.4 | **binary** (PyPI wheel) | global (`free_ends=False`) and local, alignment |
| [pyopal](https://github.com/althonos/pyopal) (Opal) | 0.7.3 | **binary** (PyPI wheel) | one query vs a one-target database; 8→16→32-bit cascade |
| [DeepBLAST](https://github.com/flatironinstitute/deepblast) | 1.0.3 (torch 2.14.1 CPU) | **binary** (PyPI wheels) | `NeedlemanWunschDecoder` forward + backward, float64 |

Looked at and left out: **block-aligner** (adaptive banding — not an exact optimum, so not
the same problem), **AnySeq** (needs the AnyDSL toolchain), GPU aligners (no GPU on the
fleet), and Biopython's **FOGSAA** (returns wrong scores for these gap settings, as its own
warning says: gap scores above the mismatch score).

## Machines

| host | CPU, ISA nwgrad dispatched | OS | compilers | passes | threads (mt) |
|---|---|---|---|---:|---|
| **nighthaven** | i5-12500 (Alder Lake), AVX2 — x86-64-v3, the primary target; `avx2` | Gentoo | gcc 16.2.1 20260926 (+ clang build) | 3 | 6 |
| **oldsolace** | i7-7700HQ (Kaby Lake), AVX2 — x86-64-v3; `avx2` | Gentoo (laptop) | gcc 16.2.1 20260926 (+ clang build) | 5 | 4 |
| **solace** | i7-11850H (Tiger Lake), AVX-512 — under WSL2; `avx512` | Ubuntu 26.04 under WSL2 (laptop) | gcc 15.2.0 (+ clang build) | 5 | 8 |
| **spot** | Apple M1, NEON — Fedora Asahi; `neon` | Fedora Asahi (laptop) | gcc 16.2.1 20260819 (Red Hat 16.2.1-2) (+ clang build) | 5 | 8 |
| **wloczykij** | 4× Opteron 6380 (Piledriver), SSE4 — 64 cores, 8 NUMA nodes; `sse4` | Gentoo | gcc 16.2.1 20260926 (+ clang build) | 3 | 32, 8:numa0 |

## Method

- **Problems.** Every parameter is an integer, so nwgrad's float64 score is exact and is the
  reference for every tool and number type. nwgrad's gap convention: affine, a gap of length
  *k* costs `go + ge·k`; linear, `ge·k`.

  | fixture | substitution scores | affine gaps | linear gaps |
  |---|---|---|---|
  | protein | BLOSUM62 | go 11, ge 1 | 4 per residue |
  | DNA, miRNA | match +2, mismatch −4 | go 4, ge 2 | global: unit edit cost (match 0, mismatch −1, gap 1) — edit distance; local: +1 / −1 / 1 |

  Each tool spells these differently (parasail, SSW, Opal and SeqAn 2 charge `open` for a
  gap's first position; WFA2 takes penalties); every mapping is checked by the scores
  themselves, below.
- **Fixtures** (deterministic, generated once, identical on every host — md5s in the results):
  - *protein*: human proteins (GRCh38 RefSeq, 20 canonical residues), each paired with a
    mutated copy (20 % substitutions, ~3 % deletions, ~3 % insertion runs of 1–5) at
    L ≈ 100, 300, 1000, 3000, 10000 (±10 %). No human protein of 9–11 k residues uses only
    the canonical 20, so L10000 takes ~L windows of longer ones (titin and others).
  - *DNA*: uniform random sequence, mutated the same way at 10 % substitutions, L = 100,
    1000, 10000.
  - *miRNA*: Manakov et al. miRNA × target-site pairs (22 × 50), the short-pair regime.
  - Single-thread sets hold ~3·10⁷ DP cells (at least 2 pairs: 3000 pairs at L100, 2 at
    L10000); multithread sets ~3·10⁸.
- **Correctness.** Every arm writes its per-pair scores; they must equal nwgrad's float64
  score-only result on **every pair** (float32 within 0.5 + 10⁻⁴·|score|), or the arm is
  recorded as *wrong* and its time is not used. The soft gradient's log-partition values
  are checked against nwgrad's default soft arm; DeepBLAST solves a different model (it adds
  the match score in every cell whatever the move) and is compared on cost per cell only.
- **Timing.** One process per arm; setup (reading, encoding, building objects) untimed;
  one warm-up pass, then the best of 3 passes (5 on the laptops). Mcell/s = DP cells /
  best pass. nwgrad's default arm runs first in each cell; any other arm whose pass exceeds
  **10× nwgrad's default time** is stopped and shown as `>10x`. The warm-up pass gets 3 s
  on top of that, for one-time costs that do not scale with the work (imports, JIT, plan
  building, first-touch page faults); without it, cells whose 10× budget was a few tens of
  ms produced false `>10x` — the first run did that, and every timeout was re-run under
  this rule.
- **Builds.** nwgrad and every source-built competitor use release-wheel codegen —
  `-fcf-protection=none -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=0 -fno-stack-protector
  -fno-stack-clash-protection -mtls-dialect=gnu` on x86-64 — because the hosts' toolchains
  inject hardening the manylinux wheels do not have; checked from the compilers' predefined
  macros. nwgrad is built twice (gcc and clang, both on libstdc++ and `std::simd`) and runs
  its own runtime ISA dispatch; header-only competitors get `-march=native`, their only way
  to wide SIMD.
- **Python arms and OpenBLAS.** Importing numpy starts OpenBLAS's thread pool, which spins
  for ~6 core-seconds in the first ~0.7 s of every Python arm (it does not grow with the
  number of passes). The C++ drivers do not import numpy, so nwgrad's C++-API arm is a
  control for its Python-API arm in the same cell: C++/Python, geometric mean, was
  0.996–1.019 single-threaded and 1.001–1.007 multithreaded, binding overhead included —
  the effect on any Python arm is at most ~2 %.
- **Precision of one cell.** Each arm is one process, and processes differ: the PR #12 A/B
  below ran every cell in 8 separate processes per build and found the same build sitting
  at one of two stable levels from process to process — within a process the passes agree
  to ~1 %. On nighthaven 11 % of (cell, build) pairs spread by more than 8 % across
  processes and 6 % by more than 15 % (up to ~35 %), almost all in traceback cells
  (path / hard); score-only cells rarely. Disabling ASLR did not remove it. So a single
  cell's ratio here can be off by that much in either direction; the geometric means over
  many cells, which the conclusions rest on, are not.
- **Saturating integer cascades** (parasail `sat`, SSW, Opal) are filed under the narrowest
  type that completed the cell — int16 if no pair needed more, int32 otherwise — marked
  `sat`.
- **Load.** The Gentoo desktops were watched throughout: an emerge, or another user's
  process above 60 % of a core, paused the runner between arms (on 64-core wloczykij:
  foreign load adding up to more than 4 cores, 3 in its multithread phase — other users'
  jobs, CI runners and the VM the coordinator ran in shared the machine; its multithread
  rows carry up to ~2.5 foreign cores). On the laptops the user was allowed to work. The
  load average is recorded with every arm, and rows run under load are marked.
- **Multithreading.** nwgrad's batch with `n_threads` = its own default (physical cores;
  8 on spot, see below), SeqAn3's `parallel` and SeqAn 2's `Parallel` policy are native.
  Tools without threads of their own get a **harness-parallel** row (‡): the pairs split over
  that many threads (C/C++) or forked processes (Python) — what any user could do, kept
  apart from the native comparison.

## Results at a glance


#### Score only

Geometric mean over the single-thread cells (fixture sets × global/local × affine/linear) that both nwgrad and the tool completed with correct scores; above 1, nwgrad is faster. `+k` = k more cells where the tool was stopped at 10× nwgrad's time. `n` = cells in the mean.

| host | dtype | parasail | seqan3 | seqan2 | ssw | ksw2 | wfa2 | edlib | biopython | pyopal |
|---|---|---|---|---|---|---|---|---|---|---|
| nighthaven | int16 | 1.99x (n=14) | 1.14x (n=14) | 0.75x (n=14) | 2.23x (n=4) | – | – | – | – | 9.40x (n=3) +11 |
| nighthaven | int32 | 1.02x (n=36) | 2.08x (n=36) | 1.34x (n=36) | – | 3.01x (n=18) | 0.54x (n=7) +1 | 0.30x (n=4) | – | 4.19x (n=2) +8 |
| nighthaven | int64 | 9.42x (n=6) +30 | 2.20x (n=36) | 1.33x (n=36) | – | – | – | – | – | – |
| nighthaven | float32 | – | 7.12x (n=7) +29 | 7.34x (n=17) +19 | – | – | – | – | – | – |
| nighthaven | float64 | – | 7.13x (n=32) +4 | 5.54x (n=36) | – | – | – | – | 4.24x (n=35) +1 | – |
| oldsolace | int16 | 2.00x (n=14) | 1.15x (n=14) | 5.76x (n=12) +2 | 2.31x (n=4) | – | – | – | – | +14 |
| oldsolace | int32 | 0.94x (n=36) | 2.02x (n=36) | 3.38x (n=36) | – | 3.49x (n=18) | 0.60x (n=7) +1 | 0.34x (n=4) | – | 4.98x (n=2) +5 |
| oldsolace | int64 | 7.86x (n=27) +9 | 1.86x (n=36) | 1.55x (n=36) | – | – | – | – | – | – |
| oldsolace | float32 | – | 6.63x (n=21) +15 | 6.50x (n=31) +5 | – | – | – | – | – | – |
| oldsolace | float64 | – | 5.22x (n=35) +1 | 4.07x (n=36) | – | – | – | – | 3.56x (n=33) +3 | – |
| solace | int16 | 2.32x (n=14) | 1.01x (n=14) | 5.92x (n=11) +3 | 2.22x (n=4) | – | – | – | – | 8.89x (n=2) +12 |
| solace | int32 | 1.48x (n=36) | 2.62x (n=34) +2 | 4.63x (n=36) | – | 4.36x (n=16) +2 | 0.72x (n=6) +2 | 0.69x (n=4) | – | 5.75x (n=2) +18 |
| solace | int64 | +36 | 3.63x (n=33) +3 | 3.17x (n=36) | – | – | – | – | – | – |
| solace | float32 | – | 5.70x (n=13) +23 | 5.35x (n=17) +19 | – | – | – | – | – | – |
| solace | float64 | – | 6.17x (n=24) +12 | 5.84x (n=36) | – | – | – | – | 4.42x (n=32) +4 | – |
| spot | int16 | 2.23x (n=14) | 7.86x (n=2) +12 | 7.32x (n=4) +10 | 4.07x (n=4) | – | – | – | – | 9.82x (n=1) +13 |
| spot | int32 | 1.74x (n=36) | 6.62x (n=35) +1 | 3.71x (n=36) | – | 3.28x (n=18) | 0.43x (n=7) +1 | 0.20x (n=4) | – | 6.26x (n=1) +7 |
| spot | int64 | 2.09x (n=36) | 2.34x (n=36) | 1.45x (n=36) | – | – | – | – | – | – |
| spot | float32 | – | 6.97x (n=12) +24 | 7.03x (n=26) +10 | – | – | – | – | – | – |
| spot | float64 | – | 5.52x (n=24) +12 | 4.73x (n=36) | – | – | – | – | 3.98x (n=36) | – |
| wloczykij | int16 | 1.29x (n=14) | 1.11x (n=14) | 4.14x (n=14) | 1.45x (n=4) | – | – | – | – | 7.27x (n=9) +5 |
| wloczykij | int32 | 0.91x (n=36) | 1.58x (n=36) | 2.48x (n=36) | – | 2.05x (n=18) | 0.55x (n=7) +1 | 0.25x (n=4) | – | 3.06x (n=2) +4 |
| wloczykij | int64 | 9.27x (n=22) +14 | 1.73x (n=36) | 1.25x (n=36) | – | – | – | – | – | – |
| wloczykij | float32 | – | 4.32x (n=36) | 3.54x (n=36) | – | – | – | – | – | – |
| wloczykij | float64 | – | 2.58x (n=36) | 2.07x (n=36) | – | – | – | – | 2.36x (n=36) | – |

#### Score + alignment

Geometric mean over the single-thread cells (fixture sets × global/local × affine/linear) that both nwgrad and the tool completed with correct scores; above 1, nwgrad is faster. `+k` = k more cells where the tool was stopped at 10× nwgrad's time. `n` = cells in the mean.

| host | dtype | parasail | seqan3 | seqan2 | ssw | ksw2 | wfa2 | edlib | biopython | scikit-bio | pyopal |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nighthaven | int16 | 1.76x (n=14) | 1.37x (n=14) | 1.14x (n=14) | 4.08x (n=4) | – | – | – | – | – | 8.29x (n=8) +6 |
| nighthaven | int32 | 1.44x (n=36) | 3.01x (n=36) | 1.91x (n=36) | – | 1.81x (n=18) | 0.39x (n=7) +1 | 0.52x (n=4) | – | – | 4.82x (n=2) +4 |
| nighthaven | int64 | 5.67x (n=32) +4 | 2.02x (n=36) | 1.56x (n=36) | – | – | – | – | – | – | – |
| nighthaven | float32 | – | 6.84x (n=20) +16 | 5.62x (n=34) +2 | – | – | – | – | – | – | – |
| nighthaven | float64 | – | 5.66x (n=36) | 3.74x (n=36) | – | – | – | – | 7.57x (n=23) +13 | 5.57x (n=24) +12 | – |
| oldsolace | int16 | 1.52x (n=14) | 0.93x (n=14) | 5.41x (n=8) +6 | 3.29x (n=4) | – | – | – | – | – | 7.97x (n=7) +7 |
| oldsolace | int32 | 1.03x (n=36) | 2.18x (n=36) | 4.70x (n=36) | – | 1.79x (n=18) | 0.37x (n=7) +1 | 0.57x (n=4) | – | – | 5.47x (n=2) +5 |
| oldsolace | int64 | 4.57x (n=34) +2 | 2.28x (n=36) | 2.13x (n=36) | – | – | – | – | – | – | – |
| oldsolace | float32 | – | 6.32x (n=27) +9 | 4.78x (n=36) | – | – | – | – | – | – | – |
| oldsolace | float64 | – | 4.89x (n=36) | 3.13x (n=36) | – | – | – | – | 6.92x (n=24) +12 | 3.75x (n=26) +10 | – |
| solace | int16 | 1.68x (n=14) | 1.39x (n=14) | 5.66x (n=7) +7 | 3.93x (n=4) | – | – | – | – | – | 7.79x (n=7) +7 |
| solace | int32 | 1.97x (n=36) | 3.03x (n=29) +7 | 6.28x (n=27) +9 | – | 2.65x (n=18) | 0.64x (n=7) +1 | 0.97x (n=4) | – | – | 5.98x (n=1) +13 |
| solace | int64 | 7.21x (n=15) +21 | 3.39x (n=35) +1 | 3.74x (n=34) +2 | – | – | – | – | – | – | – |
| solace | float32 | – | 5.68x (n=18) +18 | 5.39x (n=25) +11 | – | – | – | – | – | – | – |
| solace | float64 | – | 5.90x (n=28) +8 | 4.81x (n=35) +1 | – | – | – | – | 6.97x (n=10) +26 | – | – |
| spot | int16 | 1.72x (n=14) | 6.59x (n=6) +8 | 4.65x (n=9) +5 | 5.35x (n=4) | – | – | – | – | – | 8.29x (n=6) +8 |
| spot | int32 | 1.09x (n=36) | 5.51x (n=35) +1 | 2.94x (n=36) | – | 1.79x (n=18) | 0.28x (n=7) +1 | 0.34x (n=4) | – | – | 5.05x (n=2) |
| spot | int64 | 1.34x (n=36) | 1.84x (n=36) | 1.33x (n=36) | – | – | – | – | – | – | – |
| spot | float32 | – | 6.83x (n=31) +5 | 4.71x (n=35) +1 | – | – | – | – | – | – | – |
| spot | float64 | – | 4.53x (n=36) | 2.97x (n=36) | – | – | – | – | 5.70x (n=36) | – | – |
| wloczykij | int16 | 1.16x (n=14) | 1.24x (n=14) | 4.53x (n=14) | 2.31x (n=4) | – | – | – | – | – | 6.01x (n=10) +4 |
| wloczykij | int32 | 1.00x (n=36) | 1.97x (n=36) | 2.31x (n=36) | – | 1.14x (n=18) | 0.33x (n=7) +1 | 0.40x (n=4) | – | – | 3.85x (n=2) +1 |
| wloczykij | int64 | 6.43x (n=26) +10 | 1.98x (n=36) | 1.21x (n=36) | – | – | – | – | – | – | – |
| wloczykij | float32 | – | 5.73x (n=36) | 3.29x (n=36) | – | – | – | – | – | – | – |
| wloczykij | float64 | – | 3.31x (n=36) | 1.91x (n=36) | – | – | – | – | 5.38x (n=35) +1 | 2.52x (n=32) +4 | – |

#### Soft gradient

Geometric mean over the single-thread cells (fixture sets × global/local × affine/linear) that both nwgrad and the tool completed with correct scores; above 1, nwgrad is faster. `+k` = k more cells where the tool was stopped at 10× nwgrad's time. `n` = cells in the mean.

| host | dtype | deepblast |
|---|---|---|
| nighthaven | float64 | +7 |
| oldsolace | float64 | +7 |
| wloczykij | float64 | 6.31x (n=3) +4 |

## nwgrad's variants (default in bold everywhere else)

The default is `traceback="auto"`, `fill="interpair"`, `kernel="auto"`, the Python batch
API, gcc. Each variant changes one of these; the ratio is the variant's speed over the
default's in the same cell (above 1: the variant is faster).

| host | variant | dtype | cells | variant / default, geometric mean | range | faster in |
|---|---|---|---:|---:|---|---:|
| nighthaven | `api=functions` | float32 | 72 | 0.75 | 0.19–1.00 | 0 |
| nighthaven | `api=functions` | float64 | 108 | 0.82 | 0.31–1.37 | 11 |
| nighthaven | `fill=rowwise` | float64 | 72 | 0.74 | 0.36–1.14 | 1 |
| nighthaven | `fill=striped` | float32 | 108 | 0.84 | 0.30–1.02 | 0 |
| nighthaven | `fill=striped` | float64 | 144 | 0.92 | 0.49–1.29 | 2 |
| nighthaven | `kernel=scalar_fallback` | float32 | 108 | 0.20 | 0.10–0.67 | 0 |
| nighthaven | `kernel=scalar_fallback` | float64 | 108 | 0.34 | 0.10–0.75 | 0 |
| nighthaven | `kernel=sse2` | float32 | 108 | 0.68 | 0.49–0.98 | 0 |
| nighthaven | `kernel=sse2` | float64 | 108 | 0.62 | 0.42–0.91 | 0 |
| nighthaven | `kernel=sse4` | float32 | 108 | 0.71 | 0.52–0.97 | 0 |
| nighthaven | `kernel=sse4` | float64 | 108 | 0.67 | 0.42–0.93 | 0 |
| nighthaven | `soft_impl=log` | float64 | 36 | 0.10 | 0.10–0.14 | 0 |
| nighthaven | `traceback=grid` | float32 | 36 | 1.10 | 0.97–2.53 | 5 |
| nighthaven | `traceback=grid` | float64 | 36 | 1.00 | 0.98–1.05 | 0 |
| nighthaven | `traceback=hirschberg` | float32 | 72 | 0.69 | 0.32–1.30 | 1 |
| nighthaven | `traceback=hirschberg` | float64 | 72 | 0.66 | 0.30–1.02 | 0 |
| nighthaven | `traceback=hirschberg_pmax` | float32 | 36 | 0.74 | 0.33–1.32 | 1 |
| nighthaven | `traceback=hirschberg_pmax` | float64 | 36 | 0.61 | 0.27–1.01 | 0 |
| nighthaven | `traceback=pointers` | float32 | 72 | 0.95 | 0.43–2.53 | 5 |
| nighthaven | `traceback=pointers` | float64 | 72 | 0.90 | 0.39–1.04 | 0 |
| nighthaven | `traceback=scores` | float32 | 72 | 0.93 | 0.23–1.50 | 23 |
| nighthaven | `traceback=scores` | float64 | 72 | 0.82 | 0.19–1.30 | 12 |
| oldsolace | `api=functions` | float32 | 72 | 0.75 | 0.20–1.11 | 4 |
| oldsolace | `api=functions` | float64 | 108 | 0.89 | 0.31–1.34 | 24 |
| oldsolace | `fill=rowwise` | float64 | 72 | 0.67 | 0.20–1.24 | 4 |
| oldsolace | `fill=striped` | float32 | 108 | 0.84 | 0.35–1.13 | 9 |
| oldsolace | `fill=striped` | float64 | 144 | 0.92 | 0.53–1.24 | 15 |
| oldsolace | `kernel=scalar_fallback` | float32 | 108 | 0.22 | 0.10–0.68 | 0 |
| oldsolace | `kernel=scalar_fallback` | float64 | 108 | 0.37 | 0.16–0.69 | 0 |
| oldsolace | `kernel=sse2` | float32 | 108 | 0.67 | 0.42–1.09 | 2 |
| oldsolace | `kernel=sse2` | float64 | 108 | 0.63 | 0.40–1.09 | 1 |
| oldsolace | `kernel=sse4` | float32 | 108 | 0.71 | 0.43–1.07 | 1 |
| oldsolace | `kernel=sse4` | float64 | 108 | 0.69 | 0.42–1.08 | 1 |
| oldsolace | `soft_impl=log` | float64 | 36 | 0.10 | 0.10–0.10 | 0 |
| oldsolace | `traceback=grid` | float32 | 36 | 1.08 | 0.92–2.65 | 8 |
| oldsolace | `traceback=grid` | float64 | 36 | 0.99 | 0.93–1.11 | 4 |
| oldsolace | `traceback=hirschberg` | float32 | 72 | 0.66 | 0.31–1.08 | 3 |
| oldsolace | `traceback=hirschberg` | float64 | 72 | 0.65 | 0.35–1.09 | 4 |
| oldsolace | `traceback=hirschberg_pmax` | float32 | 36 | 0.83 | 0.44–1.09 | 6 |
| oldsolace | `traceback=hirschberg_pmax` | float64 | 36 | 0.65 | 0.37–1.06 | 3 |
| oldsolace | `traceback=pointers` | float32 | 72 | 0.98 | 0.55–2.58 | 12 |
| oldsolace | `traceback=pointers` | float64 | 72 | 0.91 | 0.54–1.10 | 10 |
| oldsolace | `traceback=scores` | float32 | 72 | 1.06 | 0.52–2.69 | 36 |
| oldsolace | `traceback=scores` | float64 | 72 | 0.98 | 0.52–1.39 | 27 |
| solace | `api=functions` | float32 | 72 | 0.73 | 0.12–1.09 | 3 |
| solace | `api=functions` | float64 | 108 | 0.76 | 0.20–1.33 | 10 |
| solace | `fill=rowwise` | float64 | 72 | 0.69 | 0.27–1.03 | 0 |
| solace | `fill=striped` | float32 | 108 | 0.84 | 0.16–1.12 | 3 |
| solace | `fill=striped` | float64 | 144 | 0.88 | 0.31–1.04 | 0 |
| solace | `kernel=avx2` | float32 | 108 | 0.86 | 0.51–2.92 | 25 |
| solace | `kernel=avx2` | float64 | 108 | 0.77 | 0.51–1.59 | 15 |
| solace | `kernel=scalar_fallback` | float32 | 108 | 0.18 | 0.10–0.66 | 0 |
| solace | `kernel=scalar_fallback` | float64 | 108 | 0.27 | 0.10–0.67 | 0 |
| solace | `kernel=sse2` | float32 | 108 | 0.58 | 0.23–1.53 | 9 |
| solace | `kernel=sse2` | float64 | 108 | 0.49 | 0.24–1.13 | 2 |
| solace | `kernel=sse4` | float32 | 108 | 0.61 | 0.32–1.69 | 11 |
| solace | `kernel=sse4` | float64 | 108 | 0.52 | 0.25–1.10 | 4 |
| solace | `soft_impl=log` | float64 | 36 | 0.10 | 0.10–0.10 | 0 |
| solace | `traceback=grid` | float32 | 36 | 1.08 | 0.95–2.56 | 5 |
| solace | `traceback=grid` | float64 | 36 | 0.99 | 0.93–1.02 | 0 |
| solace | `traceback=hirschberg` | float32 | 72 | 0.65 | 0.26–1.03 | 0 |
| solace | `traceback=hirschberg` | float64 | 72 | 0.64 | 0.31–1.03 | 0 |
| solace | `traceback=hirschberg_pmax` | float32 | 36 | 0.88 | 0.42–1.15 | 5 |
| solace | `traceback=hirschberg_pmax` | float64 | 36 | 0.71 | 0.35–1.04 | 0 |
| solace | `traceback=pointers` | float32 | 72 | 1.01 | 0.65–2.55 | 5 |
| solace | `traceback=pointers` | float64 | 72 | 0.95 | 0.60–1.12 | 2 |
| solace | `traceback=scores` | float32 | 72 | 0.95 | 0.41–1.64 | 19 |
| solace | `traceback=scores` | float64 | 72 | 0.86 | 0.33–1.24 | 17 |
| spot | `api=functions` | float32 | 72 | 0.82 | 0.30–1.16 | 1 |
| spot | `api=functions` | float64 | 108 | 1.00 | 0.45–1.71 | 27 |
| spot | `fill=rowwise` | float64 | 72 | 0.77 | 0.40–1.36 | 3 |
| spot | `fill=striped` | float32 | 108 | 0.88 | 0.40–1.33 | 3 |
| spot | `fill=striped` | float64 | 144 | 0.96 | 0.56–1.90 | 5 |
| spot | `kernel=scalar_fallback` | float32 | 108 | 0.24 | 0.10–0.66 | 0 |
| spot | `kernel=scalar_fallback` | float64 | 108 | 0.39 | 0.10–1.00 | 0 |
| spot | `soft_impl=log` | float64 | 36 | 0.10 | 0.10–0.10 | 0 |
| spot | `traceback=grid` | float32 | 36 | 1.09 | 0.99–2.25 | 4 |
| spot | `traceback=grid` | float64 | 36 | 1.00 | 0.99–1.01 | 0 |
| spot | `traceback=hirschberg` | float32 | 72 | 0.64 | 0.27–1.02 | 0 |
| spot | `traceback=hirschberg` | float64 | 72 | 0.60 | 0.24–1.00 | 0 |
| spot | `traceback=hirschberg_pmax` | float32 | 36 | 0.77 | 0.33–1.01 | 0 |
| spot | `traceback=hirschberg_pmax` | float64 | 36 | 0.62 | 0.26–1.01 | 0 |
| spot | `traceback=pointers` | float32 | 72 | 0.93 | 0.36–2.25 | 4 |
| spot | `traceback=pointers` | float64 | 72 | 0.86 | 0.34–1.01 | 0 |
| spot | `traceback=scores` | float32 | 72 | 1.15 | 0.47–3.02 | 34 |
| spot | `traceback=scores` | float64 | 72 | 1.07 | 0.43–1.77 | 34 |
| wloczykij | `api=functions` | float32 | 72 | 0.78 | 0.25–1.13 | 3 |
| wloczykij | `api=functions` | float64 | 108 | 0.90 | 0.39–1.49 | 11 |
| wloczykij | `fill=rowwise` | float64 | 72 | 0.88 | 0.35–1.62 | 6 |
| wloczykij | `fill=striped` | float32 | 108 | 0.90 | 0.50–1.37 | 5 |
| wloczykij | `fill=striped` | float64 | 144 | 0.96 | 0.59–1.61 | 12 |
| wloczykij | `kernel=scalar_fallback` | float32 | 108 | 0.35 | 0.11–0.72 | 0 |
| wloczykij | `kernel=scalar_fallback` | float64 | 108 | 0.64 | 0.18–1.19 | 3 |
| wloczykij | `kernel=sse2` | float32 | 108 | 0.94 | 0.77–1.11 | 7 |
| wloczykij | `kernel=sse2` | float64 | 108 | 0.94 | 0.71–1.06 | 3 |
| wloczykij | `soft_impl=log` | float64 | 36 | 0.11 | 0.10–0.25 | 0 |
| wloczykij | `traceback=grid` | float32 | 36 | 1.10 | 0.95–2.47 | 4 |
| wloczykij | `traceback=grid` | float64 | 36 | 0.99 | 0.82–1.07 | 1 |
| wloczykij | `traceback=hirschberg` | float32 | 72 | 0.67 | 0.35–1.09 | 2 |
| wloczykij | `traceback=hirschberg` | float64 | 72 | 0.64 | 0.34–1.09 | 1 |
| wloczykij | `traceback=hirschberg_pmax` | float32 | 36 | 0.74 | 0.36–1.05 | 0 |
| wloczykij | `traceback=hirschberg_pmax` | float64 | 36 | 0.61 | 0.32–1.09 | 1 |
| wloczykij | `traceback=pointers` | float32 | 72 | 0.96 | 0.44–2.49 | 4 |
| wloczykij | `traceback=pointers` | float64 | 72 | 0.88 | 0.42–1.08 | 4 |
| wloczykij | `traceback=scores` | float32 | 72 | 0.96 | 0.33–2.61 | 24 |
| wloczykij | `traceback=scores` | float64 | 72 | 0.85 | 0.29–1.24 | 18 |

What they say about the defaults (each holds on every host unless named):

- **Local affine traceback, long pairs.** At L10000 (10⁸ cells, above the 2²⁵-cell pointer
  budget) `auto` takes `hirschberg_pmax` at float32 and the integers; `pointers` (or `grid`,
  which runs as pointers for local) is **2.0–2.6× faster** single-threaded, and parasail
  beats nwgrad in exactly these cells. The budget bounds per-thread memory for many
  threads; with one thread the pointer table (~100 MB) costs far less than the recursion.
- **AVX-512 on short pairs.** On solace, `kernel="avx2"` beats the AVX-512 default in 29
  float32 cells, up to **2.9×** (DNA L100, affine global); even SSE2 / SSE4 win some.
- **`traceback="scores"`** (score tables, no direction codes) beats `auto` in 20–70 cells
  per host, by up to 3× (spot, local affine L10000) — mostly long local pairs and linear
  gaps.
- **Short soft gradients.** At L100 the per-pair functions (`api=functions`) are
  1.2–1.7× faster than the batch's shared inter-pair soft fill.
- **`soft_impl="log"`** was stopped at 10× the default (scaled) everywhere — the scaled
  default is worth what it promises.
- **`kernel="scalar_fallback"`** runs at 0.18–0.24× of the SIMD default; SSE2 / SSE4 on an
  AVX2 host at ~0.7×.

## Compilers and APIs

Ratios of speeds, geometric mean over the single-thread cells, default settings. The
release wheels are built with gcc on every platform.

| host | dtype | clang / gcc (Python) | C++ API / Python (gcc) | C++ clang / C++ gcc | cells |
|---|---|---:|---:|---:|---:|
| nighthaven | int16 | 0.98 | 1.11 | 0.95 | 42 |
| nighthaven | int32 | 1.00 | 1.02 | 1.01 | 108 |
| nighthaven | int64 | 1.04 | 1.00 | 1.03 | 108 |
| nighthaven | float32 | 0.99 | 1.02 | 0.98 | 108 |
| nighthaven | float64 | 0.99 | 1.00 | 0.99 | 144 |
| oldsolace | int16 | 0.96 | 1.03 | 0.95 | 42 |
| oldsolace | int32 | 1.01 | 1.02 | 1.01 | 108 |
| oldsolace | int64 | 1.04 | 1.01 | 1.03 | 108 |
| oldsolace | float32 | 0.96 | 1.02 | 0.97 | 108 |
| oldsolace | float64 | 0.98 | 1.02 | 0.98 | 144 |
| solace | int16 | 1.01 | 1.05 | 1.00 | 42 |
| solace | int32 | 0.99 | 1.02 | 1.01 | 108 |
| solace | int64 | 1.01 | 1.01 | 1.01 | 108 |
| solace | float32 | 0.99 | 1.01 | 0.99 | 108 |
| solace | float64 | 1.00 | 1.01 | 0.99 | 144 |
| spot | int16 | 0.92 | 0.97 | 0.94 | 42 |
| spot | int32 | 1.01 | 1.00 | 1.01 | 108 |
| spot | int64 | 0.99 | 1.00 | 0.98 | 108 |
| spot | float32 | 0.90 | 1.00 | 0.90 | 108 |
| spot | float64 | 0.90 | 1.00 | 0.90 | 144 |
| wloczykij | int16 | 0.89 | 1.00 | 0.90 | 42 |
| wloczykij | int32 | 1.01 | 1.00 | 1.01 | 108 |
| wloczykij | int64 | 1.04 | 1.00 | 1.03 | 108 |
| wloczykij | float32 | 0.97 | 1.00 | 0.98 | 108 |
| wloczykij | float64 | 0.98 | 1.00 | 0.98 | 144 |

Where the two compilers part, the cause was traced in the generated code (by a parallel
investigation, [PR #12](https://github.com/michalsta/nwgrad/pull/12), `c39a51c`; **the numbers above are `main`, without
these fixes**):

- *gcc, integer score*: signed-overflow rules keep gcc from folding `(v - go) - ge` into
  `v - (go + ge)`, and it keeps an `s > 0` test inside the loop; the striped profile was
  built by a scalar, strided loop.
- *clang, L100 path / hard*: the inter-pair fills reloaded `levels` / the score row from
  memory every column and tested the runtime Local flag inside the loop.
- *clang, int64 local linear*: LLVM's x86 cmov conversion turns the scalar `max(v, 0)` clamp
  into a data-dependent, mispredicted branch.
- *clang on AArch64, float (the ~10 % on spot)*: libstdc++'s `simd` max relies on
  `[[gnu::optimize("finite-math-only")]]`, which clang ignores — `fcmgt` + `bsl` where gcc
  emits one `fmaxnm`.

That branch's own A/B (not part of this run; relative to stock gcc): on nighthaven, over 116
L100/L300 cells, patched gcc 1.04× (integer score +14–20 %), stock clang 1.00×, patched
clang 1.11×; clang's L100 path/hard cells go from 0.80–0.92× to 1.09–1.30×, spot's float
cells from 0.81–0.91× to 1.00–1.08×, int64 local linear L300 from 0.50× to 1.01–1.05×.

### PR #12, re-measured

[PR #12](https://github.com/michalsta/nwgrad/pull/12) (`c39a51c`, on the `main` measured
here) was re-built with both compilers on the hosts that were free and timed against `main`
on every single-thread cell of the short-pair sets (protein L100 / L300, DNA L100, miRNA),
where its changes act. The four builds ran interleaved in rotating order, 8 processes per
build per cell, best pass kept; every result was checked against the float64 reference.
Ratios are speed over speed (above 1, the first build is faster).

#### nighthaven (244 cells)

| cells | n | PR gcc / main gcc | PR clang / main clang | main clang / main gcc | PR clang / PR gcc |
|---|---:|---:|---:|---:|---:|
| all | 244 | 1.029 | 1.072 | 0.986 | 1.027 |
| integer score (gcc folding fix) | 44 | 1.065 | 1.073 | 1.025 | 1.034 |
| L100 path / hard (inter-pair hoisting) | 80 | 1.022 | 1.120 | 0.963 | 1.056 |
| int64 local linear (x86 clang cmov) | 12 | 1.023 | 1.180 | 0.947 | 1.092 |
| float32 / float64 | 112 | 1.022 | 1.047 | 0.982 | 1.006 |
| miRNA | 64 | 1.014 | 1.038 | 0.961 | 0.984 |

- **gcc**, largest gains: protein_L300 local affine score int32 1.22, protein_L300 local linear score int32 1.20, protein_L300 global affine score int32 1.20, protein_L300 global linear score int32 1.20
- **gcc**, regressions beyond 5 %: none

- **clang**, largest gains: protein_L300 local linear hard int64 2.02, protein_L300 local linear path int64 1.98, protein_L100 local linear hard int32 1.46, protein_L100 local linear path int32 1.43
- **clang**, regressions beyond 5 %: none
- wrong scores: none

#### oldsolace (244 cells)

| cells | n | PR gcc / main gcc | PR clang / main clang | main clang / main gcc | PR clang / PR gcc |
|---|---:|---:|---:|---:|---:|
| all | 244 | 1.047 | 1.114 | 0.953 | 1.014 |
| integer score (gcc folding fix) | 44 | 1.081 | 1.085 | 1.016 | 1.019 |
| L100 path / hard (inter-pair hoisting) | 80 | 1.043 | 1.206 | 0.915 | 1.058 |
| int64 local linear (x86 clang cmov) | 12 | 1.041 | 1.209 | 0.913 | 1.060 |
| float32 / float64 | 112 | 1.027 | 1.079 | 0.947 | 0.995 |
| miRNA | 64 | 1.036 | 1.087 | 0.916 | 0.960 |

- **gcc**, largest gains: protein_L100 local linear score int32 1.36, protein_L100 global linear score int32 1.32, protein_L100 global linear score int64 1.30, protein_L100 global affine score int32 1.30
- **gcc**, regressions beyond 5 %: protein_L100 global linear hard float64 0.82; protein_L100 global linear path float64 0.86; protein_L100 local linear hard int64 0.92; protein_L100 local linear path float64 0.92; dna_L100 global affine path float64 0.93; dna_L100 global linear hard int64 0.93; mirna local affine score int16 0.94; dna_L100 local affine score int16 0.94; mirna local linear score float64 0.94; dna_L100 global affine score int16 0.94; mirna global affine score float32 0.95

- **clang**, largest gains: protein_L300 local linear hard int64 1.58, protein_L300 local linear path int64 1.56, protein_L300 global linear hard int64 1.55, protein_L100 local linear hard int32 1.53
- **clang**, regressions beyond 5 %: dna_L100 global affine hard int32 0.90
- wrong scores: none

#### solace (244 cells)

| cells | n | PR gcc / main gcc | PR clang / main clang | main clang / main gcc | PR clang / PR gcc |
|---|---:|---:|---:|---:|---:|
| all | 244 | 1.018 | 1.044 | 0.986 | 1.011 |
| integer score (gcc folding fix) | 44 | 1.063 | 1.058 | 1.020 | 1.015 |
| L100 path / hard (inter-pair hoisting) | 80 | 1.006 | 1.076 | 0.951 | 1.017 |
| int64 local linear (x86 clang cmov) | 12 | 1.035 | 1.089 | 0.873 | 0.919 |
| float32 / float64 | 112 | 1.010 | 1.026 | 0.976 | 0.992 |
| miRNA | 64 | 1.006 | 1.025 | 0.956 | 0.974 |

- **gcc**, largest gains: protein_L300 global linear score int64 1.18, protein_L300 global affine score int64 1.17, protein_L300 local affine score int64 1.16, protein_L300 local linear score int64 1.14
- **gcc**, regressions beyond 5 %: dna_L100 global linear hard int32 0.93

- **clang**, largest gains: protein_L100 local linear hard int64 1.33, protein_L100 local linear path int64 1.32, protein_L100 local linear hard int32 1.31, protein_L100 local linear path int32 1.26
- **clang**, regressions beyond 5 %: none
- wrong scores: none

The one consistent loss — gcc on oldsolace, linear-gap global protein L100 traceback at
float64 (0.82× hard, 0.86× path, neutral on the other two hosts) — is not a code-generation
change: it is the JCC erratum of Skylake-family cores (Kaby Lake has it, Alder Lake and Tiger
Lake do not). Rebuilt with `-Wa,-mbranches-within-32B-boundaries`, which pins branch
alignment, the PR is 1.03–1.05× faster than `main` on the same cells (main itself gains only
0–6 % from the flag: it had landed on a lucky alignment). Whether the release build should
pass that flag has not been measured on the other hosts.

## Multithreaded

nwgrad's speed-up is its multithread Mcell/s over its single-thread Mcell/s on the same fixture (the multithread sets are 10x larger). Ratios as above (nwgrad / tool); ‡ = harness-parallel (the tool has no threads of its own; the pairs were split over threads or processes by the harness), otherwise the tool's own threading.

| host | threads | dtype | nwgrad speed-up | parasail | seqan3 | seqan2 | ssw | ksw2 | wfa2 | edlib | biopython | scikit-bio | pyopal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nighthaven | 6 | int32 | 5.3x (4.1–7.5) | 1.46x ‡ | 2.31x +6 | 5.73x +12 | – | 2.70x ‡ | 0.84x ‡ +2 | 1.22x ‡ | – | – | +16 |
| nighthaven | 6 | float32 | 5.3x (4.2–7.6) | – | – | – | – | – | – | – | – | – | – |
| nighthaven | 6 | float64 | 5.1x (2.9–5.6) | – | – | – | – | – | – | – | 5.64x ‡ +13 | 5.97x ‡ +10 | – |
| oldsolace | 4 | int32 | 2.9x (2.7–3.2) | 1.18x ‡ | 1.80x | – | – | 2.84x ‡ | 0.83x ‡ +2 | 1.18x ‡ | – | – | +9 |
| oldsolace | 4 | float32 | 2.9x (2.6–3.2) | – | – | – | – | – | – | – | – | – | – |
| oldsolace | 4 | float64 | 2.9x (2.6–3.1) | – | – | – | – | – | – | – | 5.16x ‡ +11 | 4.25x ‡ +7 | – |
| solace | 8 | int32 | 5.2x (4.3–6.4) | 1.72x ‡ | 2.53x +11 | – | – | 3.43x ‡ +1 | 1.36x ‡ +2 | 1.77x ‡ +1 | – | – | +37 |
| solace | 8 | float32 | 5.3x (4.2–6.8) | – | – | – | – | – | – | – | – | – | – |
| solace | 8 | float64 | 5.2x (4.0–6.1) | – | – | – | – | – | – | – | 4.63x ‡ +22 | – | – |
| spot | 8 | int32 | 4.9x (3.6–5.9) | 1.43x ‡ | 5.41x +9 | 7.94x | – | 2.35x ‡ | 0.58x ‡ +2 | 0.68x ‡ | – | – | +3 |
| spot | 8 | float32 | 5.2x (4.1–5.9) | – | – | – | – | – | – | – | – | – | – |
| spot | 8 | float64 | 5.1x (3.6–5.9) | – | – | – | – | – | – | – | 4.38x ‡ | – | – |
| wloczykij | 32 | int32 | 18.9x (9.0–28.1) | 1.05x ‡ | 2.41x +11 | – | – | 1.53x ‡ | 0.87x ‡ +1 | 2.44x ‡ | – | – | +4 |
| wloczykij | 32 | float32 | 19.2x (9.1–26.2) | – | – | – | – | – | – | – | – | – | – |
| wloczykij | 32 | float64 | 21.0x (9.8–27.8) | – | – | – | – | – | – | – | 3.66x ‡ | 3.00x ‡ +4 | – |
| wloczykij | 8:numa0 | int32 | 4.1x (3.5–5.4) | 1.00x ‡ | 1.51x | – | – | 1.47x ‡ | 0.53x ‡ +2 | 0.75x ‡ | – | – | +4 |
| wloczykij | 8:numa0 | float32 | 4.4x (3.9–5.6) | – | – | – | – | – | – | – | – | – | – |
| wloczykij | 8:numa0 | float64 | 4.6x (3.8–6.0) | – | – | – | – | – | – | – | 3.19x ‡ | 2.63x ‡ +4 | – |

## Every cell

Every cell of every host — each tool's best correct arm, the ratio to nwgrad's default, and
the best of nwgrad's own variants — is in [benchmarks-cells.md](benchmarks-cells.md)
(generated; the raw per-arm records are in
[`tools/fleetbench/results/`](../tools/fleetbench/results/)).

## Competitor notes

- **KSW2** silently clips scores above `mat[0] + 2(q+e)`: its SSE kernel takes the first
  matrix entry to be the largest score, true for match/mismatch DNA and false for BLOSUM62
  (A–A 4, W–W 11). The harness renumbers the alphabet so the largest score comes first —
  the same problem, within KSW2's contract. Without it, global BLOSUM62 scores came out
  ~25 % low.
- **WFA2-lib** built by gcc 16 with its own flags (`-O3 -march=native`) crashes in its
  blockwise match-extension kernel on most DNA pairs (an out-of-range read past its padded
  sequence buffer; gcc `-O3`, gcc `-O2 -march=native` and clang are all fine), and its
  headers misbehave when included from C++. It is built with clang and called through a
  C shim.
- **Farrar striped kernels with linear gaps** (gap open = extend): parasail `striped` and
  SSW return scores a few points low, parasail `diag` returns 0. Their cells use parasail's
  `scan`, which is exact.
- **SeqAn3** cannot build its vectorised int64 aligner for DNA scoring (a `1 << 63` on an
  `int` in `simd_match_mismatch_scoring_scheme.hpp`). SeqAn 2's SIMD backend covers 8/16/32-bit
  integer scores on x86 only: its `Vectorial` policy does not compile for int64 or floating
  scores there, and on ARM (spot) it compiles but runs without SIMD — those spot cells are
  SeqAn 2's policy as shipped, at scalar speed.
- **pyopal** aligns one query against a database; one target per call makes it pay per-call
  overhead, which dominates short pairs.

## Also found

- **nwgrad's default thread count on Apple M1 is 4, not 8.** `n_threads=0` counts unique
  (package, core_id) pairs, and Asahi Linux numbers the cores of both clusters 0–3. The
  multithread runs on spot use 8.

## Not measured

- GPU paths (DeepBLAST's CUDA decoder and GPU aligners).
- PR #12 on spot (powered off at the time — its NEON float change is unmeasured here) and on
  wloczykij (still running the main benchmark).
- scikit-bio and DeepBLAST on spot and solace (metered connections: their dependency stacks
  are hundreds of MB).

## Reproducing

See [`tools/fleetbench/README.md`](../tools/fleetbench/README.md). In short:

```bash
tools/fleetbench/make_bundle.sh bundle.tar.xz          # nwgrad + harness + competitor sources
python tools/fleetbench/make_fixtures.py --fasta GRCh38_protein.faa \
       --manakov manakov_fit_rc.tsv --out fixtures/
# on each host, with the bundle unpacked into ~/nwbench and fixtures/ beside it:
~/nwbench/bench/setup_host.sh ~/nwbench                 # venvs, both nwgrad builds, every driver
~/nwbench/venv-gcc/bin/python ~/nwbench/bench/runner.py --root ~/nwbench --reps 3 --mt <cores>
```

`coordinator.py` drives the same across a fleet over ssh and survives hosts going away;
`build_doc.py` regenerates this page from the results.
