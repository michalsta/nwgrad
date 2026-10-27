# Benchmarks against other aligners

nwgrad measured against every pairwise aligner we could find with a C, C++ or Python
interface, on the same problems, on five machines (AVX2, AVX-512, NEON, SSE4), single- and
multithreaded. Measured <!-- GEN:date --> with nwgrad from GitHub `main` (<!-- GEN:commit -->);
the harness is [`tools/fleetbench/`](../tools/fleetbench/), the raw per-arm results are in
[`tools/fleetbench/results/`](../tools/fleetbench/results/).

<!-- GEN:inshort -->

## What was compared

Comparisons are strictly **within one cell**: the same fixture, alignment mode (global /
local), gap model (affine / linear), output (score only / score + alignment / hard gradient /
soft gradient) and number type. A tool that has no implementation for a cell shows `–`
there; a cell nwgrad itself cannot run (no float32 soft path, an int16 overflow bound) is
left out entirely. In every cell each tool is represented by **its best implementation** —
SIMD where it has SIMD, its fastest variant among those that returned correct scores.

| tool | version | how it was built | what was run |
|---|---|---|---|
| **nwgrad** | `main` <!-- GEN:commit --> | from source, gcc **and** clang, release-wheel flags (below); Python batch API and the C++ API | everything; default plus variants |
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

<!-- GEN:hosts -->

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

<!-- GEN:summary -->

## nwgrad's variants (default in bold everywhere else)

The default is `traceback="auto"`, `fill="interpair"`, `kernel="auto"`, the Python batch
API, gcc. Each variant changes one of these; the ratio is the variant's speed over the
default's in the same cell (above 1: the variant is faster).

<!-- GEN:variants -->

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

<!-- GEN:compilers -->

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

<!-- GEN:ab -->

The one consistent loss — gcc on oldsolace, linear-gap global protein L100 traceback at
float64 (0.82× hard, 0.86× path, neutral on the other two hosts) — is not a code-generation
change: it is the JCC erratum of Skylake-family cores (Kaby Lake has it, Alder Lake and Tiger
Lake do not). Rebuilt with `-Wa,-mbranches-within-32B-boundaries`, which pins branch
alignment, the PR is 1.03–1.05× faster than `main` on the same cells (main itself gains only
0–6 % from the flag: it had landed on a lucky alignment). Whether the release build should
pass that flag has not been measured on the other hosts.

## Multithreaded

<!-- GEN:mt -->

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
