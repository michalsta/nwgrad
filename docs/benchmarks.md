# Benchmarks against other aligners

nwgrad measured against the most used pairwise aligners with a Python interface, on
the same problems, single-threaded. Measured 2026-10-02 with nwgrad 0.5.0; the
harness is [`tools/bench_competitors.py`](../tools/bench_competitors.py).

**In short:** nwgrad is the fastest of these at **global alignment with traceback** —
2.1–3.4× parasail and up to 8.6× Biopython on AVX2 — while also computing the gradient
none of the others provide. parasail is faster at **score-only** global alignment of
long sequences (16-bit integer lanes) and at **local** alignment. nwgrad's linear soft
gradient matches DeepBLAST on AVX2; its affine soft gradient costs 2.2–3.5× more.

## What was compared

| tool | version | what it is | arms |
|---|---|---|---|
| **nwgrad** | 0.5.0 | this package; float32 striped SIMD, runtime ISA dispatch | all four |
| [parasail](https://github.com/jeffdaily/parasail) | 1.3.4 | striped SIMD library, the de-facto fast aligner; `*_striped_sat` (16-bit, rerun at 32-bit on overflow) | score, global, local |
| [Biopython](https://biopython.org) `PairwiseAligner` | 1.88 | the most widely installed; C implementation | score, global, local |
| [pyopal](https://github.com/althonos/pyopal) | 0.7.3 | Opal SIMD database search; one query vs one target here | score, global, local |
| [scikit-bio](https://scikit.bio) `pair_align` | 0.7.4 | 0.7's new aligner (0.7 dropped `StripedSmithWaterman`) | global, local |
| [DeepBLAST](https://github.com/flatironinstitute/deepblast) `NeedlemanWunschDecoder` | 1.0.3 | differentiable NW in PyTorch + numba | soft gradient |

## Method

- **Same problem, verified.** BLOSUM62, affine gaps with nwgrad's open 11, extend 1
  (a gap of length *k* costs 11 + *k*). The other tools spell this differently; each
  mapping was found by exact score agreement on 30 pairs (every wrong alternative
  missed 29/30): parasail and pyopal `open=12, extend=1`, Biopython
  `open_gap_score=-12, extend_gap_score=-1`, scikit-bio `gap_cost=(11, 1)` **with
  `free_ends=False`** (its default is semi-global). Before anything is timed, every
  score-reporting arm must reproduce nwgrad's float64 score on all 24 gate pairs; on
  both hosts all did.
- **Fixture.** Real human proteins (GRCh38 RefSeq proteome), each paired with a mutated
  copy — 20 % substitutions, ~3 % deletions, ~3 % insertion runs of 1–5 — so the pairs
  are homolog-like rather than random. About 3×10⁷ DP cells per length: 1574 pairs at
  length ~100, 333 at ~300, 30 at ~1000, 3 at ~3000. The fixture was frozen to JSON
  and replayed on both hosts, so both aligned identical sequences.
- **Timing.** Single thread. Each (tool, length) in its own process (arms sharing one
  contaminate each other's allocator and caches), warm-up excluded (this matters for
  numba), best of 3 passes over the pair set. Reported in Mcell/s = (m × n) / time.
- **What each arm does.** *Score only*: `nw_score_affine` vs each tool's score call.
  *Score + alignment*: nwgrad's `nw_affine_grad` / `sw_affine_grad` — which also
  computes the full gradient, work no competitor does — vs each tool's traceback
  (parasail's CIGAR, Biopython's first alignment, pyopal `mode="full"`, scikit-bio's
  path). *Soft gradient*: one forward–backward pass producing the gradient.

## Results

`x` is nwgrad's speed divided by the tool's: above 1, nwgrad is faster. In the
soft-gradient tables the ratio is against `nwgrad-linear`, the like-for-like arm (see
below).

### nighthaven — Intel Core i5-12500 (Alder Lake), AVX2 for nwgrad and parasail

The quieter of the two hosts; quote these.

#### Global affine, score only

| length | `nwgrad` | `parasail` | `biopython` | `pyopal` |
|---:|---:|---:|---:|---:|
| 100 | 505.2 | 427.9 (1.18x) | 371.1 (1.36x) | 119.0 (4.25x) |
| 300 | 621.3 | 902.5 (0.69x) | 387.9 (1.60x) | 432.0 (1.44x) |
| 1000 | 892.8 | 1548.7 (0.58x) | 384.7 (2.32x) | 610.5 (1.46x) |
| 3000 | 957.0 | 1901.0 (0.50x) | 381.0 (2.51x) | 642.6 (1.49x) |

#### Global affine, score + alignment

| length | `nwgrad` | `parasail` | `biopython` | `pyopal` | `scikit-bio` |
|---:|---:|---:|---:|---:|---:|
| 100 | 487.4 | 142.6 (3.42x) | 98.8 (4.93x) | 106.2 (4.59x) | 123.7 (3.94x) |
| 300 | 615.4 | 259.3 (2.37x) | 107.5 (5.72x) | 312.2 (1.97x) | 175.6 (3.50x) |
| 1000 | 889.4 | 396.9 (2.24x) | 107.8 (8.25x) | 355.0 (2.51x) | 132.1 (6.73x) |
| 3000 | 956.2 | 463.3 (2.06x) | 111.8 (8.55x) | 195.2 (4.90x) | 148.0 (6.46x) |

#### Local affine, score + alignment

| length | `nwgrad` | `parasail` | `biopython` | `pyopal` | `scikit-bio` |
|---:|---:|---:|---:|---:|---:|
| 100 | 214.6 | 207.7 (1.03x) | 66.6 (3.22x) | 92.1 (2.33x) | 124.6 (1.72x) |
| 300 | 247.7 | 511.6 (0.48x) | 71.5 (3.46x) | 237.8 (1.04x) | 173.2 (1.43x) |
| 1000 | 223.2 | 551.1 (0.40x) | 70.3 (3.18x) | 319.9 (0.70x) | 134.9 (1.65x) |
| 3000 | 234.2 | 690.3 (0.34x) | 71.0 (3.30x) | 190.9 (1.23x) | 142.7 (1.64x) |

#### Soft (forward–backward) gradient

| length | `nwgrad` | `nwgrad-linear` | `deepblast` |
|---:|---:|---:|---:|
| 100 | 5.6 | 14.8 | 14.6 (1.01x) |
| 300 | 5.2 | 13.9 | 16.1 (0.87x) |
| 1000 | 4.1 | 14.5 | 15.1 (0.97x) |
| 3000 | 5.0 | 14.7 | 12.7 (1.16x) |

### skynet — 60-vCPU KVM guest on AMD Opteron 6380 (Piledriver)

nwgrad runs its SSE2 level here (Piledriver has no AVX2, and its 256-bit AVX is
split into two 128-bit halves, so nwgrad deliberately does not use it); parasail picks
SSE4.1. Another session kept the load average at 2.3–3.5 during this run — mild for
single-threaded timing on 60 vCPUs, but this host is a VM, so treat these as
secondary.

#### Global affine, score only

| length | `nwgrad` | `parasail` | `biopython` | `pyopal` |
|---:|---:|---:|---:|---:|
| 100 | 151.6 | 158.2 (0.96x) | 135.4 (1.12x) | 45.6 (3.32x) |
| 300 | 187.6 | 323.1 (0.58x) | 155.5 (1.21x) | 169.8 (1.10x) |
| 1000 | 192.6 | 560.0 (0.34x) | 154.7 (1.25x) | 243.2 (0.79x) |
| 3000 | 235.9 | 657.1 (0.36x) | 159.9 (1.48x) | 268.0 (0.88x) |

#### Global affine, score + alignment

| length | `nwgrad` | `parasail` | `biopython` | `pyopal` | `scikit-bio` |
|---:|---:|---:|---:|---:|---:|
| 100 | 146.8 | 56.6 (2.59x) | 36.7 (4.00x) | 39.2 (3.74x) | 39.2 (3.75x) |
| 300 | 186.4 | 104.5 (1.78x) | 40.8 (4.57x) | 110.6 (1.68x) | 93.0 (2.00x) |
| 1000 | 191.1 | 145.6 (1.31x) | 39.8 (4.80x) | 88.9 (2.15x) | 60.6 (3.15x) |
| 3000 | 234.1 | 152.5 (1.53x) | 38.5 (6.07x) | 49.9 (4.69x) | 73.0 (3.21x) |

#### Local affine, score + alignment

| length | `nwgrad` | `parasail` | `biopython` | `pyopal` | `scikit-bio` |
|---:|---:|---:|---:|---:|---:|
| 100 | 73.6 | 77.4 (0.95x) | 22.1 (3.33x) | 36.3 (2.03x) | 37.6 (1.96x) |
| 300 | 85.4 | 192.2 (0.44x) | 24.2 (3.53x) | 94.4 (0.90x) | 86.5 (0.99x) |
| 1000 | 71.9 | 234.5 (0.31x) | 23.3 (3.09x) | 88.5 (0.81x) | 56.7 (1.27x) |
| 3000 | 73.0 | 231.1 (0.32x) | 23.0 (3.18x) | 53.0 (1.38x) | 67.7 (1.08x) |

#### Soft (forward–backward) gradient

| length | `nwgrad` | `nwgrad-linear` | `deepblast` |
|---:|---:|---:|---:|
| 100 | 1.2 | 3.5 | 3.4 (1.03x) |
| 300 | 1.0 | 2.8 | 3.7 (0.75x) |
| 1000 | 1.0 | 2.2 | 3.7 (0.60x) |
| 3000 | 0.9 | 2.0 | 3.0 (0.67x) |

## Reading the results

- **Global alignment with traceback is nwgrad's strongest case**, at every length on
  both hosts — and the nwgrad arm also computes the gradient. It is the operation a
  training loop actually needs.
- **Score-only global, long sequences: parasail is up to 2× faster** (0.50× at length
  3000 on AVX2). Its `_sat` kernels compute in 16-bit integers, twice as many cells per
  vector as nwgrad's float32. nwgrad uses floating point because the matrix it
  differentiates is learned and real-valued; integer scoring is not an option for it.
- **Local alignment: parasail is 2–3× faster from length 300 up.** nwgrad's local path
  stays flat at ~230 Mcell/s on AVX2 at every length while its global path climbs to
  ~950, so local is not getting what global gets. Not yet diagnosed — the clearest
  optimization target this benchmark found.
- **Soft gradient.** DeepBLAST's model is single-state (one softmax over three moves per
  cell), so the like-for-like arm is nwgrad's *linear* soft gradient: it ties DeepBLAST
  on AVX2 (13.9–14.8 vs 12.7–16.1 Mcell/s); on SSE2 it ties at length 100 and trails
  1.3–1.7× from 300 up. nwgrad's affine soft gradient runs three coupled states and is
  2.2–3.5× slower again. At ~1 µs per
  cell the scalar log-space soft path is the second optimization target. DeepBLAST also
  solves a different model — it adds the match score at every cell whatever the move,
  and leaves end gaps free — so its scores cannot be checked against nwgrad's; only the
  cost per cell is compared.
- **pyopal** is built for one query against a database; one target per call makes it
  pay per-call overhead, which dominates at length 100.

## Not measured

- **Multithreading.** nwgrad's `SeqPairBatch` runs pairs across all cores natively;
  none of the other tools has a batch API for independent pairs (pyopal's threads
  parallelise one query over many targets). Single-threaded numbers are the fair
  per-core comparison.
- **AVX-512 and ARM.** solace (AVX-512) and spot (M1) were not used.
- **GPU** paths (DeepBLAST's CUDA decoder).

## Reproducing

```bash
pip install parasail biopython pyopal scikit-bio torch \
            git+https://github.com/flatironinstitute/deepblast.git
python tools/bench_competitors.py --pairs-out pairs.json          # needs the proteome
python tools/bench_competitors.py --pairs-in pairs.json           # replay elsewhere
python tools/bench_competitors.py --quick --lengths 100 300       # minutes, rough
```

Without a FASTA (`--fasta`, default the GRCh38 proteome in the repo root) the harness
falls back to synthetic sequences.
