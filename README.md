# nwgrad

**Differentiable sequence alignment.** nwgrad runs Needleman–Wunsch (global) and
Smith–Waterman (local) alignment and returns not just the score but its **gradient
with respect to the substitution matrix and the gap penalties** — so alignment
parameters can be learned by gradient descent, inside any ML pipeline.

```python
score, grad = nwgrad.SeqPair("PLEASANTLY", "MEANLY", params).score_and_grad()
params = params + 0.01 * grad   # every matrix cell and all four gap costs move together
```

## Why nwgrad

- **Gradients, not just scores.** The *hard* gradient counts substitution pairs and
  gap positions along the optimal path (a subgradient of the Viterbi score). The
  *soft* gradient is the exact gradient of the log-partition function over **all**
  alignments, computed by forward–backward in log space. Matrix and gap fields share
  one sign convention, so a single update rule moves them all.
- **Fast.** Striped SIMD kernels for SSE2, AVX2, AVX-512 and NEON ship in one binary
  and are chosen at runtime. Every one is **bit-exact** with the scalar reference: the
  instruction set is a speed knob, never a correctness one. Batches run on all cores;
  measured at 6.4 billion DP cells per second on a 16-thread laptop CPU (Intel
  i7-11850H, AVX-512, float32).
- **Linear memory for long sequences.** Affine global alignment defaults to
  Hirschberg (Myers–Miller) traceback, which keeps O(m+n) state instead of O(m×n)
  tables: titin (35,991 residues) aligned against itself needs about 0.9 MB of DP
  memory per thread rather than 3.9 GB.
- **Built for training loops.** Sequences are validated and encoded once and reused
  across steps. After a parameter update, banded re-alignment around the previous
  path costs O(length × bandwidth) instead of a full O(m × n) DP.
- **Any alphabet.** Protein, DNA, RNA or your own symbol set; asymmetric matrices and
  asymmetric gap costs; BLOSUM, PAM, VTML and NUC44 included.
- **Python and C++.** nanobind bindings with zero-copy numpy interop over a
  header-only C++20 core.

## Installation

```bash
pip install nwgrad
```

Prebuilt wheels cover CPython 3.10–3.14 on Linux x86_64 / aarch64 and macOS arm64.
Anywhere else pip builds from source, which needs a C++20 compiler with libstdc++'s
`<experimental/simd>`: GCC 11+, or Clang 13+ built against libstdc++. Apple's
system clang (libc++) and MSVC lack that header — on macOS use Homebrew `gcc`;
Windows is not supported.

## Quick start

```python
import nwgrad
from nwgrad.matrices import BLOSUM62        # ready-made SubstMatrix, alphabet included

params = nwgrad.AlignParams(BLOSUM62,
                            gap_open_a=11.0, gap_extend_a=1.0,
                            gap_open_b=11.0, gap_extend_b=1.0)

sp = nwgrad.SeqPair("PLEASANTLY", "MEANLY", params,
                    gap_model="affine",     # or "linear"
                    mode="global",          # or "local"
                    grad_mode="hard")       # or "soft", or "none"

score, grad = sp.score_and_grad()
print(score)
print(sp.formatted())                       # the alignment, ready to read
g = grad.to_dict()                          # numpy matrix + the four gap fields
print(g["matrix"].shape, g["gap_open_a"], g["gap_extend_b"])
```

A gradient is itself an `AlignParams`, so `params + lr * grad` is a complete update
step. Gap penalties are costs: a gap of length `k` costs `gap_extend * k` (linear) or
`gap_open + gap_extend * k` (affine), matching BioPython; `_a` applies to gaps in
sequence A, `_b` to gaps in sequence B.

## Learning a substitution matrix

Any differentiable objective over alignment scores works. Here, a contrastive one:
make a set of pairs you believe are homologous score higher than a set of decoys.

```python
pos = nwgrad.SeqPairBatch()                 # all cores by default
neg = nwgrad.SeqPairBatch()
pos.add_many(homologs_a, homologs_b, params, grad_mode="soft")
neg.add_many(decoys_a, decoys_b, params, grad_mode="soft")

for step in range(100):
    margin = pos.score_and_grad() - neg.score_and_grad()     # sums of log Z
    grad = pos.compute_grad() - neg.compute_grad()            # d margin / d params
    params = params + 1e-3 * grad
    pos.set_params(params)
    neg.set_params(params)
```

`add_many()` builds the pairs in parallel C++ with no per-pair Python objects, so
batches of millions of pairs are cheap to set up. Per-pair scores and gradients are
available too (`batch.scores()`, `batch.grads()`, `batch.weighted_grad(weights)`, `batch[i]`), for
objectives that weight each pair differently.

## Good to know

- **Precision.** The plain names (`SeqPair`, `SeqPairBatch`, `nw_score`, …) run the
  DP in float32; `SeqPairDouble`, `SeqPairBatchDouble`, `nw_score_double`, … run it in
  float64. Inputs and outputs are float64 either way.
- **Many short pairs.** For millions of short pairs (e.g. miRNA × target site),
  `batch.fill = "interpair"` aligns several pairs per vector instruction — about 6×
  the default `"striped"` fill per pair on AVX2 — with bit-identical results; see
  [`SeqPairBatch.fill`](docs/api.md#seqpairbatch).
- **Stateless one-offs.** Twelve functions (`nw_score`, `sw_affine_grad`,
  `nw_affine_soft_grad`, …) align a single pair without keeping any state.
- **C++.** `python -m nwgrad --include` prints the path to the header-only library.

## Documentation

- [Python API reference](https://github.com/michalsta/nwgrad/blob/main/docs/api.md) —
  `Alphabet`, `SubstMatrix`, `AlignParams`, `SeqPair`, `SeqPairBatch`, `BatchAligner`
  and the convenience functions, with every argument and contract.
- [Precision, traceback modes and SIMD kernels](https://github.com/michalsta/nwgrad/blob/main/docs/tuning.md) —
  what the defaults are, when to change them, and what each choice costs.
- [C++ library and building from source](https://github.com/michalsta/nwgrad/blob/main/docs/cpp.md) —
  using the headers directly, running the C++ and sanitizer test suites.

## License

MIT — see [LICENSE](https://github.com/michalsta/nwgrad/blob/main/LICENSE).
