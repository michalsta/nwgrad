# nwgrad

High-performance C++ sequence alignment library with **gradient computation** for substitution matrix optimization — exposed to Python via [nanobind](https://github.com/wjakob/nanobind).

nwgrad implements Needleman-Wunsch (global) and Smith-Waterman (local) alignment with both linear and affine gap penalties. Its primary novel feature is computing the **gradient of the alignment score with respect to substitution matrix entries**, enabling BLOSUM matrix optimization in gradient-based ML pipelines.

## Features

- **Global (NW) and local (SW) alignment** — shared template core, selected at compile time
- **Linear and affine gap models** — single gap penalty or gap-open + gap-extend
- **Hard subgradient** — substitution-pair counts along the optimal traceback
- **Soft (differentiable) gradient** — forward-backward in log-space; log-partition function and expected substitution counts
- **Multithreaded batch processing** — lock-free work dispatch, per-thread gradient accumulation
- **Double precision throughout** — no float truncation in the hot path
- **Zero-copy Python interface** — nanobind buffer protocol; no unnecessary array copies

## Installation

```bash
pip install nwgrad
```

Or from source:

```bash
git clone https://github.com/your-org/nwgrad
cd nwgrad
pip install .
```

Requires Python ≥ 3.8 and a C++20 compiler (GCC 11+ or Clang 13+).

## Quick Start

```python
import numpy as np
import nwgrad

# BLOSUM62 — 20×20 float64, canonical AA order: ACDEFGHIKLMNPQRSTVWY
blosum62 = nwgrad.BlosumMatrix(BLOSUM62_ARRAY)

# Single-pair alignment score
score = nwgrad.nw_score_affine("PLEASANTLY", "MEANLY", blosum62,
                               gap_open=11.0, gap_extend=1.0)

# Score + hard subgradient (substitution-pair counts)
score, grad = nwgrad.nw_affine_grad("PLEASANTLY", "MEANLY", blosum62,
                                    gap_open=11.0, gap_extend=1.0)
# grad is a (20, 20) numpy array

# Score + soft (differentiable) gradient (log-partition function)
log_z, grad = nwgrad.nw_affine_soft_grad("PLEASANTLY", "MEANLY", blosum62,
                                         gap_open=11.0, gap_extend=1.0)
```

### Batch alignment

```python
aligner = nwgrad.BatchAligner(
    matrix=blosum62,
    gap_open=11.0,
    gap_extend=1.0,
    gap_model="affine",   # "linear" | "affine"
    mode="global",        # "global" | "local"
    grad_mode="hard",     # "hard" | "soft" | "none"
    n_threads=8,
)

seqs_a = ["PLEASANTLY", "ACDEFGHIKL", ...]
seqs_b = ["MEANLY",     "CDEFGHIKLM", ...]

result = aligner.align(seqs_a, seqs_b)
result.scores   # np.ndarray float64, shape (N,)
result.grad     # np.ndarray float64, shape (20, 20), summed over all pairs
```

## API Reference

### `BlosumMatrix`

```python
nwgrad.BlosumMatrix(matrix: np.ndarray)
```

Constructs a substitution matrix from a `(20, 20)` float64 numpy array in canonical amino acid order (`ACDEFGHIKLMNPQRSTVWY`). The matrix is stored internally as a `double[256][256]` ASCII-indexed table for O(1) lookup without any char-to-index mapping.

| Method | Description |
|---|---|
| `score(a, b)` | Look up the substitution score for single characters `a`, `b` |
| `to_matrix()` | Export as `(20, 20)` float64 numpy array |

---

### Single-pair score functions

All return a `float` (the alignment score).

| Function | Gap model | Alignment |
|---|---|---|
| `nw_score(a, b, matrix, gap_extend)` | Linear | Global (NW) |
| `sw_score(a, b, matrix, gap_extend)` | Linear | Local (SW) |
| `nw_score_affine(a, b, matrix, gap_open, gap_extend)` | Affine | Global (NW) |
| `sw_score_affine(a, b, matrix, gap_open, gap_extend)` | Affine | Local (SW) |

---

### Hard gradient functions

All return `(score: float, grad: np.ndarray[20, 20])`.

The gradient is the substitution-pair count vector of the optimal alignment — i.e., `grad[i, j]` is the number of times amino acid `i` is aligned to amino acid `j` in the traceback. This is the subgradient of the score with respect to the substitution matrix.

| Function | Gap model | Alignment |
|---|---|---|
| `nw_grad(a, b, matrix, gap_extend)` | Linear | Global |
| `sw_grad(a, b, matrix, gap_extend)` | Linear | Local |
| `nw_affine_grad(a, b, matrix, gap_open, gap_extend)` | Affine | Global |
| `sw_affine_grad(a, b, matrix, gap_open, gap_extend)` | Affine | Local |

---

### Soft gradient functions

All return `(log_z: float, grad: np.ndarray[20, 20])`.

`log_z` is `log Σ exp(score(alignment))` summed over all alignments (the log-partition function). `grad[i, j]` is the expected number of times amino acid `i` is aligned to amino acid `j`, taken over all alignments weighted by their Boltzmann probability. This is the true gradient of `log_z` with respect to `matrix[i, j]`.

| Function | Gap model | Alignment |
|---|---|---|
| `nw_soft_grad(a, b, matrix, gap_extend)` | Linear | Global |
| `sw_soft_grad(a, b, matrix, gap_extend)` | Linear | Local |
| `nw_affine_soft_grad(a, b, matrix, gap_open, gap_extend)` | Affine | Global |
| `sw_affine_soft_grad(a, b, matrix, gap_open, gap_extend)` | Affine | Local |

---

### `BatchAligner`

```python
nwgrad.BatchAligner(
    matrix,
    gap_open=11.0,
    gap_extend=1.0,
    gap_model="affine",   # "linear" | "affine"
    mode="global",        # "global" | "local"
    grad_mode="hard",     # "hard" | "soft" | "none"
    n_threads=1,
)
```

| Parameter | Type | Description |
|---|---|---|
| `matrix` | `BlosumMatrix` | Substitution matrix |
| `gap_open` | `float` | Gap-open penalty (affine only; ignored for linear) |
| `gap_extend` | `float` | Gap-extension penalty |
| `gap_model` | `str` | `"linear"` or `"affine"` |
| `mode` | `str` | `"global"` (NW) or `"local"` (SW) |
| `grad_mode` | `str` | `"hard"`, `"soft"`, or `"none"` |
| `n_threads` | `int` | Number of worker threads |

**`.align(sequences_a, sequences_b) -> BatchResult`**

Aligns each pair `(sequences_a[i], sequences_b[i])`. Both lists must have the same length.

---

### `BatchResult`

| Attribute | Type | Description |
|---|---|---|
| `scores` | `np.ndarray[N]` float64 | Alignment score (or log-partition for soft) per pair |
| `grad` | `np.ndarray[20, 20]` float64 | Gradient summed over all pairs |

## Gap penalty conventions

For **linear** gap model, a gap of length `k` costs `gap_extend * k`.

For **affine** gap model, a gap of length `k` costs `gap_open + gap_extend * k`.

This matches BioPython's convention where `open` is charged once per gap regardless of length.

## Performance notes

- Full O(m×n) DP tables are retained for gradient computation — no Hirschberg-style memory optimisation.
- Each worker thread allocates its DP buffers once (sized to the largest sequence in the batch) and reuses them across all assigned pairs.
- Work is distributed via a shared `std::atomic` counter — no per-task mutex, no work-stealing queue.
- Gradient accumulation is per-thread; a single mutex is taken once at join to merge partial gradients.

## C++ header-only library

The C++ implementation is header-only (`src/nwgrad/cpp/nwgrad/`). To use it directly from C++:

```bash
python -m nwgrad --include
# prints: /path/to/nwgrad/cpp
```

Add that path to your include path and `#include "nwgrad/aligner.hpp"` etc.

## Building from source

```bash
pip install scikit-build-core nanobind
pip install -e ".[dev]"
pytest tests/Python/
```

C++ unit tests (requires CMake):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
ctest --test-dir build
```

## License

MIT — see [LICENSE](LICENSE).
