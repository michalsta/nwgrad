# Tutorial 2 — Batch Alignment and Multithreading

This tutorial covers `BatchAligner`: aligning many sequence pairs at once using a
thread pool, and collecting scores and gradients efficiently.

## Why batch?

Calling the single-pair functions in a Python loop works but misses two
opportunities:

1. **Thread-level parallelism** — nwgrad's thread pool distributes pairs across
   CPU cores via a lock-free work queue.
2. **Memory reuse** — each worker thread allocates its DP buffers once (sized to
   the largest sequence in the batch) and reuses them for all assigned pairs.

## Basic usage

```python
import numpy as np
import nwgrad

# ... construct blosum62 as in Tutorial 1 ...

seqs_a = ["PLEASANTLY", "ACDEFGHIKL", "MADEEKLF", "ACDE"]
seqs_b = ["MEANLY",     "CDEFGHIKLM", "MADEEKLF", "ACDF"]

aligner = nwgrad.BatchAligner(
    matrix=blosum62,
    gap_open=11.0,
    gap_extend=1.0,
    gap_model="affine",   # "linear" | "affine"
    mode="global",        # "global" | "local"
    grad_mode="hard",     # "hard" | "soft" | "none"
    n_threads=4,
)

result = aligner.align(seqs_a, seqs_b)

print(result.scores)   # np.ndarray float64, shape (4,)
print(result.grad)     # np.ndarray float64, shape (20, 20)
```

`result.scores[i]` is the alignment score for pair `i`.  
`result.grad` is the gradient **summed over all pairs** in the batch.

## All four alignment modes

`BatchAligner` supports all combinations of gap model and alignment mode:

```python
configs = [
    dict(gap_model="linear",  mode="global", gap_extend=1.0),
    dict(gap_model="linear",  mode="local",  gap_extend=1.0),
    dict(gap_model="affine",  mode="global", gap_open=11.0, gap_extend=1.0),
    dict(gap_model="affine",  mode="local",  gap_open=11.0, gap_extend=1.0),
]

for cfg in configs:
    aligner = nwgrad.BatchAligner(matrix=blosum62, grad_mode="hard",
                                  n_threads=4, **cfg)
    result = aligner.align(seqs_a, seqs_b)
    print(f"{cfg['gap_model']:6s} {cfg['mode']:6s}  scores={result.scores}")
```

## Gradient modes

| `grad_mode` | `result.scores[i]` | `result.grad` |
|---|---|---|
| `"hard"` | alignment score | substitution-pair counts (integer-valued) |
| `"soft"` | log-partition function `log Z` | expected substitution counts (fractional) |
| `"none"` | alignment score | zero matrix |

Use `"none"` when you only need scores — it skips the traceback/backward pass.

```python
# Score-only, no gradient computation
score_only = nwgrad.BatchAligner(
    matrix=blosum62, gap_model="affine", mode="global",
    grad_mode="none", n_threads=8,
)
result = score_only.align(seqs_a, seqs_b)
assert result.grad.sum() == 0.0
```

## Thread safety and reproducibility

Results are identical regardless of thread count — the per-thread gradients are
merged under a mutex at join time, and floating-point addition order is
deterministic within each thread.

```python
ref = nwgrad.BatchAligner(
    matrix=blosum62, gap_model="affine", mode="global",
    grad_mode="hard", n_threads=1,
).align(seqs_a, seqs_b)

for n in [2, 4, 8]:
    got = nwgrad.BatchAligner(
        matrix=blosum62, gap_model="affine", mode="global",
        grad_mode="hard", n_threads=n,
    ).align(seqs_a, seqs_b)
    np.testing.assert_allclose(got.scores, ref.scores, atol=1e-10)
    np.testing.assert_allclose(got.grad,   ref.grad,   atol=1e-10)
```

## Aligning all pairs in a dataset

Batch input is a flat list of arbitrary sequence pairs — not necessarily an
all-vs-all matrix. To align every sequence against every other:

```python
sequences = ["ACDEFG", "MADEEKLF", "PLEASANTLY", "ACDE", "CDEFGHIKLM"]

# Build all-vs-all pairs (upper triangle only, excluding self)
pairs_a, pairs_b = [], []
for i in range(len(sequences)):
    for j in range(i + 1, len(sequences)):
        pairs_a.append(sequences[i])
        pairs_b.append(sequences[j])

aligner = nwgrad.BatchAligner(
    matrix=blosum62, gap_open=11.0, gap_extend=1.0,
    gap_model="affine", mode="global", grad_mode="hard", n_threads=4,
)
result = aligner.align(pairs_a, pairs_b)

# result.scores[k] is the score for pairs_a[k] vs pairs_b[k]
# result.grad is the gradient summed over all pairs
```

## Large batches with random sequences

```python
rng = np.random.default_rng(42)
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"

def random_seqs(n, min_len=10, max_len=200):
    alphabet = list(AA_ORDER)
    return [
        "".join(rng.choice(alphabet, size=int(rng.integers(min_len, max_len))).tolist())
        for _ in range(n)
    ]

N = 10_000
seqs_a = random_seqs(N)
seqs_b = random_seqs(N)

aligner = nwgrad.BatchAligner(
    matrix=blosum62, gap_open=11.0, gap_extend=1.0,
    gap_model="affine", mode="global", grad_mode="hard",
    n_threads=8,
)
result = aligner.align(seqs_a, seqs_b)
print(f"Mean score: {result.scores.mean():.2f}")
print(f"Total aligned pairs: {N}")
```

## Matching single-pair results

The batch result is numerically identical to summing individual single-pair calls:

```python
# Batch result
batch_result = nwgrad.BatchAligner(
    matrix=blosum62, gap_open=11.0, gap_extend=1.0,
    gap_model="affine", mode="global", grad_mode="hard", n_threads=1,
).align(seqs_a, seqs_b)

# Single-pair sum
expected_grad = np.zeros((20, 20))
expected_scores = []
for a, b in zip(seqs_a, seqs_b):
    s, g = nwgrad.nw_affine_grad(a, b, blosum62, gap_open=11.0, gap_extend=1.0)
    expected_scores.append(s)
    expected_grad += g

np.testing.assert_allclose(batch_result.scores, expected_scores, atol=1e-10)
np.testing.assert_allclose(batch_result.grad,   expected_grad,   atol=1e-10)
```

Next: [Tutorial 3 — Gradient-based substitution matrix optimisation](03_matrix_optimization.md)
