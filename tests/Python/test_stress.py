"""Large stress tests for BatchAligner with multithreading.

Covers:
- Large random batches (default 1 000–5 000 pairs, sequences up to 500 aa)
- All four gap_model × mode combinations
- Configurable thread counts (default 2, 4, 8, 16)
- Score and gradient determinism: identical results regardless of thread count
- Python-level concurrent calls (multiple Python threads each calling align)
- Gradient accumulation correctness at scale: batch == sum of singles
- Soft vs hard gradient monotone relationship at scale
- Memory-pressure pairs (very long sequences, up to 1 000 aa by default)
- Reproducibility: same RNG seed always produces the same output

Dual-mode: runs under pytest or directly from the command line.

    pytest tests/Python/test_stress.py
    python tests/Python/test_stress.py
    python tests/Python/test_stress.py --n-large 500 --thread-counts 2,4,8
    python tests/Python/test_stress.py --help
"""

import argparse
import dataclasses
import sys
import threading
import time
from typing import List, Tuple

import numpy as np
import pytest
import nwgrad
from test_subst_matrix import BLOSUM62

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"


# ── parameters ─────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class Params:
    seed: int = 0xDEADBEEF

    # "large" batch — score/grad determinism, soft tests, integer-grad check
    n_large: int = 1000
    large_lo: int = 10
    large_hi: int = 100

    # "XL" batch — throughput smoke test
    n_xl: int = 5000
    xl_lo: int = 5
    xl_hi: int = 50

    # long-sequence batch — memory pressure
    n_long: int = 200
    long_lo: int = 200
    long_hi: int = 500

    # very-long-sequence batch — crash/OOM check
    n_vlong: int = 50
    vlong_lo: int = 500
    vlong_hi: int = 1000

    # grad-vs-singles correctness subset
    n_grad_singles: int = 100
    grad_singles_lo: int = 10
    grad_singles_hi: int = 60

    # thread counts probed in determinism tests
    thread_counts: Tuple[int, ...] = (2, 4, 8, 16)

    # Python-level concurrent callers
    n_concurrent_score: int = 8    # threads for concurrent-score test
    n_concurrent_grad: int = 6     # threads for concurrent-grad test

    # aligner defaults
    gap_open: float = 11.0
    gap_extend: float = 1.0

    # throughput limit (seconds) for XL batch
    throughput_limit: float = 60.0


# Module-level default — pytest fixtures and parametrize decorators read this.
PARAMS = Params()


# ── helpers ────────────────────────────────────────────────────────────────────

def make_blosum() -> nwgrad.SubstMatrix:
    return nwgrad.SubstMatrix(np.asarray(BLOSUM62, dtype=np.float64))


def random_seqs(rng: np.random.Generator, n: int, lo: int, hi: int) -> List[str]:
    """Return n random amino-acid sequences, each length in [lo, hi]."""
    alphabet = list(AA_ORDER)
    return [
        "".join(rng.choice(alphabet, size=int(rng.integers(lo, hi + 1))).tolist())
        for _ in range(n)
    ]


def _ref(blosum, gap_model="affine", mode="global", gap_open=11.0, gap_extend=1.0,
         grad_mode="hard"):
    return nwgrad.BatchAligner(matrix=blosum, gap_open=gap_open, gap_extend=gap_extend,
                               gap_model=gap_model, mode=mode,
                               grad_mode=grad_mode, n_threads=1)


def _multi(blosum, n_threads, gap_model="affine", mode="global",
           gap_open=11.0, gap_extend=1.0, grad_mode="hard"):
    return nwgrad.BatchAligner(matrix=blosum, gap_open=gap_open, gap_extend=gap_extend,
                               gap_model=gap_model, mode=mode,
                               grad_mode=grad_mode, n_threads=n_threads)


# ── check_* functions (no pytest deps, called by both wrappers and CLI) ────────

def check_score_determinism(seqs_a, seqs_b, blosum, n_threads, p: Params = PARAMS,
                             gap_model="affine", mode="global"):
    go, ge = (p.gap_open, p.gap_extend) if gap_model == "affine" else (0.0, p.gap_extend)
    ref = _ref(blosum, gap_model, mode, go, ge, grad_mode="none").align(seqs_a, seqs_b)
    got = _multi(blosum, n_threads, gap_model, mode, go, ge, grad_mode="none").align(seqs_a, seqs_b)
    np.testing.assert_array_equal(
        np.array(got.scores), np.array(ref.scores),
        err_msg=f"scores differ: {gap_model}/{mode} n_threads={n_threads}",
    )


def check_grad_determinism(seqs_a, seqs_b, blosum, n_threads, p: Params = PARAMS):
    go, ge = p.gap_open, p.gap_extend
    ref = _ref(blosum, gap_open=go, gap_extend=ge, grad_mode="hard").align(seqs_a, seqs_b)
    got = _multi(blosum, n_threads, gap_open=go, gap_extend=ge, grad_mode="hard").align(seqs_a, seqs_b)
    np.testing.assert_array_equal(
        np.array(got.grad), np.array(ref.grad),
        err_msg=f"gradient differs at n_threads={n_threads}",
    )


def check_batch_grad_equals_sum_of_singles(seqs_a, seqs_b, blosum,
                                           gap_model, mode, gap_open, gap_extend, grad_fn,
                                           p: Params = PARAMS):
    expected = np.zeros((20, 20))
    for a, b in zip(seqs_a, seqs_b):
        if gap_model == "linear":
            _, g = grad_fn(a, b, blosum, gap_extend)
        else:
            _, g = grad_fn(a, b, blosum, gap_open, gap_extend)
        expected += np.array(g)

    ba = nwgrad.BatchAligner(matrix=blosum, gap_open=gap_open, gap_extend=gap_extend,
                             gap_model=gap_model, mode=mode, grad_mode="hard", n_threads=4)
    result = ba.align(seqs_a, seqs_b)
    np.testing.assert_allclose(np.array(result.grad), expected, atol=1e-10,
                                err_msg=f"grad mismatch: {gap_model}/{mode}")


def check_concurrent_scores(seqs_a, seqs_b, blosum, n_concurrent: int, p: Params = PARAMS):
    aligner = nwgrad.BatchAligner(matrix=blosum, gap_open=p.gap_open, gap_extend=p.gap_extend,
                                   gap_model="affine", mode="global",
                                   grad_mode="none", n_threads=2)
    ref_scores = np.array(aligner.align(seqs_a, seqs_b).scores)

    results = [None] * n_concurrent
    errors: list = []

    def run(idx):
        try:
            results[idx] = np.array(aligner.align(seqs_a, seqs_b).scores)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(n_concurrent)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"exceptions in concurrent threads: {errors}"
    for i, scores in enumerate(results):
        np.testing.assert_array_equal(scores, ref_scores,
                                       err_msg=f"thread {i} scores diverged")


def check_concurrent_gradients(seqs_a, seqs_b, blosum, n_concurrent: int, p: Params = PARAMS):
    aligner = nwgrad.BatchAligner(matrix=blosum, gap_open=p.gap_open, gap_extend=p.gap_extend,
                                   gap_model="affine", mode="global",
                                   grad_mode="hard", n_threads=4)
    ref = aligner.align(seqs_a, seqs_b)
    ref_scores = np.array(ref.scores)
    ref_grad = np.array(ref.grad)

    score_results = [None] * n_concurrent
    grad_results = [None] * n_concurrent
    errors: list = []

    def run(idx):
        try:
            r = aligner.align(seqs_a, seqs_b)
            score_results[idx] = np.array(r.scores)
            grad_results[idx] = np.array(r.grad)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(n_concurrent)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"exceptions in concurrent gradient threads: {errors}"
    for i in range(n_concurrent):
        np.testing.assert_array_equal(score_results[i], ref_scores,
                                       err_msg=f"concurrent thread {i}: scores differ")
        np.testing.assert_array_equal(grad_results[i], ref_grad,
                                       err_msg=f"concurrent thread {i}: gradient differs")


def check_long_seq_determinism(seqs_a, seqs_b, blosum, n_threads, p: Params = PARAMS):
    go, ge = p.gap_open, p.gap_extend
    ref = _ref(blosum, gap_open=go, gap_extend=ge, grad_mode="hard").align(seqs_a, seqs_b)
    got = _multi(blosum, n_threads, gap_open=go, gap_extend=ge, grad_mode="hard").align(seqs_a, seqs_b)
    np.testing.assert_array_equal(np.array(got.scores), np.array(ref.scores),
                                   err_msg=f"long-seq scores differ at n_threads={n_threads}")
    np.testing.assert_array_equal(np.array(got.grad), np.array(ref.grad),
                                   err_msg=f"long-seq gradient differs at n_threads={n_threads}")


def check_very_long_no_crash(seqs_a, seqs_b, blosum, p: Params = PARAMS):
    result = nwgrad.BatchAligner(matrix=blosum, gap_open=p.gap_open, gap_extend=p.gap_extend,
                                  gap_model="affine", mode="global",
                                  grad_mode="hard", n_threads=4).align(seqs_a, seqs_b)
    scores = np.array(result.scores)
    assert scores.shape == (len(seqs_a),), f"wrong score shape: {scores.shape}"
    assert np.all(np.isfinite(scores)), "non-finite scores in very-long batch"
    assert np.array(result.grad).shape == (20, 20)


def check_soft_grad_determinism(seqs_a, seqs_b, blosum, n_threads, p: Params = PARAMS):
    go, ge = p.gap_open, p.gap_extend
    ref = nwgrad.BatchAligner(matrix=blosum, gap_open=go, gap_extend=ge,
                               gap_model="affine", mode="global",
                               grad_mode="soft", n_threads=1).align(seqs_a, seqs_b)
    got = nwgrad.BatchAligner(matrix=blosum, gap_open=go, gap_extend=ge,
                               gap_model="affine", mode="global",
                               grad_mode="soft", n_threads=n_threads).align(seqs_a, seqs_b)
    np.testing.assert_allclose(np.array(got.scores), np.array(ref.scores), atol=1e-10,
                                err_msg=f"soft scores differ at n_threads={n_threads}")
    np.testing.assert_allclose(np.array(got.grad), np.array(ref.grad), atol=1e-10,
                                err_msg=f"soft gradient differs at n_threads={n_threads}")


def check_soft_geq_hard(seqs_a, seqs_b, blosum, p: Params = PARAMS):
    go, ge = p.gap_open, p.gap_extend
    hard_res = nwgrad.BatchAligner(matrix=blosum, gap_open=go, gap_extend=ge,
                                    gap_model="affine", mode="global",
                                    grad_mode="none", n_threads=4).align(seqs_a, seqs_b)
    soft_res = nwgrad.BatchAligner(matrix=blosum, gap_open=go, gap_extend=ge,
                                    gap_model="affine", mode="global",
                                    grad_mode="soft", n_threads=4).align(seqs_a, seqs_b)
    hard_scores = np.array(hard_res.scores)
    soft_scores = np.array(soft_res.scores)
    violations = np.where(soft_scores < hard_scores - 1e-9)[0]
    assert len(violations) == 0, (
        f"soft_score < hard_score for {len(violations)} pairs; "
        f"worst gap: {(hard_scores - soft_scores)[violations].max():.6g}"
    )


def check_hard_grad_integer(seqs_a, seqs_b, blosum, p: Params = PARAMS):
    go, ge = p.gap_open, p.gap_extend
    result = nwgrad.BatchAligner(matrix=blosum, gap_open=go, gap_extend=ge,
                                  gap_model="affine", mode="global",
                                  grad_mode="hard", n_threads=4).align(seqs_a, seqs_b)
    g = np.array(result.grad)
    assert (g >= 0).all(), "negative entries in hard gradient"
    assert np.allclose(g, np.round(g), atol=1e-9), "hard gradient is not integer-valued"


def check_reproducibility(blosum, p: Params = PARAMS):
    def run():
        rng = np.random.default_rng(12345)
        seqs_a = random_seqs(rng, 300, p.large_lo, p.large_hi)
        seqs_b = random_seqs(rng, 300, p.large_lo, p.large_hi)
        return nwgrad.BatchAligner(matrix=blosum, gap_open=p.gap_open, gap_extend=p.gap_extend,
                                    gap_model="affine", mode="global",
                                    grad_mode="hard", n_threads=4).align(seqs_a, seqs_b)

    r1, r2 = run(), run()
    np.testing.assert_array_equal(np.array(r1.scores), np.array(r2.scores))
    np.testing.assert_array_equal(np.array(r1.grad),   np.array(r2.grad))


def check_more_threads_than_pairs(blosum, p: Params = PARAMS):
    rng = np.random.default_rng(p.seed + 7)
    seqs_a = random_seqs(rng, 3, p.large_lo, p.large_hi)
    seqs_b = random_seqs(rng, 3, p.large_lo, p.large_hi)
    go, ge = p.gap_open, p.gap_extend
    ref = _ref(blosum, gap_open=go, gap_extend=ge, grad_mode="hard").align(seqs_a, seqs_b)
    got = _multi(blosum, 16, gap_open=go, gap_extend=ge, grad_mode="hard").align(seqs_a, seqs_b)
    np.testing.assert_array_equal(np.array(got.scores), np.array(ref.scores))
    np.testing.assert_array_equal(np.array(got.grad),   np.array(ref.grad))


def check_throughput(seqs_a, seqs_b, blosum, limit: float, p: Params = PARAMS):
    aligner = nwgrad.BatchAligner(matrix=blosum, gap_open=p.gap_open, gap_extend=p.gap_extend,
                                   gap_model="affine", mode="global",
                                   grad_mode="hard", n_threads=8)
    t0 = time.monotonic()
    result = aligner.align(seqs_a, seqs_b)
    elapsed = time.monotonic() - t0
    assert np.array(result.scores).shape == (len(seqs_a),)
    assert elapsed < limit, f"{len(seqs_a)}-pair batch took {elapsed:.1f}s — too slow (limit {limit}s)"
    return elapsed


def check_grad_symmetry_symmetric_input(seqs, blosum, grad_mode: str, p: Params = PARAMS):
    go, ge = p.gap_open, p.gap_extend
    result = nwgrad.BatchAligner(matrix=blosum, gap_open=go, gap_extend=ge,
                                  gap_model="affine", mode="global",
                                  grad_mode=grad_mode, n_threads=4).align(seqs, seqs)
    g = np.array(result.grad)
    np.testing.assert_allclose(g, g.T, atol=1e-10,
                                err_msg=f"{grad_mode} gradient not symmetric on symmetric input")


def check_concurrent_different_aligners(seqs_a, seqs_b, blosum, p: Params = PARAMS):
    go, ge = p.gap_open, p.gap_extend
    configs = [
        dict(gap_model="affine",  mode="global", gap_open=go,  gap_extend=ge,
             grad_mode="hard",  n_threads=2),
        dict(gap_model="affine",  mode="local",  gap_open=go,  gap_extend=ge,
             grad_mode="none",  n_threads=2),
        dict(gap_model="linear", mode="global", gap_open=0.0, gap_extend=ge,
             grad_mode="hard",  n_threads=2),
        dict(gap_model="linear", mode="local",  gap_open=0.0, gap_extend=ge,
             grad_mode="soft",  n_threads=2),
    ]
    refs = [np.array(nwgrad.BatchAligner(matrix=blosum, **cfg).align(seqs_a, seqs_b).scores)
            for cfg in configs]

    results = [None] * len(configs)
    errors: list = []

    def run(idx, cfg):
        try:
            results[idx] = np.array(
                nwgrad.BatchAligner(matrix=blosum, **cfg).align(seqs_a, seqs_b).scores
            )
        except Exception as e:
            errors.append((idx, e))

    threads = [threading.Thread(target=run, args=(i, cfg)) for i, cfg in enumerate(configs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"exceptions in concurrent-aligner test: {errors}"
    for i, (ref, got) in enumerate(zip(refs, results)):
        np.testing.assert_array_equal(got, ref,
                                       err_msg=f"concurrent aligner {i} scores differ")


# ── pytest fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def blosum():
    return make_blosum()


@pytest.fixture(scope="module")
def large_batch():
    rng = np.random.default_rng(PARAMS.seed)
    return (random_seqs(rng, PARAMS.n_large, PARAMS.large_lo, PARAMS.large_hi),
            random_seqs(rng, PARAMS.n_large, PARAMS.large_lo, PARAMS.large_hi))


@pytest.fixture(scope="module")
def xl_batch():
    rng = np.random.default_rng(PARAMS.seed + 1)
    return (random_seqs(rng, PARAMS.n_xl, PARAMS.xl_lo, PARAMS.xl_hi),
            random_seqs(rng, PARAMS.n_xl, PARAMS.xl_lo, PARAMS.xl_hi))


@pytest.fixture(scope="module")
def long_batch():
    rng = np.random.default_rng(PARAMS.seed + 2)
    return (random_seqs(rng, PARAMS.n_long, PARAMS.long_lo, PARAMS.long_hi),
            random_seqs(rng, PARAMS.n_long, PARAMS.long_lo, PARAMS.long_hi))


@pytest.fixture(scope="module")
def vlong_batch():
    rng = np.random.default_rng(PARAMS.seed + 3)
    return (random_seqs(rng, PARAMS.n_vlong, PARAMS.vlong_lo, PARAMS.vlong_hi),
            random_seqs(rng, PARAMS.n_vlong, PARAMS.vlong_lo, PARAMS.vlong_hi))


@pytest.fixture(scope="module")
def grad_singles_batch():
    rng = np.random.default_rng(PARAMS.seed + 10)
    return (random_seqs(rng, PARAMS.n_grad_singles, PARAMS.grad_singles_lo, PARAMS.grad_singles_hi),
            random_seqs(rng, PARAMS.n_grad_singles, PARAMS.grad_singles_lo, PARAMS.grad_singles_hi))


# ── pytest test_* wrappers ──────────────────────────────────────────────────────

# 1. Score determinism — large batch
@pytest.mark.parametrize("n_threads", PARAMS.thread_counts)
def test_score_determinism_large_batch(n_threads, blosum, large_batch):
    check_score_determinism(*large_batch, blosum, n_threads)


# 2. Score determinism — XL batch
@pytest.mark.parametrize("n_threads", PARAMS.thread_counts)
def test_score_determinism_xl_batch(n_threads, blosum, xl_batch):
    check_score_determinism(*xl_batch, blosum, n_threads)


# 3. Gradient determinism — large batch
@pytest.mark.parametrize("n_threads", PARAMS.thread_counts)
def test_grad_determinism_large_batch(n_threads, blosum, large_batch):
    check_grad_determinism(*large_batch, blosum, n_threads)


# 4. All four gap_model × mode combinations
@pytest.mark.parametrize("gap_model,mode,gap_open,gap_extend", [
    ("linear", "global", 0.0,  1.0),
    ("linear", "local",  0.0,  1.0),
    ("affine", "global", 11.0, 1.0),
    ("affine", "local",  11.0, 1.0),
])
@pytest.mark.parametrize("n_threads", [4, 8])
def test_all_modes_score_determinism(n_threads, gap_model, mode, gap_open, gap_extend,
                                     blosum, large_batch):
    check_score_determinism(*large_batch, blosum, n_threads, gap_model=gap_model, mode=mode)


# 5. Gradient == sum of singles
@pytest.mark.parametrize("gap_model,mode,gap_open,gap_extend,grad_fn", [
    ("linear", "global", 0.0,  1.0, nwgrad.nw_grad),
    ("affine", "global", 11.0, 1.0, nwgrad.nw_affine_grad),
    ("affine", "local",  11.0, 1.0, nwgrad.sw_affine_grad),
])
def test_batch_grad_equals_sum_of_singles(gap_model, mode, gap_open, gap_extend,
                                          grad_fn, blosum, grad_singles_batch):
    check_batch_grad_equals_sum_of_singles(
        *grad_singles_batch, blosum, gap_model, mode, gap_open, gap_extend, grad_fn)


# 6. Python-level concurrent callers
def test_concurrent_python_threads_scores(blosum, large_batch):
    check_concurrent_scores(*large_batch, blosum, PARAMS.n_concurrent_score)


def test_concurrent_python_threads_gradients(blosum):
    rng = np.random.default_rng(PARAMS.seed + 99)
    seqs_a = random_seqs(rng, 500, PARAMS.large_lo, PARAMS.large_hi)
    seqs_b = random_seqs(rng, 500, PARAMS.large_lo, PARAMS.large_hi)
    check_concurrent_gradients(seqs_a, seqs_b, blosum, PARAMS.n_concurrent_grad)


# 7. Long sequences
@pytest.mark.parametrize("n_threads", [4, 8])
def test_long_sequences_score_determinism(n_threads, blosum, long_batch):
    check_long_seq_determinism(*long_batch, blosum, n_threads)


def test_very_long_sequences_no_crash(blosum, vlong_batch):
    check_very_long_no_crash(*vlong_batch, blosum)


# 8. Soft gradient determinism
@pytest.mark.parametrize("n_threads", [2, 4, 8])
def test_soft_grad_determinism_large_batch(n_threads, blosum, large_batch):
    check_soft_grad_determinism(*large_batch, blosum, n_threads)


# 9. soft >= hard score
def test_soft_score_geq_hard_score_large_batch(blosum, large_batch):
    check_soft_geq_hard(*large_batch, blosum)


# 10. Hard gradient non-negative and integer-valued
def test_hard_grad_integer_valued_large_batch(blosum, large_batch):
    check_hard_grad_integer(*large_batch, blosum)


# 11. Reproducibility
def test_reproducibility(blosum):
    check_reproducibility(blosum)


# 12. n_threads > batch_size
def test_more_threads_than_pairs(blosum):
    check_more_threads_than_pairs(blosum)


# 13. Throughput smoke test
def test_throughput_does_not_regress(blosum, xl_batch):
    check_throughput(*xl_batch, blosum, PARAMS.throughput_limit)


# 14. Gradient symmetry on symmetric input
def test_hard_grad_symmetry_symmetric_input(blosum):
    rng = np.random.default_rng(PARAMS.seed + 100)
    seqs = random_seqs(rng, 300, PARAMS.large_lo, PARAMS.large_hi)
    check_grad_symmetry_symmetric_input(seqs, blosum, "hard")


def test_soft_grad_symmetry_symmetric_input(blosum):
    rng = np.random.default_rng(PARAMS.seed + 101)
    seqs = random_seqs(rng, 200, PARAMS.large_lo, PARAMS.large_hi)
    check_grad_symmetry_symmetric_input(seqs, blosum, "soft")


# 15. Concurrent different aligners
def test_concurrent_different_aligners(blosum):
    rng = np.random.default_rng(PARAMS.seed + 55)
    seqs_a = random_seqs(rng, 200, PARAMS.large_lo, PARAMS.large_hi)
    seqs_b = random_seqs(rng, 200, PARAMS.large_lo, PARAMS.large_hi)
    check_concurrent_different_aligners(seqs_a, seqs_b, blosum)


# ── CLI runner ──────────────────────────────────────────────────────────────────

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="nwgrad BatchAligner stress tests (pytest or standalone CLI)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--seed", type=lambda x: int(x, 0), default=PARAMS.seed,
                   help="RNG seed (hex or decimal)")
    p.add_argument("--n-large", type=int, default=PARAMS.n_large,
                   metavar="N", help="pairs in the 'large' batch")
    p.add_argument("--large-len", type=int, nargs=2, default=[PARAMS.large_lo, PARAMS.large_hi],
                   metavar=("LO", "HI"), help="sequence length range for large batch")
    p.add_argument("--n-xl", type=int, default=PARAMS.n_xl,
                   metavar="N", help="pairs in the XL (throughput) batch")
    p.add_argument("--xl-len", type=int, nargs=2, default=[PARAMS.xl_lo, PARAMS.xl_hi],
                   metavar=("LO", "HI"), help="sequence length range for XL batch")
    p.add_argument("--n-long", type=int, default=PARAMS.n_long,
                   metavar="N", help="pairs in the long-sequence batch")
    p.add_argument("--long-len", type=int, nargs=2, default=[PARAMS.long_lo, PARAMS.long_hi],
                   metavar=("LO", "HI"), help="sequence length range for long batch")
    p.add_argument("--n-vlong", type=int, default=PARAMS.n_vlong,
                   metavar="N", help="pairs in the very-long-sequence batch")
    p.add_argument("--vlong-len", type=int, nargs=2, default=[PARAMS.vlong_lo, PARAMS.vlong_hi],
                   metavar=("LO", "HI"), help="sequence length range for very-long batch")
    p.add_argument("--n-grad-singles", type=int, default=PARAMS.n_grad_singles,
                   metavar="N", help="pairs used in gradient-vs-singles check")
    p.add_argument("--thread-counts", type=lambda s: tuple(int(x) for x in s.split(",")),
                   default=PARAMS.thread_counts,
                   metavar="N,N,...", help="thread counts to probe in determinism tests")
    p.add_argument("--n-concurrent-score", type=int, default=PARAMS.n_concurrent_score,
                   metavar="N", help="Python threads for concurrent-score test")
    p.add_argument("--n-concurrent-grad", type=int, default=PARAMS.n_concurrent_grad,
                   metavar="N", help="Python threads for concurrent-gradient test")
    p.add_argument("--gap-open", type=float, default=PARAMS.gap_open)
    p.add_argument("--gap-extend", type=float, default=PARAMS.gap_extend)
    p.add_argument("--throughput-limit", type=float, default=PARAMS.throughput_limit,
                   metavar="SECS", help="max allowed wall time for throughput test")
    p.add_argument("--test", metavar="NAME", default="all",
                   help=("test group to run: all | score_determinism | grad_determinism | "
                         "all_modes | grad_singles | concurrent_scores | concurrent_grads | "
                         "long_seqs | soft | integer_grad | reproducibility | "
                         "more_threads | throughput | symmetry | concurrent_aligners"))
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


def _params_from_args(args) -> Params:
    return Params(
        seed=args.seed,
        n_large=args.n_large,
        large_lo=args.large_len[0],
        large_hi=args.large_len[1],
        n_xl=args.n_xl,
        xl_lo=args.xl_len[0],
        xl_hi=args.xl_len[1],
        n_long=args.n_long,
        long_lo=args.long_len[0],
        long_hi=args.long_len[1],
        n_vlong=args.n_vlong,
        vlong_lo=args.vlong_len[0],
        vlong_hi=args.vlong_len[1],
        n_grad_singles=args.n_grad_singles,
        thread_counts=args.thread_counts,
        n_concurrent_score=args.n_concurrent_score,
        n_concurrent_grad=args.n_concurrent_grad,
        gap_open=args.gap_open,
        gap_extend=args.gap_extend,
        throughput_limit=args.throughput_limit,
    )


def _cli_run(name: str, fn, results: list, verbose: bool):
    t0 = time.monotonic()
    try:
        fn()
        elapsed = time.monotonic() - t0
        results.append((name, True, elapsed, None))
        print(f"  PASS  {name}  ({elapsed:.2f}s)")
    except Exception as exc:
        elapsed = time.monotonic() - t0
        results.append((name, False, elapsed, exc))
        print(f"  FAIL  {name}  ({elapsed:.2f}s)")
        if verbose:
            import traceback
            traceback.print_exc()
        else:
            print(f"        {exc}")


def main(argv=None):
    args = _parse_args(argv)
    p = _params_from_args(args)
    want = args.test
    v = args.verbose

    blosum = make_blosum()

    # Pre-generate all batches once.
    print(f"Generating batches (seed={p.seed:#x}) …")
    rng = np.random.default_rng(p.seed)
    large_a  = random_seqs(rng, p.n_large,       p.large_lo,       p.large_hi)
    large_b  = random_seqs(rng, p.n_large,       p.large_lo,       p.large_hi)
    xl_a     = random_seqs(rng, p.n_xl,          p.xl_lo,          p.xl_hi)
    xl_b     = random_seqs(rng, p.n_xl,          p.xl_lo,          p.xl_hi)
    long_a   = random_seqs(rng, p.n_long,        p.long_lo,        p.long_hi)
    long_b   = random_seqs(rng, p.n_long,        p.long_lo,        p.long_hi)
    vlong_a  = random_seqs(rng, p.n_vlong,       p.vlong_lo,       p.vlong_hi)
    vlong_b  = random_seqs(rng, p.n_vlong,       p.vlong_lo,       p.vlong_hi)
    gs_a     = random_seqs(rng, p.n_grad_singles, p.grad_singles_lo, p.grad_singles_hi)
    gs_b     = random_seqs(rng, p.n_grad_singles, p.grad_singles_lo, p.grad_singles_hi)
    sym_seqs = random_seqs(rng, 300,             p.large_lo,       p.large_hi)
    cd_a     = random_seqs(rng, 200,             p.large_lo,       p.large_hi)
    cd_b     = random_seqs(rng, 200,             p.large_lo,       p.large_hi)
    conc_a   = random_seqs(rng, 500,             p.large_lo,       p.large_hi)
    conc_b   = random_seqs(rng, 500,             p.large_lo,       p.large_hi)
    print("Done.\n")

    results = []
    r = results

    def run(name, fn):
        _cli_run(name, fn, r, v)

    def do(group):
        return want in ("all", group)

    if do("score_determinism"):
        print("── Score determinism (large batch) ──────────────────────────────")
        for nt in p.thread_counts:
            run(f"score_determinism_large[n_threads={nt}]",
                lambda nt=nt: check_score_determinism(large_a, large_b, blosum, nt, p))
        print("── Score determinism (XL batch) ─────────────────────────────────")
        for nt in p.thread_counts:
            run(f"score_determinism_xl[n_threads={nt}]",
                lambda nt=nt: check_score_determinism(xl_a, xl_b, blosum, nt, p))

    if do("grad_determinism"):
        print("── Gradient determinism (large batch) ───────────────────────────")
        for nt in p.thread_counts:
            run(f"grad_determinism_large[n_threads={nt}]",
                lambda nt=nt: check_grad_determinism(large_a, large_b, blosum, nt, p))

    if do("all_modes"):
        print("── All gap_model × mode combinations ────────────────────────────")
        combos = [
            ("linear", "global", 0.0,  p.gap_extend),
            ("linear", "local",  0.0,  p.gap_extend),
            ("affine", "global", p.gap_open, p.gap_extend),
            ("affine", "local",  p.gap_open, p.gap_extend),
        ]
        for nt in [4, 8]:
            for gm, mode, go, ge in combos:
                run(f"all_modes[{gm}/{mode},n_threads={nt}]",
                    lambda nt=nt, gm=gm, mode=mode: check_score_determinism(
                        large_a, large_b, blosum, nt, p, gap_model=gm, mode=mode))

    if do("grad_singles"):
        print("── Gradient == sum of singles ────────────────────────────────────")
        singles_combos = [
            ("linear", "global", 0.0,  p.gap_extend, nwgrad.nw_grad),
            ("affine", "global", p.gap_open, p.gap_extend, nwgrad.nw_affine_grad),
            ("affine", "local",  p.gap_open, p.gap_extend, nwgrad.sw_affine_grad),
        ]
        for gm, mode, go, ge, fn in singles_combos:
            run(f"grad_singles[{gm}/{mode}]",
                lambda gm=gm, mode=mode, go=go, ge=ge, fn=fn:
                    check_batch_grad_equals_sum_of_singles(gs_a, gs_b, blosum, gm, mode, go, ge, fn, p))

    if do("concurrent_scores"):
        print("── Concurrent Python threads (scores) ───────────────────────────")
        run("concurrent_scores",
            lambda: check_concurrent_scores(large_a, large_b, blosum, p.n_concurrent_score, p))

    if do("concurrent_grads"):
        print("── Concurrent Python threads (gradients) ────────────────────────")
        run("concurrent_gradients",
            lambda: check_concurrent_gradients(conc_a, conc_b, blosum, p.n_concurrent_grad, p))

    if do("long_seqs"):
        print("── Long sequences ────────────────────────────────────────────────")
        for nt in [4, 8]:
            run(f"long_seq_determinism[n_threads={nt}]",
                lambda nt=nt: check_long_seq_determinism(long_a, long_b, blosum, nt, p))
        run("very_long_no_crash",
            lambda: check_very_long_no_crash(vlong_a, vlong_b, blosum, p))

    if do("soft"):
        print("── Soft gradient ─────────────────────────────────────────────────")
        for nt in [2, 4, 8]:
            run(f"soft_grad_determinism[n_threads={nt}]",
                lambda nt=nt: check_soft_grad_determinism(large_a, large_b, blosum, nt, p))
        run("soft_geq_hard",
            lambda: check_soft_geq_hard(large_a, large_b, blosum, p))

    if do("integer_grad"):
        print("── Hard gradient integer-valued ──────────────────────────────────")
        run("hard_grad_integer",
            lambda: check_hard_grad_integer(large_a, large_b, blosum, p))

    if do("reproducibility"):
        print("── Reproducibility ───────────────────────────────────────────────")
        run("reproducibility", lambda: check_reproducibility(blosum, p))

    if do("more_threads"):
        print("── n_threads > batch_size ────────────────────────────────────────")
        run("more_threads_than_pairs", lambda: check_more_threads_than_pairs(blosum, p))

    if do("throughput"):
        print("── Throughput ────────────────────────────────────────────────────")
        run(f"throughput[n={p.n_xl}, limit={p.throughput_limit}s]",
            lambda: check_throughput(xl_a, xl_b, blosum, p.throughput_limit, p))

    if do("symmetry"):
        print("── Gradient symmetry on symmetric input ──────────────────────────")
        run("hard_grad_symmetry",
            lambda: check_grad_symmetry_symmetric_input(sym_seqs, blosum, "hard", p))
        run("soft_grad_symmetry",
            lambda: check_grad_symmetry_symmetric_input(sym_seqs, blosum, "soft", p))

    if do("concurrent_aligners"):
        print("── Concurrent different aligners ─────────────────────────────────")
        run("concurrent_different_aligners",
            lambda: check_concurrent_different_aligners(cd_a, cd_b, blosum, p))

    # Summary
    n_pass = sum(1 for _, ok, _, _ in results if ok)
    n_fail = sum(1 for _, ok, _, _ in results if not ok)
    total  = time.monotonic()
    print(f"\n{'─' * 60}")
    print(f"{n_pass} passed, {n_fail} failed  ({len(results)} total)")
    if n_fail:
        print("\nFailed tests:")
        for name, ok, _, exc in results:
            if not ok:
                print(f"  {name}: {exc}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
