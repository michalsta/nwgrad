"""Steps 6-7: BatchAligner tests.

Covers correctness (scores and gradients match single-pair functions), thread
safety (multi-thread results equal single-thread), and basic API surface.
"""

import numpy as np
import pytest
import nwgrad
from test_subst_matrix import BLOSUM62

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"

PAIRS = [
    ("A",          "A"),
    ("ACDE",       "ACDE"),
    ("ACDE",       "ACDF"),
    ("A",          "AC"),
    ("PLEASANTLY", "MEANLY"),
    ("ACDEFGHIKL", "CDEFGHIKLM"),
    ("MADEEKLF",   "MADEEKLF"),
    ("ACDEFG",     "ACDE"),
    ("MADEEKLF",   "ACDEFGHIKL"),
]


@pytest.fixture(scope="module")
def blosum():
    return nwgrad.SubstMatrix(BLOSUM62)


# ── Single-threaded correctness ───────────────────────────────────────────────

@pytest.mark.parametrize("gap_model,mode,gap_open,gap_extend,score_fn,grad_fn", [
    ("linear", "global", 0.0,  1.0, nwgrad.nw_score, nwgrad.nw_grad),
    ("linear", "local",  0.0,  1.0, nwgrad.sw_score, nwgrad.sw_grad),
    ("affine", "global", 11.0, 1.0, nwgrad.nw_score_affine, nwgrad.nw_affine_grad),
    ("affine", "local",  11.0, 1.0, nwgrad.sw_score_affine, nwgrad.sw_affine_grad),
])
def test_batch_scores_match_single(gap_model, mode, gap_open, gap_extend,
                                   score_fn, grad_fn, blosum):
    """Batch scores must match the single-pair score functions."""
    seqs_a = [p[0] for p in PAIRS]
    seqs_b = [p[1] for p in PAIRS]

    if gap_model == "linear":
        expected = [score_fn(a, b, blosum, gap_extend) for a, b in PAIRS]
    else:
        expected = [score_fn(a, b, blosum, gap_open, gap_extend) for a, b in PAIRS]

    aligner = nwgrad.BatchAligner(
        matrix=blosum,
        gap_open=gap_open,
        gap_extend=gap_extend,
        gap_model=gap_model,
        mode=mode,
        grad_mode="none",
        n_threads=1,
    )
    result = aligner.align(seqs_a, seqs_b)
    scores = np.array(result.scores)

    assert scores.shape == (len(PAIRS),)
    for i, exp in enumerate(expected):
        assert scores[i] == pytest.approx(exp), (
            f"pair {i} ({PAIRS[i]}): got {scores[i]}, expected {exp}"
        )


@pytest.mark.parametrize("gap_model,mode,gap_open,gap_extend,grad_fn", [
    ("linear", "global", 0.0,  1.0, nwgrad.nw_grad),
    ("linear", "local",  0.0,  1.0, nwgrad.sw_grad),
    ("affine", "global", 11.0, 1.0, nwgrad.nw_affine_grad),
    ("affine", "local",  11.0, 1.0, nwgrad.sw_affine_grad),
])
def test_batch_grad_matches_sum_of_singles(gap_model, mode, gap_open, gap_extend,
                                           grad_fn, blosum):
    """Batch gradient must equal the sum of individual pair gradients."""
    seqs_a = [p[0] for p in PAIRS]
    seqs_b = [p[1] for p in PAIRS]

    if gap_model == "linear":
        expected_grads = [np.array(grad_fn(a, b, blosum, gap_extend)[1]) for a, b in PAIRS]
    else:
        expected_grads = [np.array(grad_fn(a, b, blosum, gap_open, gap_extend)[1])
                          for a, b in PAIRS]
    expected_total = sum(expected_grads)

    aligner = nwgrad.BatchAligner(
        matrix=blosum,
        gap_open=gap_open,
        gap_extend=gap_extend,
        gap_model=gap_model,
        mode=mode,
        grad_mode="hard",
        n_threads=1,
    )
    result = aligner.align(seqs_a, seqs_b)
    got = np.array(result.grad)

    assert got.shape == (20, 20)
    np.testing.assert_allclose(got, expected_total, atol=1e-10)


# ── Multi-threaded correctness ────────────────────────────────────────────────

@pytest.mark.parametrize("n_threads", [2, 4, 8])
def test_multithread_scores_match_singlethread(n_threads, blosum):
    seqs_a = [p[0] for p in PAIRS] * 5   # repeat to give threads enough work
    seqs_b = [p[1] for p in PAIRS] * 5

    ref_aligner = nwgrad.BatchAligner(
        matrix=blosum, gap_open=11.0, gap_extend=1.0,
        gap_model="affine", mode="global", grad_mode="hard", n_threads=1,
    )
    ref = ref_aligner.align(seqs_a, seqs_b)
    ref_scores = np.array(ref.scores)
    ref_grad   = np.array(ref.grad)

    multi_aligner = nwgrad.BatchAligner(
        matrix=blosum, gap_open=11.0, gap_extend=1.0,
        gap_model="affine", mode="global", grad_mode="hard", n_threads=n_threads,
    )
    got = multi_aligner.align(seqs_a, seqs_b)
    got_scores = np.array(got.scores)
    got_grad   = np.array(got.grad)

    np.testing.assert_allclose(got_scores, ref_scores, atol=1e-10,
                                err_msg="scores differ between 1 and %d threads" % n_threads)
    np.testing.assert_allclose(got_grad, ref_grad, atol=1e-10,
                                err_msg="gradient differs between 1 and %d threads" % n_threads)


# ── API surface tests ────────────────────────────────────────────────────────

def test_empty_batch(blosum):
    aligner = nwgrad.BatchAligner(matrix=blosum, n_threads=1)
    result = aligner.align([], [])
    assert np.array(result.scores).shape == (0,)
    assert np.array(result.grad).shape == (20, 20)
    assert np.array(result.grad).sum() == 0.0


def test_grad_none_returns_zero_grad(blosum):
    """grad_mode='none' should still return a zero 20×20 gradient."""
    aligner = nwgrad.BatchAligner(
        matrix=blosum, gap_open=11.0, gap_extend=1.0,
        gap_model="affine", mode="global", grad_mode="none", n_threads=1,
    )
    result = aligner.align(["ACDE"], ["ACDE"])
    assert np.array(result.grad).sum() == pytest.approx(0.0)


def test_mismatched_lengths_raises(blosum):
    aligner = nwgrad.BatchAligner(matrix=blosum)
    with pytest.raises(Exception):
        aligner.align(["A", "C"], ["A"])


def test_result_scores_shape(blosum):
    aligner = nwgrad.BatchAligner(matrix=blosum, n_threads=1)
    seqs = ["ACDE", "MADEEKLF", "A"]
    result = aligner.align(seqs, seqs)
    assert np.array(result.scores).shape == (3,)


def test_result_grad_shape(blosum):
    aligner = nwgrad.BatchAligner(matrix=blosum, n_threads=1)
    result = aligner.align(["ACDE"], ["ACDE"])
    assert np.array(result.grad).shape == (20, 20)


# ── Stress test: large batch, multi-thread ───────────────────────────────────

def test_large_batch_multithreaded(blosum):
    """Larger batch: multi-thread result matches single-thread."""
    rng = np.random.default_rng(42)
    alphabet = list(AA_ORDER)
    seqs_a = ["".join(rng.choice(alphabet, size=int(rng.integers(5, 30))).tolist())
              for _ in range(200)]
    seqs_b = ["".join(rng.choice(alphabet, size=int(rng.integers(5, 30))).tolist())
              for _ in range(200)]

    ref = nwgrad.BatchAligner(
        matrix=blosum, gap_open=11.0, gap_extend=1.0,
        gap_model="affine", mode="global", grad_mode="hard", n_threads=1,
    ).align(seqs_a, seqs_b)

    got = nwgrad.BatchAligner(
        matrix=blosum, gap_open=11.0, gap_extend=1.0,
        gap_model="affine", mode="global", grad_mode="hard", n_threads=4,
    ).align(seqs_a, seqs_b)

    np.testing.assert_allclose(np.array(got.scores), np.array(ref.scores), atol=1e-10)
    np.testing.assert_allclose(np.array(got.grad),   np.array(ref.grad),   atol=1e-10)
