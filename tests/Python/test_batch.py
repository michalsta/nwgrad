"""BatchAligner tests: correctness, thread safety, and API surface."""

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


def make_params(gap_extend, gap_open=0.0, matrix_arr=None):
    arr = BLOSUM62 if matrix_arr is None else matrix_arr
    return nwgrad.AlignParams(arr, gap_open_a=gap_open, gap_extend_a=gap_extend,
                                   gap_open_b=gap_open, gap_extend_b=gap_extend)


def grad_matrix(g):
    return g.matrix.to_matrix()


@pytest.fixture(scope="module")
def blosum():
    return nwgrad.SubstMatrix(BLOSUM62)


# ── Single-threaded correctness ───────────────────────────────────────────────

@pytest.mark.parametrize("gap_model,mode,gap_open,gap_extend,score_fn,grad_fn", [
    ("linear", "global", 0.0,  1.0, nwgrad.nw_score,        nwgrad.nw_grad),
    ("linear", "local",  0.0,  1.0, nwgrad.sw_score,        nwgrad.sw_grad),
    ("affine", "global", 11.0, 1.0, nwgrad.nw_score_affine, nwgrad.nw_affine_grad),
    ("affine", "local",  11.0, 1.0, nwgrad.sw_score_affine, nwgrad.sw_affine_grad),
])
def test_batch_scores_match_single(gap_model, mode, gap_open, gap_extend,
                                   score_fn, grad_fn):
    seqs_a = [p[0] for p in PAIRS]
    seqs_b = [p[1] for p in PAIRS]
    p = make_params(gap_extend, gap_open)

    expected = [score_fn(a, b, p) for a, b in PAIRS]

    aligner = nwgrad.BatchAligner(
        params=p,
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
def test_batch_grad_matches_sum_of_singles(gap_model, mode, gap_open, gap_extend, grad_fn):
    seqs_a = [p[0] for p in PAIRS]
    seqs_b = [p[1] for p in PAIRS]
    p = make_params(gap_extend, gap_open)

    expected_grads = [grad_matrix(grad_fn(a, b, p)[1]) for a, b in PAIRS]
    expected_total = sum(expected_grads)

    aligner = nwgrad.BatchAligner(
        params=p,
        gap_model=gap_model,
        mode=mode,
        grad_mode="hard",
        n_threads=1,
    )
    result = aligner.align(seqs_a, seqs_b)
    got = grad_matrix(result.grad)

    assert got.shape == (20, 20)
    np.testing.assert_allclose(got, expected_total, atol=1e-10)


# ── Multi-threaded correctness ────────────────────────────────────────────────

@pytest.mark.parametrize("n_threads", [2, 4, 8])
def test_multithread_scores_match_singlethread(n_threads):
    seqs_a = [p[0] for p in PAIRS] * 5
    seqs_b = [p[1] for p in PAIRS] * 5
    p = make_params(1.0, 11.0)

    ref_aligner = nwgrad.BatchAligner(
        params=p, gap_model="affine", mode="global", grad_mode="hard", n_threads=1,
    )
    ref = ref_aligner.align(seqs_a, seqs_b)
    ref_scores = np.array(ref.scores)
    ref_grad   = grad_matrix(ref.grad)

    multi_aligner = nwgrad.BatchAligner(
        params=p, gap_model="affine", mode="global", grad_mode="hard", n_threads=n_threads,
    )
    got = multi_aligner.align(seqs_a, seqs_b)
    got_scores = np.array(got.scores)
    got_grad   = grad_matrix(got.grad)

    np.testing.assert_allclose(got_scores, ref_scores, atol=1e-10,
                                err_msg="scores differ between 1 and %d threads" % n_threads)
    np.testing.assert_allclose(got_grad, ref_grad, atol=1e-10,
                                err_msg="gradient differs between 1 and %d threads" % n_threads)


# ── API surface tests ────────────────────────────────────────────────────────

def test_empty_batch():
    p = make_params(1.0, 11.0)
    aligner = nwgrad.BatchAligner(params=p, n_threads=1)
    result = aligner.align([], [])
    assert np.array(result.scores).shape == (0,)
    assert grad_matrix(result.grad).shape == (20, 20)
    assert grad_matrix(result.grad).sum() == 0.0


def test_grad_none_returns_zero_grad():
    p = make_params(1.0, 11.0)
    aligner = nwgrad.BatchAligner(
        params=p, gap_model="affine", mode="global", grad_mode="none", n_threads=1,
    )
    result = aligner.align(["ACDE"], ["ACDE"])
    assert grad_matrix(result.grad).sum() == pytest.approx(0.0)


def test_mismatched_lengths_raises():
    p = make_params(1.0)
    aligner = nwgrad.BatchAligner(params=p)
    with pytest.raises(Exception):
        aligner.align(["A", "C"], ["A"])


def test_result_scores_shape():
    p = make_params(1.0)
    aligner = nwgrad.BatchAligner(params=p, n_threads=1)
    seqs = ["ACDE", "MADEEKLF", "A"]
    result = aligner.align(seqs, seqs)
    assert np.array(result.scores).shape == (3,)


def test_result_grad_shape():
    p = make_params(1.0)
    aligner = nwgrad.BatchAligner(params=p, n_threads=1)
    result = aligner.align(["ACDE"], ["ACDE"])
    assert grad_matrix(result.grad).shape == (20, 20)


# ── Stress test: large batch, multi-thread ───────────────────────────────────

def test_large_batch_multithreaded():
    rng = np.random.default_rng(42)
    alphabet = list(AA_ORDER)
    seqs_a = ["".join(rng.choice(alphabet, size=int(rng.integers(5, 30))).tolist())
              for _ in range(200)]
    seqs_b = ["".join(rng.choice(alphabet, size=int(rng.integers(5, 30))).tolist())
              for _ in range(200)]
    p = make_params(1.0, 11.0)

    ref = nwgrad.BatchAligner(
        params=p, gap_model="affine", mode="global", grad_mode="hard", n_threads=1,
    ).align(seqs_a, seqs_b)

    got = nwgrad.BatchAligner(
        params=p, gap_model="affine", mode="global", grad_mode="hard", n_threads=4,
    ).align(seqs_a, seqs_b)

    np.testing.assert_allclose(np.array(got.scores), np.array(ref.scores), atol=1e-10)
    np.testing.assert_allclose(grad_matrix(got.grad), grad_matrix(ref.grad), atol=1e-10)
