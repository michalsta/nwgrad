"""SeqPairBatch.scores() and SeqPairBatch.weighted_grad(): per-pair scores as one
array, and a weighted sum over the cached per-pair gradients."""

import numpy as np
import pytest
import nwgrad

SEQS_A = ["ACGTACGT", "AAT", "GATTACA", "CCGGA", "T", "ACGTTGCAAC"]
SEQS_B = ["AGTACT", "ACT", "GATCA", "CGGTA", "TT", "ACGTGCAAGC"]
BATCHES = [nwgrad.SeqPairBatch, nwgrad.SeqPairBatchDouble]
PROBLEMS = [(gm, mode) for gm in ("linear", "affine") for mode in ("global", "local")]
FIELDS = ("gap_open_a", "gap_extend_a", "gap_open_b", "gap_extend_b")


def dna_params():
    M = np.full((4, 4), -1.0)
    M[np.diag_indices(4)] = 2.0
    M[0, 2] = 0.5   # asymmetric, so an orientation error would show
    return nwgrad.AlignParams(nwgrad.SubstMatrix(M, alphabet="ACGT"),
                              gap_open_a=2.0, gap_extend_a=1.0,
                              gap_open_b=1.5, gap_extend_b=0.5)


def make_batch(cls, gap_model="affine", mode="local", n_threads=2, grad_mode="hard", **kw):
    params = dna_params()
    batch = cls(n_threads=n_threads, **kw)
    batch.add_many(SEQS_A, SEQS_B, params, gap_model=gap_model, mode=mode, grad_mode=grad_mode)
    return batch, params


def manual_weighted_sum(batch, weights):
    grads = [batch[i].grad.to_dict() for i in range(len(batch))]
    out = {"matrix": sum(w * g["matrix"] for w, g in zip(weights, grads))}
    for f in FIELDS:
        out[f] = sum(w * g[f] for w, g in zip(weights, grads))
    return out


def assert_grad_close(got, expected, rel=1e-12):
    got = got.to_dict() if hasattr(got, "to_dict") else got
    np.testing.assert_allclose(got["matrix"], expected["matrix"], rtol=rel, atol=1e-12)
    for f in FIELDS:
        assert got[f] == pytest.approx(expected[f], rel=rel, abs=1e-12), f


# ── scores() ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cls", BATCHES)
@pytest.mark.parametrize("gap_model, mode", PROBLEMS)
def test_scores_match_per_pair(cls, gap_model, mode):
    batch, _ = make_batch(cls, gap_model, mode)
    total = batch.score_and_grad()
    s = batch.scores()
    assert isinstance(s, np.ndarray) and s.dtype == np.float64 and s.shape == (len(SEQS_A),)
    assert list(s) == [batch[i].score for i in range(len(batch))]
    assert s.sum() == pytest.approx(total)


def test_scores_is_a_copy():
    batch, _ = make_batch(nwgrad.SeqPairBatchDouble)
    batch.score_and_grad()
    s = batch.scores()
    s[:] = 0.0
    assert batch.scores()[0] == batch[0].score != 0.0


def test_scores_of_empty_batch():
    assert nwgrad.SeqPairBatchDouble().scores().shape == (0,)


def test_scores_before_alignment_raise():
    batch, _ = make_batch(nwgrad.SeqPairBatchDouble)
    with pytest.raises(RuntimeError, match="score not valid"):
        batch.scores()


def test_scores_after_set_params_raise():
    batch, params = make_batch(nwgrad.SeqPairBatchDouble)
    batch.score_and_grad()
    batch.set_params(params)
    with pytest.raises(RuntimeError, match="score not valid"):
        batch.scores()


# ── weighted_grad() ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("cls", BATCHES)
@pytest.mark.parametrize("gap_model, mode", PROBLEMS)
def test_weighted_grad_matches_manual_sum(cls, gap_model, mode):
    batch, _ = make_batch(cls, gap_model, mode)
    batch.score_and_grad()
    w = np.random.default_rng(0).normal(size=len(batch))
    assert_grad_close(batch.weighted_grad(w), manual_weighted_sum(batch, w))


@pytest.mark.parametrize("cls", BATCHES)
def test_unit_weights_equal_compute_grad(cls):
    batch, _ = make_batch(cls)
    batch.score_and_grad()
    assert_grad_close(batch.weighted_grad(np.ones(len(batch))), batch.compute_grad().to_dict())


def test_zero_weights_give_zero_gradient():
    batch, _ = make_batch(nwgrad.SeqPairBatchDouble)
    batch.score_and_grad()
    g = batch.weighted_grad(np.zeros(len(batch))).to_dict()
    assert not g["matrix"].any()
    assert all(g[f] == 0.0 for f in FIELDS)
    assert g["alphabet"] == "ACGT"


def test_weighted_grad_is_linear_and_runs_no_alignment():
    batch, _ = make_batch(nwgrad.SeqPairBatchDouble)
    batch.score_and_grad()
    rng = np.random.default_rng(1)
    w1, w2 = rng.normal(size=len(batch)), rng.normal(size=len(batch))
    combined = batch.weighted_grad(0.3 * w1 - 2.0 * w2).to_dict()
    g1, g2 = batch.weighted_grad(w1).to_dict(), batch.weighted_grad(w2).to_dict()
    expected = {k: 0.3 * g1[k] - 2.0 * g2[k] for k in ("matrix",) + FIELDS}
    assert_grad_close(combined, expected, rel=1e-10)


@pytest.mark.parametrize("cls", BATCHES)
def test_weighted_grad_is_independent_of_threads_and_schedule(cls):
    w = np.random.default_rng(2).normal(size=len(SEQS_A))
    results = []
    for n_threads in (1, 3, 8):
        for schedule in ("dynamic", "sorted"):
            batch, _ = make_batch(cls, n_threads=n_threads)
            batch.schedule = schedule
            batch.score_and_grad()
            g = batch.weighted_grad(w).to_dict()
            results.append((g["matrix"].tobytes(), tuple(g[f] for f in FIELDS)))
    assert all(r == results[0] for r in results)


def test_weighted_grad_after_align_full_and_compute_grad():
    batch, _ = make_batch(nwgrad.SeqPairBatchDouble)
    batch.alloc_dp()
    batch.align_full()
    batch.compute_grad()
    w = np.linspace(-1.0, 1.0, len(batch))
    assert_grad_close(batch.weighted_grad(w), manual_weighted_sum(batch, w))


def test_weighted_grad_with_soft_gradients():
    batch, _ = make_batch(nwgrad.SeqPairBatchDouble, grad_mode="soft", mode="global")
    batch.score_and_grad()
    w = np.linspace(0.5, 2.0, len(batch))
    assert_grad_close(batch.weighted_grad(w), manual_weighted_sum(batch, w))


@pytest.mark.parametrize("weights", [
    np.array([1, 0, 2, -1, 3, 1]),                       # integer dtype
    np.array([1, 0, 2, -1, 3, 1], dtype=np.float32),     # single precision
    np.arange(12.0)[::2],                                # non-contiguous view
])
def test_weighted_grad_converts_arrays(weights):
    batch, _ = make_batch(nwgrad.SeqPairBatchDouble)
    batch.score_and_grad()
    as_f64 = np.ascontiguousarray(weights, dtype=np.float64)
    assert_grad_close(batch.weighted_grad(weights), manual_weighted_sum(batch, as_f64))


def test_weighted_grad_rejects_a_list():
    batch, _ = make_batch(nwgrad.SeqPairBatchDouble)
    batch.score_and_grad()
    with pytest.raises(TypeError):
        batch.weighted_grad([1.0] * len(batch))


def test_weighted_grad_wrong_length_raises():
    batch, _ = make_batch(nwgrad.SeqPairBatchDouble)
    batch.score_and_grad()
    with pytest.raises(ValueError, match="one weight per pair"):
        batch.weighted_grad(np.ones(len(batch) - 1))


def test_weighted_grad_of_empty_batch_raises():
    with pytest.raises(RuntimeError, match="empty batch"):
        nwgrad.SeqPairBatchDouble().weighted_grad(np.ones(0))


def test_weighted_grad_before_alignment_raises():
    batch, _ = make_batch(nwgrad.SeqPairBatchDouble)
    with pytest.raises(RuntimeError, match="gradient not computed"):
        batch.weighted_grad(np.ones(len(batch)))


def test_weighted_grad_after_set_params_raises():
    batch, params = make_batch(nwgrad.SeqPairBatchDouble)
    batch.score_and_grad()
    batch.set_params(params)
    with pytest.raises(RuntimeError, match="gradient not computed"):
        batch.weighted_grad(np.ones(len(batch)))


def test_weighted_grad_over_many_blocks_is_thread_independent_and_matches_numpy():
    """More pairs than one summation block (4096): same bits for any thread count."""
    rng = np.random.default_rng(7)
    n = 3 * 4096 + 17
    letters = np.array(list("ACGT"))
    seqs_a = ["".join(rng.choice(letters, size=rng.integers(4, 13))) for _ in range(n)]
    seqs_b = ["".join(rng.choice(letters, size=rng.integers(4, 17))) for _ in range(n)]
    w = rng.normal(size=n)
    results = []
    for threads in (1, 3, 8):
        batch = nwgrad.SeqPairBatchDouble(n_threads=threads)
        batch.add_many(seqs_a, seqs_b, dna_params(), gap_model="affine", mode="local")
        batch.score_and_grad()
        results.append(batch.weighted_grad(w).to_dict())
    for r in results[1:]:
        assert np.array_equal(r["matrix"], results[0]["matrix"])
        assert all(r[f] == results[0][f] for f in FIELDS)
    matrices, gaps = batch.grads()
    np.testing.assert_allclose(results[0]["matrix"], np.tensordot(w, matrices, axes=1),
                               rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose([results[0][f] for f in FIELDS], w @ gaps, rtol=1e-12, atol=1e-9)
