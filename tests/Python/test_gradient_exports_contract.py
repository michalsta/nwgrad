"""Independent derivative, cache validity, and ownership checks for batch exports."""

import numpy as np
import pytest
import nwgrad

from test_weighted_grad import FIELDS, PROBLEMS, make_batch
from test_grads import per_pair


@pytest.mark.parametrize("class_name", ["SeqPairBatch", "SeqPairBatchDouble"])
@pytest.mark.parametrize("gap_model,mode", PROBLEMS)
@pytest.mark.parametrize("component", ["matrix", "gaps"])
def test_weighted_soft_grad_matches_composed_objective_finite_difference(
        class_name, gap_model, mode, component, request):
    """Check the chain rule against scores, independently of cached gradients."""
    if mode == "local" and component == "gaps":
        request.node.add_marker(pytest.mark.xfail(
            strict=True, reason="local soft gap gradients count nonexistent border transitions"))
    batch, params = make_batch(getattr(nwgrad, class_name), gap_model, mode,
                              grad_mode="soft", n_threads=1)
    batch.score_and_grad()
    scores = batch.scores()
    # L = log(sum(exp(score_i))); dL/dscore_i is the softmax weight.
    weights = np.exp(scores - scores.max())
    weights /= weights.sum()
    analytic = batch.weighted_grad(weights).to_dict()
    original = params.to_dict()

    def objective(matrix, gaps):
        shifted = nwgrad.AlignParams(matrix, alphabet="ACGT", **gaps)
        batch.set_params(shifted)
        batch.score_and_grad()
        values = batch.scores()
        peak = values.max()
        return peak + np.log(np.exp(values - peak).sum())

    step = 1e-5
    gaps = {field: original[field] for field in FIELDS}
    for i, j in (np.ndindex(4, 4) if component == "matrix" else []):
        plus, minus = original["matrix"].copy(), original["matrix"].copy()
        plus[i, j] += step
        minus[i, j] -= step
        numeric = (objective(plus, gaps) - objective(minus, gaps)) / (2 * step)
        assert analytic["matrix"][i, j] == pytest.approx(numeric, abs=2e-8)
    for field in (FIELDS if component == "gaps" else []):
        plus, minus = gaps.copy(), gaps.copy()
        plus[field] += step
        minus[field] -= step
        numeric = (objective(original["matrix"], plus)
                   - objective(original["matrix"], minus)) / (2 * step)
        assert analytic[field] == pytest.approx(numeric, abs=2e-8), field


@pytest.mark.parametrize("class_name", ["SeqPairBatch", "SeqPairBatchDouble"])
def test_exported_arrays_outlive_batch(class_name):
    import gc

    batch, params = make_batch(getattr(nwgrad, class_name))
    batch.score_and_grad()
    matrices, gaps = batch.grads()
    scores = batch.scores()
    expected = matrices.copy(), gaps.copy(), scores.copy()
    del batch, params
    gc.collect()
    # Keep newly allocated arrays live to expose views into released storage.
    trash = [np.full(1024, -99.0) for _ in range(100)]
    for got, want in zip((matrices, gaps, scores), expected):
        np.testing.assert_array_equal(got, want)
    assert trash[0][0] == -99.0


@pytest.mark.parametrize("class_name", ["SeqPairBatch", "SeqPairBatchDouble"])
def test_gradient_exports_reject_invalid_later_pair(class_name):
    batch, params = make_batch(getattr(nwgrad, class_name))
    batch.score_and_grad()
    # A valid first pair must not hide an invalid cache later in the batch.
    batch[-1].set_params(params)
    with pytest.raises(RuntimeError, match="gradient not computed"):
        batch.grads()
    with pytest.raises(RuntimeError, match="gradient not computed"):
        batch.weighted_grad(np.ones(len(batch)))
    with pytest.raises(RuntimeError, match="score not valid"):
        batch.scores()
    batch.score_and_grad()
    for got, want in zip(batch.grads(), per_pair(batch)):
        np.testing.assert_array_equal(got, want)
