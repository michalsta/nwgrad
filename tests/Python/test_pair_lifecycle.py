"""Pair lifecycle on views and standalone pairs: shared state, paths kept per pair, and
failed fills that must not leave old results visible."""
import numpy as np
import pytest
import nwgrad


@pytest.mark.parametrize("suffix", ["", "Double"])
def test_views_share_state_and_keep_paths_per_pair(suffix):
    p = nwgrad.AlignParams(np.eye(4), alphabet="ACGT")
    batch = getattr(nwgrad, "SeqPairBatch" + suffix)(["ACGT", "ACG"], ["ACGT", "ACG"], p, n_threads=1)
    batch.score_and_grad(keep_paths=True)
    first, other = batch[0], batch[0]  # two handles must observe the same state
    batch[1].score_and_grad(keep_paths=False)
    assert first.stored_path_valid and not batch[1].stored_path_valid
    first.align(keep_paths=False)
    assert not other.stored_path_valid and other.grad_valid
    other.align()
    assert first.stored_path_valid and first.grad_valid
    batch.drop_paths()
    assert not first.stored_path_valid
    assert other.score_valid and other.grad_valid and other.path_valid
    assert batch.compute_grad() is not None


@pytest.mark.parametrize("suffix", ["", "Double"])
@pytest.mark.parametrize("operation", ["align", "banded_align"])
@pytest.mark.parametrize("before", ["kept", "dropped"])
def test_failed_fill_invalidates_the_old_results(suffix, operation, before):
    p = nwgrad.AlignParams(np.eye(4) * 800, alphabet="ACGT")
    pair = getattr(nwgrad, "SeqPair" + suffix)("ACGT", "ACGT", p, grad_mode="soft")
    pair.soft_impl = "log"
    pair.align()
    assert pair.grad_valid
    if before == "dropped":
        pair.drop_paths()
    # Viterbi can finish and store a path before the scaled soft pass rejects exp(800).
    pair.soft_impl = "scaled"
    with pytest.raises(ValueError, match="scaled_or_log"):
        pair.banded_align(4) if operation == "banded_align" else pair.align()
    assert not pair.score_valid and not pair.grad_valid
    assert pair.score is None and pair.grad is None
    pair.soft_impl = "log"
    pair.score_and_grad(keep_paths=False)
    assert pair.score_valid and pair.grad_valid
