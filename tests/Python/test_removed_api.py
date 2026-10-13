"""The 0.6.0 compatibility surface is removed; batch types are always fixed."""

import numpy as np
import pytest
import nwgrad


@pytest.mark.parametrize("suffix", ["", "Double"])
@pytest.mark.filterwarnings("ignore:SeqPairBatch.add_many")   # deprecated, removed in 0.8
def test_removed_methods_and_per_add_types(suffix):
    cls = getattr(nwgrad, "SeqPairBatch" + suffix)
    batch = cls(n_threads=1)
    params = nwgrad.AlignParams(np.eye(4), alphabet="ACGT")
    for method in ("alloc_dp", "align_full", "realign_banded", "drop_dp"):
        assert not hasattr(batch, method)
    pair = getattr(nwgrad, "SeqPair" + suffix)("ACGT", "ACGT", params)
    assert not hasattr(pair, "alloc_dp")
    for field, value in (("gap_model", "affine"), ("mode", "global"), ("grad_mode", "hard")):
        with pytest.raises(TypeError):
            batch.add_many(["ACGT"], ["ACGT"], params, **{field: value})
        assert len(batch) == 0


@pytest.mark.parametrize("suffix", ["", "Double"])
def test_type_is_fixed_before_successful_add(suffix):
    cls = getattr(nwgrad, "SeqPairBatch" + suffix)
    default = cls(n_threads=1)
    assert (default.gap_model, default.mode, default.grad_mode) == ("affine", "global", "hard")
    kw = dict(n_threads=1, gap_model="linear", mode="local", grad_mode="soft")
    params = nwgrad.AlignParams(np.eye(4), alphabet="ACGT")
    batch = cls([], [], params, **kw)
    assert (batch.gap_model, batch.mode, batch.grad_mode) == ("linear", "local", "soft")
    with pytest.raises(ValueError):
        cls(["ACGZ"], ["ACGT"], params, **kw)
    batch = cls(["ACGT"], ["ACGT"], params, **kw)
    assert (batch.gap_model, batch.mode, batch.grad_mode) == ("linear", "local", "soft")
    batch.score_and_grad(keep_paths=True)
    assert batch[0].grad_valid
    gradients = batch.grads()
    total = batch.compute_grad().to_dict()
    batch.drop_paths()
    assert batch[0].grad_valid and not batch[0].stored_path_valid
    for before, after in zip(gradients, batch.grads()):
        np.testing.assert_array_equal(before, after)
    np.testing.assert_array_equal(total["matrix"], batch.compute_grad().to_dict()["matrix"])


@pytest.mark.parametrize("suffix", ["", "Double"])
def test_supported_pair_view_can_release_gradient_through_batch(suffix):
    params = nwgrad.AlignParams(np.eye(4), alphabet="ACGT")
    batch = getattr(nwgrad, "SeqPairBatch" + suffix)(["ACGT"], ["ACGT"], params, n_threads=1)
    view = batch[0]
    view.align_full()
    assert not view.grad_valid
    batch.compute_grad()
    assert view.grad_valid
