"""The 0.6.0 and 0.7.0 compatibility surfaces are removed; batch types are always fixed."""

import numpy as np
import pytest
import nwgrad


@pytest.mark.parametrize("suffix", ["", "Double"])
def test_removed_methods(suffix):
    params = nwgrad.AlignParams(np.eye(4), alphabet="ACGT")
    batch = getattr(nwgrad, "SeqPairBatch" + suffix)(["ACGT"], ["ACGT"], params, n_threads=1)
    for method in ("add_many", "alloc_dp", "align_full", "realign_banded", "drop_dp"):
        assert not hasattr(batch, method)
    for pair in (getattr(nwgrad, "SeqPair" + suffix)("ACGT", "ACGT", params), batch[0]):
        for name in ("alloc_dp", "align_full", "realign_banded", "compute_grad", "drop_dp",
                     "dp_valid"):
            assert not hasattr(pair, name)


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
