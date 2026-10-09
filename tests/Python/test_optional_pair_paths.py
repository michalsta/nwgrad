"""Optional standalone path storage preserves scores, gradients and banded guides."""
import numpy as np
import pytest
import nwgrad


@pytest.mark.parametrize("suffix", ["", "Double"])
@pytest.mark.parametrize("gap_model", ["linear", "affine"])
@pytest.mark.parametrize("mode", ["local", "global"])
@pytest.mark.parametrize("grad_mode", ["hard", "soft"])
@pytest.mark.parametrize("traceback", ["scores", "pointers", "hirschberg"])
def test_optional_paths_match_default(suffix, gap_model, mode, grad_mode, traceback):
    cls = getattr(nwgrad, "SeqPair" + suffix)
    p = nwgrad.AlignParams(np.array([[3., -1., -2., -1.], [-1., 2., -1., -2.],
                                   [-2., -1., 4., -1.], [-1., -2., -1., 3.]]),
                            alphabet="ACGT", gap_open_a=2., gap_extend_a=.5,
                            gap_open_b=3., gap_extend_b=.25)
    kwargs = dict(gap_model=gap_model, mode=mode, grad_mode=grad_mode, traceback=traceback)
    kept = cls("ACGTACGTTAC", "ACGTCGTAC", p, **kwargs)
    light = cls("ACGTACGTTAC", "ACGTCGTAC", p, **kwargs)
    kept.hb_cutoff = light.hb_cutoff = 2
    score, grad = kept.score_and_grad()
    score2, grad2 = light.score_and_grad(keep_paths=False)
    assert score == score2
    for field, value in grad.to_dict().items():
        np.testing.assert_equal(value, grad2.to_dict()[field])
    assert kept.stored_path_valid and not light.stored_path_valid
    assert kept.grad_valid and light.grad_valid
    assert list(kept.guide_j) == list(light.guide_j)
    with pytest.raises(RuntimeError):
        light.aligned()
    with pytest.raises(RuntimeError):
        light.coordinates()
    # Cached gradient remains usable without a stored path.
    light.drop_paths()
    assert light.grad_valid
    # Guide reuse and banded computations restore paths and publish gradients.
    kept.banded_align(4)
    light.banded_align(4)
    assert light.grad_valid
    assert light.aligned() == kept.aligned()
    np.testing.assert_equal(light.grad.to_dict(), kept.grad.to_dict())
    light.score_and_grad(keep_paths=False)
    assert not light.stored_path_valid and light.grad_valid
    light.score_and_grad()  # default restores the supported path availability
    assert light.stored_path_valid and light.grad_valid


@pytest.mark.parametrize("keep_paths", [False, True])
def test_gradient_free_pair_rejects_fused_gradient(keep_paths):
    p = nwgrad.AlignParams(np.eye(4), alphabet="ACGT")
    pair = nwgrad.SeqPairDouble("ACGT", "ACGT", p, grad_mode="none")
    with pytest.raises(RuntimeError, match="grad_mode"):
        pair.score_and_grad(keep_paths=keep_paths)
