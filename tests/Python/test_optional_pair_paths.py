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
    assert kept.dp_valid and not light.dp_valid
    assert kept.grad_valid and light.grad_valid
    assert list(kept.guide_j) == list(light.guide_j)
    with pytest.raises(RuntimeError):
        light.aligned()
    with pytest.raises(RuntimeError):
        light.coordinates()
    # Cached gradient remains usable without a stored path.
    light.compute_grad()
    light.drop_dp()
    assert light.grad_valid
    # Guide reuse and legacy pair operations still restore paths and defer gradients.
    kept.realign_banded(4)
    light.realign_banded(4)
    assert not light.grad_valid
    kept.compute_grad(); light.compute_grad()
    assert light.aligned() == kept.aligned()
    np.testing.assert_equal(light.grad.to_dict(), kept.grad.to_dict())
    light.score_and_grad(keep_paths=False)
    assert not light.dp_valid and light.grad_valid
    light.score_and_grad()  # default restores the supported path availability
    assert light.dp_valid and light.grad_valid


@pytest.mark.parametrize("suffix", ["", "Double"])
def test_optional_paths_on_views_preserve_other_pairs(suffix):
    p = nwgrad.AlignParams(np.eye(4), alphabet="ACGT")
    batch = getattr(nwgrad, "SeqPairBatch" + suffix)(n_threads=1)
    batch.add_many(["ACGT", "ACG"], ["ACGT", "ACG"], p)
    batch.score_and_grad(keep_paths=True)
    first, other = batch[0], batch[0]  # two handles must observe the same state
    batch[1].score_and_grad(keep_paths=False)
    assert first.dp_valid and not batch[1].dp_valid
    first.align_full()
    assert not other.grad_valid
    other.compute_grad()
    assert first.grad_valid
    first.align_full()
    batch.drop_paths()
    assert not other.grad_valid
    with pytest.raises(RuntimeError, match="dropped"):
        other.compute_grad()
    other.score_and_grad(keep_paths=False)
    assert first.grad_valid and not first.dp_valid
    assert batch.compute_grad() is not None


@pytest.mark.parametrize("keep_paths", [False, True])
def test_gradient_free_pair_rejects_fused_gradient(keep_paths):
    p = nwgrad.AlignParams(np.eye(4), alphabet="ACGT")
    pair = nwgrad.SeqPairDouble("ACGT", "ACGT", p, grad_mode="none")
    with pytest.raises(RuntimeError, match="grad_mode"):
        pair.score_and_grad(keep_paths=keep_paths)


@pytest.mark.parametrize("suffix", ["", "Double"])
@pytest.mark.parametrize("operation", ["align_full", "realign_banded"])
@pytest.mark.parametrize("before", ["unreleased", "released", "dropped"])
def test_failed_fill_cannot_publish_an_old_gradient(suffix, operation, before):
    p = nwgrad.AlignParams(np.eye(4) * 800, alphabet="ACGT")
    pair = getattr(nwgrad, "SeqPair" + suffix)("ACGT", "ACGT", p, grad_mode="soft")
    pair.soft_impl = "log"
    pair.align_full()
    if before == "released":
        pair.compute_grad()
    elif before == "dropped":
        pair.drop_dp()
    # Viterbi can finish and store a path before the scaled soft pass rejects exp(800).
    pair.soft_impl = "scaled"
    with pytest.raises(ValueError, match="scaled_or_log"):
        getattr(pair, operation)(4) if operation == "realign_banded" else pair.align_full()
    assert not pair.score_valid and not pair.grad_valid
    with pytest.raises(RuntimeError):
        pair.compute_grad()
    pair.soft_impl = "log"
    pair.score_and_grad(keep_paths=False)
    assert pair.score_valid and pair.grad_valid
