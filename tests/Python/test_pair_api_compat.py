"""Compatibility behavior retained in 0.7 and removed in 0.8."""
import numpy as np
import pytest
import nwgrad

@pytest.mark.parametrize("suffix", ["", "Double"])
def test_optional_paths_on_views_preserve_other_pairs(suffix):
    p = nwgrad.AlignParams(np.eye(4), alphabet="ACGT")
    batch = getattr(nwgrad, "SeqPairBatch" + suffix)(["ACGT", "ACG"], ["ACGT", "ACG"], p, n_threads=1)
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


@pytest.mark.parametrize("suffix", ["", "Double"])
@pytest.mark.parametrize("view", [False, True])
def test_deprecations_and_mixed_lifecycle(suffix, view):
    import warnings
    from pathlib import Path
    p = nwgrad.AlignParams(np.eye(4), alphabet="ACGT")
    if view:
        batch = getattr(nwgrad, "SeqPairBatch"+suffix)(["ACGT", "AC"], ["ACGT", "AC"], p, n_threads=1)
        batch.score_and_grad()
        pair = batch[0]
    else:
        pair = getattr(nwgrad, "SeqPair"+suffix)("ACGT", "ACGT", p)
    operations = [
        ("align_full", lambda: pair.align_full()),
        ("compute_grad", lambda: pair.compute_grad()),
        ("realign_banded", lambda: pair.realign_banded(4)),
        ("dp_valid", lambda: pair.dp_valid),
        ("drop_dp", lambda: pair.drop_dp()),
    ]
    for name, call in operations:
        with warnings.catch_warnings(record=True) as got:
            warnings.simplefilter("always", DeprecationWarning)
            call()
        assert len(got) == 1
        assert got[0].category is DeprecationWarning
        assert name in str(got[0].message)
        assert "0.7.0" in str(got[0].message) and "0.8.0" in str(got[0].message)
        assert Path(got[0].filename).resolve() == Path(__file__).resolve()
    assert not pair.grad_valid
    with pytest.warns(DeprecationWarning), pytest.raises(RuntimeError, match="dropped"):
        pair.compute_grad()
    pair.align(keep_paths=False)
    assert pair.grad_valid
    with pytest.warns(DeprecationWarning):
        pair.align_full()
    assert not pair.grad_valid
    if view:
        batch.compute_grad()  # compatibility bridge publishes the view's gradient
        assert pair.grad_valid
    with pytest.warns(DeprecationWarning):
        pair.realign_banded(4)
    assert not pair.grad_valid
    pair.banded_align(4, keep_paths=False)
    assert pair.grad_valid and not pair.stored_path_valid
    with pytest.warns(DeprecationWarning):
        pair.compute_grad()  # already published; succeeds without a stored path


@pytest.mark.parametrize("suffix", ["", "Double"])
def test_warning_as_error_prevents_mutation(suffix):
    import warnings
    pair = getattr(nwgrad, "SeqPair"+suffix)("AC", "AC", nwgrad.AlignParams(np.eye(4), alphabet="ACGT"))
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        with pytest.raises(DeprecationWarning):
            pair.align_full()
    assert not pair.score_valid
