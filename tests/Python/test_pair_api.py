"""Final pair API: results are published immediately, paths are independent."""
import warnings
import numpy as np
import pytest
import nwgrad as n


def params():
    return n.AlignParams(np.eye(4)*3-1, alphabet="ACGT",
                         gap_open_a=2, gap_extend_a=.5,
                         gap_open_b=3, gap_extend_b=.25)


@pytest.mark.parametrize("suffix", ["", "Double"])
@pytest.mark.parametrize("gap_model", ["linear", "affine"])
@pytest.mark.parametrize("mode", ["global", "local"])
@pytest.mark.parametrize("gd", ["hard", "soft", "none"])
@pytest.mark.parametrize("tb", ["scores", "pointers", "hirschberg"])
@pytest.mark.parametrize("seqs", [("ACGTAC", "AGTAC"), ("", "AC")])
@pytest.mark.parametrize("keep", [False, True])
@pytest.mark.parametrize("view", [False, True])
def test_final_lifecycle(suffix, gap_model, mode, gd, tb, seqs, keep, view):
    kw = dict(gap_model=gap_model, mode=mode, grad_mode=gd, traceback=tb)
    p = params()
    if view:
        owner = getattr(n, "SeqPairBatch"+suffix)([seqs[0], "AC"], [seqs[1], "AC"], p, n_threads=1, **kw)
        owner.hb_cutoff = 2
        pair = owner[0]
    else:
        pair = getattr(n, "SeqPair"+suffix)(*seqs, p, **kw)
        pair.hb_cutoff = 2
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert pair.align(keep_paths=keep) == pair.score
        assert pair.score_valid and pair.path_valid
        assert pair.stored_path_valid == keep
        assert pair.grad_valid == (gd != "none")
        if gd == "none":
            assert pair.grad is None
        else:
            before = pair.grad.to_dict()
            ref = getattr(n, "SeqPair"+suffix)(*seqs, p, **kw)
            ref.hb_cutoff = 2
            score, grad = ref.score_and_grad(keep_paths=keep)
            assert score == pair.score
            np.testing.assert_equal(before, grad.to_dict())
        if keep:
            path = pair.aligned()
            assert path[0].replace("-", "") in seqs[0]
            pair.coordinates()
        else:
            with pytest.raises(RuntimeError):
                pair.aligned()
        guide = list(pair.guide_j)
        pair.drop_paths()
        assert not pair.stored_path_valid and pair.score_valid
        assert list(pair.guide_j) == guide
        assert pair.grad_valid == (gd != "none")
        if view:
            assert not owner[1].score_valid
            owner.set_params(p)
        else:
            pair.set_params(p)
        assert not pair.score_valid and not pair.grad_valid
        assert list(pair.guide_j) == guide
        assert pair.banded_align(8, keep_paths=keep) == pair.score
        assert pair.grad_valid == (gd != "none")
        assert pair.stored_path_valid == keep
        if gd != "none":
            np.testing.assert_equal(before, pair.grad.to_dict())


@pytest.mark.parametrize("suffix", ["", "Double"])
def test_default_paths_and_missing_guide(suffix):
    pair = getattr(n, "SeqPair"+suffix)("AC", "AC", params())
    with pytest.raises(RuntimeError, match="align.*banded_align"):
        pair.banded_align(2)
    pair.align()
    assert pair.stored_path_valid
    pair.drop_paths()
    pair.banded_align(2)
    assert pair.stored_path_valid


@pytest.mark.parametrize("suffix", ["", "Double"])
@pytest.mark.parametrize("banded", [False, True])
def test_failed_modern_fill_invalidates_results(suffix, banded):
    p = n.AlignParams(np.eye(4)*800, alphabet="ACGT")
    pair = getattr(n, "SeqPair"+suffix)("ACGT", "ACGT", p, grad_mode="soft")
    pair.soft_impl = "log"
    pair.align()
    pair.soft_impl = "scaled"
    with pytest.raises(ValueError, match="scaled_or_log"):
        pair.banded_align(4) if banded else pair.align()
    assert not pair.score_valid and not pair.grad_valid
    pair.soft_impl = "log"
    pair.align(keep_paths=False)
    assert pair.grad_valid and not pair.stored_path_valid
