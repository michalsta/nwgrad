"""SeqPairBatch(seqs_a, seqs_b, params, ...): the pairs given once, at construction.

The constructor is the add_many() of an empty batch (one segment), so its batch must
be exactly the one add_many() builds; add_many() itself is deprecated, still works,
and warns.  The pairless form, SeqPairBatch(n_threads, traceback, ...), stays for
streaming align()."""

import warnings

import numpy as np
import pytest

import nwgrad

ALPHA = "ACGT"
BATCH = {"_double": nwgrad.SeqPairBatchDouble, "": nwgrad.SeqPairBatch}
CASES = [("linear", "global"), ("linear", "local"), ("affine", "global"), ("affine", "local")]


def problem(rng, n, K):
    A = ["".join(rng.choice(list(ALPHA), size=int(rng.integers(0, 30)))) for _ in range(n)]
    B = ["".join(rng.choice(list(ALPHA), size=int(rng.integers(0, 30)))) for _ in range(n)]
    P = nwgrad.AlignParams.from_arrays(rng.normal(0, 1.5, size=(K, 4, 4)), rng.uniform(0.2, 2.0, size=(K, 4)), ALPHA)
    tracks = [list(rng.integers(0, K, size=len(a))) for a in A] if K > 1 else None
    return A, B, P, tracks


def legacy(cls, A, B, P, **kw):
    """The deprecated way: an empty batch, then add_many()."""
    tracks = kw.pop("tracks", None)
    b = cls(**kw)
    with pytest.warns(DeprecationWarning, match=r"add_many\(\) is deprecated"):
        b.add_many(A, B, P, tracks=tracks)
    return b


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("prec", ["_double", ""])
@pytest.mark.parametrize("K", [1, 3])
@pytest.mark.parametrize("grad_mode", ["hard", "soft", "none"])
def test_constructor_is_add_many(case, prec, K, grad_mode):
    rng = np.random.default_rng(7)
    A, B, P, tracks = problem(rng, 120, K)
    kw = dict(n_threads=3, gap_model=case[0], mode=case[1], grad_mode=grad_mode)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)   # the new form never warns
        new = BATCH[prec](A, B, P, tracks=tracks, **kw)
    old = legacy(BATCH[prec], A, B, P, tracks=tracks, **kw)
    assert len(new) == len(old) == len(A)
    assert (new.gap_model, new.mode, new.grad_mode, new.n_threads) == (case[0], case[1], grad_mode, 3)
    assert new.tracked == old.tracked == (K > 1)
    s_new, s_old = new.score_and_grad(), old.score_and_grad()
    if grad_mode == "soft":
        np.testing.assert_allclose(new.scores(), old.scores(), rtol=1e-12)
    else:
        assert s_new == s_old
        np.testing.assert_array_equal(new.scores(), old.scores())
    if grad_mode != "none":
        for x, y in zip(new.grads(), old.grads()):
            np.testing.assert_allclose(x, y, rtol=1e-12, atol=1e-12)


def test_constructor_keywords_and_settings():
    rng = np.random.default_rng(8)
    A, B, P, _ = problem(rng, 50, 4)
    flat = np.concatenate([np.full(len(a), 3, dtype=np.uint16) for a in A])
    b = nwgrad.SeqPairBatchDouble(A, B, P, traceback="pointers", kernel="auto", tracks=flat,
                                  gap_model="linear", mode="local")
    assert b.tracked and b.num_sets == 4 and b[0].track == [3] * len(A[0])
    assert b.traceback == "pointers"
    b.fill = "striped"            # settings stay settable after construction
    b.track_storage = "counts"
    b.score_and_grad()
    G = b.weighted_grad(np.ones(len(A)))
    assert np.count_nonzero(G.matrices[:3]) == 0   # every residue on set 3
    with pytest.raises(TypeError):
        nwgrad.SeqPairBatch(A, B, P, 4)   # n_threads and the rest are keyword-only here
    empty = nwgrad.SeqPairBatchDouble([], [], P)
    assert len(empty) == 0


def test_constructor_errors():
    P = nwgrad.AlignParams.from_arrays(np.zeros((2, 4, 4)), np.ones((2, 4)), ALPHA)
    with pytest.raises(ValueError, match="not in alphabet"):
        nwgrad.SeqPairBatch(["ACXG"], ["AC"], P)
    with pytest.raises(ValueError, match="equal length"):
        nwgrad.SeqPairBatch(["AC"], ["AC", "G"], P)
    with pytest.raises(ValueError, match="is not a parameter set"):
        nwgrad.SeqPairBatch(["AC"], ["AC"], P, tracks=[[0, 2]])
    with pytest.raises(ValueError, match="unknown gap_model"):
        nwgrad.SeqPairBatch(["AC"], ["AC"], P, gap_model="convex")


def test_pairless_form_and_add_many_compat():
    rng = np.random.default_rng(9)
    A, B, P, _ = problem(rng, 40, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        streamer = nwgrad.SeqPairBatchDouble(2, gap_model="affine", mode="global")
        res = streamer.align(A, B, P)
        assert len(streamer) == 0 and len(res.scores) == len(A)
    full = nwgrad.SeqPairBatchDouble(A, B, P, n_threads=2)
    full.score_and_grad()
    np.testing.assert_array_equal(full.scores(), res.scores)
    # add_many still appends, to a constructed batch too (a second segment)
    with pytest.warns(DeprecationWarning, match=r"SeqPairBatch\(seqs_a, seqs_b, params"):
        full.add_many(A[:5], B[:5], P)
    assert len(full) == len(A) + 5


@pytest.mark.parametrize("dtype", [None, object])
def test_numpy_string_arrays(dtype):
    """A numpy string array yields a new str object per item; the sequences must stay
    alive for the whole call (they were read through dangling views before)."""
    rng = np.random.default_rng(10)
    A, B, P, _ = problem(rng, 3000, 1)
    A = [a + "ACGT" * 5 for a in A]   # long enough that a freed buffer gets reused
    B = [b + "TGCA" * 5 for b in B]
    ref = nwgrad.SeqPairBatchDouble(A, B, P, n_threads=2)
    ref.score_and_grad()
    na, nb = np.array(A, dtype=dtype), np.array(B, dtype=dtype)
    got = nwgrad.SeqPairBatchDouble(na, nb, P, n_threads=2)
    got.score_and_grad()
    np.testing.assert_array_equal(got.scores(), ref.scores())
    res = nwgrad.SeqPairBatchDouble(2).align(na, nb, P)
    np.testing.assert_array_equal(res.scores, ref.scores())
    with pytest.warns(DeprecationWarning):
        legacy = nwgrad.SeqPairBatchDouble(2)
        legacy.add_many(na, nb, P)
    legacy.score_and_grad()
    np.testing.assert_array_equal(legacy.scores(), ref.scores())
    for bad in ("ACGT", [b"AC"], ["AC", 3]):
        with pytest.raises(TypeError):
            nwgrad.SeqPairBatchDouble(bad, ["AC"] * len(bad), P)
