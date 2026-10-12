"""Tracks under every alphabet size the inter-pair kernels distinguish.

The inter-pair fills pick each row's scores from a profile of 1 << levels vectors (a
blend tree on B's bits: 1, 2, 3-4 and 5-8 letters) or, past 8 letters, gather them per
lane; the soft pass gathers past its own limit.  Each size: the tracked batch under
every fill against the single-pair functions (exact), equal sets against the untracked
batch, and soft batches against the single-pair soft functions.
"""

import numpy as np
import pytest

import nwgrad

ALPHABETS = ["A", "AC", "ACG", "ACGT", "ACGTN", "ACDEFGHI", "ACDEFGHIK", "ARNDCQEGHILKMFPSTWYV"]
CASES = [("linear", "global"), ("linear", "local"), ("affine", "global"), ("affine", "local")]
HARD = {("linear", "global"): "nw_grad", ("linear", "local"): "sw_grad",
        ("affine", "global"): "nw_affine_grad", ("affine", "local"): "sw_affine_grad"}
SOFT = {("linear", "global"): "nw_soft_grad", ("linear", "local"): "sw_soft_grad",
        ("affine", "global"): "nw_affine_soft_grad", ("affine", "local"): "sw_affine_soft_grad"}
STORAGE = ["paths"]   # the hard-gradient store of every batch built here (track_storage)


@pytest.fixture(autouse=True, params=["paths", "counts"])
def track_storage(request):
    """Every test here runs once per store of a tracked batch's hard gradients."""
    STORAGE[0] = request.param
    yield request.param
    STORAGE[0] = "paths"


def _batch(cls):
    def make(*args, **kwargs):
        b = cls(*args, **kwargs)
        b.track_storage = STORAGE[0]
        return b
    return make


BATCH = {"_double": _batch(nwgrad.SeqPairBatchDouble), "": _batch(nwgrad.SeqPairBatch)}
FILLS = ["interpair", "striped", "rowwise"]


def problem(rng, alpha, n_pairs, K, maxlen=28):
    na = len(alpha)
    A = ["".join(rng.choice(list(alpha), size=int(rng.integers(0, maxlen)))) for _ in range(n_pairs)]
    B = ["".join(rng.choice(list(alpha), size=int(rng.integers(0, maxlen)))) for _ in range(n_pairs)]
    mats = rng.normal(0, 1.5, size=(K, na, na))
    gaps = rng.uniform(0.2, 2.0, size=(K, 4))
    tracks = [list(rng.integers(0, K, size=len(a))) for a in A]
    gtracks = [list(rng.integers(0, K, size=len(a) + 1)) for a in A]
    P = nwgrad.AlignParams.from_arrays(mats, gaps, alpha)
    return A, B, P, tracks, gtracks


@pytest.mark.parametrize("alpha", ALPHABETS, ids=lambda a: f"n{len(a)}")
@pytest.mark.parametrize("case", CASES, ids=lambda c: "-".join(c))
@pytest.mark.parametrize("prec", ["_double", ""])
def test_hard_batch_matches_per_pair(alpha, case, prec):
    rng = np.random.default_rng(100 + len(alpha))
    K = 3
    A, B, P, tracks, gtracks = problem(rng, alpha, 70, K)
    f = getattr(nwgrad, HARD[case] + prec)
    ref = [f(a, b, P, track=tracks[k], gap_track=gtracks[k]) for k, (a, b) in enumerate(zip(A, B))]
    w = rng.normal(size=len(A))
    for fill in FILLS:
        for tb in ("pointers", "auto"):
            batch = BATCH[prec](4, tb, gap_model=case[0], mode=case[1], grad_mode="hard")
            batch.fill = fill
            batch.add_many(A, B, P, tracks=tracks, gap_tracks=gtracks)
            batch.score_and_grad()
            sc = batch.scores()
            gm, gg = batch.grads()
            for k, (s, g) in enumerate(ref):
                assert sc[k] == s, (fill, tb, k)
                np.testing.assert_array_equal(gm[k], g.matrices)
                np.testing.assert_array_equal(gg[k], g.gaps)
            G = batch.weighted_grad(w)
            acc_m, acc_g = np.zeros(gm.shape[1:]), np.zeros(gg.shape[1:])
            for k in range(len(A)):
                acc_m += w[k] * gm[k]
                acc_g += w[k] * gg[k]
            np.testing.assert_array_equal(G.matrices, acc_m)
            np.testing.assert_array_equal(G.gaps, acc_g)


@pytest.mark.parametrize("alpha", ALPHABETS, ids=lambda a: f"n{len(a)}")
@pytest.mark.parametrize("case", CASES, ids=lambda c: "-".join(c))
@pytest.mark.parametrize("prec", ["_double", ""])
def test_equal_sets_are_untracked(alpha, case, prec):
    rng = np.random.default_rng(200 + len(alpha))
    A, B, P1, _, _ = problem(rng, alpha, 90, 1)
    d = P1.to_dict()
    K = 3
    PK = nwgrad.AlignParams.from_arrays(np.repeat(P1.matrices, K, axis=0), np.repeat(P1.gaps, K, axis=0), alpha)
    tracks = [list(rng.integers(0, K, size=len(a))) for a in A]
    gtracks = [list(rng.integers(0, K, size=len(a) + 1)) for a in A]
    for fill in FILLS:
        bt = BATCH[prec](4, gap_model=case[0], mode=case[1])
        bt.fill = fill
        bt.add_many(A, B, PK, tracks=tracks, gap_tracks=gtracks)
        bu = BATCH[prec](4, gap_model=case[0], mode=case[1])
        bu.fill = fill
        bu.add_many(A, B, nwgrad.AlignParams(d["matrix"], alpha, d["gap_open_a"], d["gap_extend_a"],
                                             d["gap_open_b"], d["gap_extend_b"]))
        bt.score_and_grad()
        bu.score_and_grad()
        np.testing.assert_array_equal(bt.scores(), bu.scores())
        gmt, ggt = bt.grads()
        gmu, ggu = bu.grads()
        np.testing.assert_array_equal(gmt.sum(axis=1), gmu)
        np.testing.assert_array_equal(ggt.sum(axis=1), ggu)


@pytest.mark.parametrize("alpha", ALPHABETS, ids=lambda a: f"n{len(a)}")
@pytest.mark.parametrize("case", CASES, ids=lambda c: "-".join(c))
def test_soft_batch_matches_per_pair(alpha, case):
    rng = np.random.default_rng(300 + len(alpha))
    K = 3
    A, B, P, tracks, gtracks = problem(rng, alpha, 50, K, maxlen=22)
    f = getattr(nwgrad, SOFT[case] + "_double")
    for fill in ("interpair", "striped"):
        batch = nwgrad.SeqPairBatchDouble(4, gap_model=case[0], mode=case[1], grad_mode="soft")
        batch.fill = fill
        batch.soft_temperature = 0.8
        batch.add_many(A, B, P, tracks=tracks, gap_tracks=gtracks)
        batch.score_and_grad()
        sc = batch.scores()
        gm, gg = batch.grads()
        for k, (a, b) in enumerate(zip(A, B)):
            s, g = f(a, b, P, soft_impl="scaled", temperature=0.8, track=tracks[k], gap_track=gtracks[k])
            assert sc[k] == pytest.approx(s, rel=1e-11, abs=1e-11)
            np.testing.assert_allclose(gm[k], g.matrices, rtol=1e-9, atol=1e-11)
            np.testing.assert_allclose(gg[k], g.gaps, rtol=1e-9, atol=1e-11)
