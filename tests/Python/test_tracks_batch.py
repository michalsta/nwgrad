"""Tracks in SeqPairBatch / SeqPair: storage, every fill, gradients stored as paths.

Each pair of a tracked batch must give exactly what the single-pair functions give for
that pair (themselves checked against an exhaustive oracle in test_tracks.py), under
every fill — the inter-pair fill reads each lane's own rows — and every traceback; the
sums (weighted_grad, compute_grad) must be the per-pair sums, and a batch whose sets
are all equal must reproduce the untracked batch bit for bit.
"""

import numpy as np
import pytest

import nwgrad

ALPHA = "ACGT"
N = len(ALPHA)

FUNCS = {
    ("linear", "global"): "nw_grad",
    ("linear", "local"): "sw_grad",
    ("affine", "global"): "nw_affine_grad",
    ("affine", "local"): "sw_affine_grad",
}
CASES = list(FUNCS)
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
PAIR = {"_double": nwgrad.SeqPairDouble, "": nwgrad.SeqPair}


def to_params(mats, gaps):
    return nwgrad.AlignParams.from_arrays(np.ascontiguousarray(mats), np.ascontiguousarray(gaps), ALPHA)


def random_problem(rng, n_pairs, K, maxlen=30, integral=False, equal=False, mirna=False):
    if mirna:   # PosimiR-shaped: A ~ 18..25, B ~ 50, one set per position
        la = rng.integers(18, 26, size=n_pairs)
        lb = rng.integers(45, 56, size=n_pairs)
    else:
        la = rng.integers(0, maxlen, size=n_pairs)
        lb = rng.integers(0, maxlen, size=n_pairs)
    A = ["".join(rng.choice(list(ALPHA), size=int(m))) for m in la]
    B = ["".join(rng.choice(list(ALPHA), size=int(n))) for n in lb]
    if integral:
        mats = rng.integers(-3, 4, size=(K, N, N)).astype(float)
        gaps = rng.integers(1, 4, size=(K, 4)).astype(float)
    else:
        mats = rng.normal(0, 1.5, size=(K, N, N))
        gaps = rng.uniform(0.2, 2.0, size=(K, 4))
    if equal:
        mats[:] = mats[0]
        gaps[:] = gaps[0]
    if mirna:
        tracks = [list(np.minimum(np.arange(len(a)), K - 1)) for a in A]
    else:
        tracks = [list(rng.integers(0, K, size=len(a))) for a in A]
    gtracks = [list(rng.integers(0, K, size=len(a) + 1)) for a in A]
    return A, B, mats, gaps, tracks, gtracks


def per_pair(case, prec, A, B, P, tracks, gtracks):
    f = getattr(nwgrad, FUNCS[case] + prec)
    out = []
    for k, (a, b) in enumerate(zip(A, B)):
        s, g = f(a, b, P, track=tracks[k], gap_track=gtracks[k] if gtracks else None)
        out.append((s, g.matrices, g.gaps))
    return out


def make_batch(case, prec, A, B, P, tracks, gtracks, fill="interpair", traceback="auto",
               grad_mode="hard", threads=4):
    gm, mode = case
    batch = BATCH[prec](A, B, P,
                        n_threads=threads, traceback=traceback, gap_model=gm, mode=mode, grad_mode=grad_mode, tracks=tracks, gap_tracks=gtracks)
    batch.fill = fill
    return batch


FILLS = ["interpair", "striped", "rowwise"]


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("prec", ["_double", ""])
@pytest.mark.parametrize("fill", FILLS)
@pytest.mark.parametrize("mirna", [False, True])
def test_batch_matches_per_pair(case, prec, fill, mirna):
    rng = np.random.default_rng(31 + mirna)
    K = 25 if mirna else 3
    A, B, mats, gaps, tracks, gtracks = random_problem(rng, 120, K, mirna=mirna)
    P = to_params(mats, gaps)
    use_g = None if mirna else gtracks
    ref = per_pair(case, prec, A, B, P, tracks, use_g)
    batch = make_batch(case, prec, A, B, P, tracks, use_g, fill=fill)
    assert batch.tracked and batch.num_sets == K
    batch.score_and_grad()
    scores = batch.scores()
    gm4, gg3 = batch.grads()
    assert gm4.shape == (len(A), K, N, N) and gg3.shape == (len(A), K, 4)
    for k, (s, m, g) in enumerate(ref):
        assert scores[k] == s, k
        np.testing.assert_array_equal(gm4[k], m)
        np.testing.assert_array_equal(gg3[k], g)
        pk = batch[k].grad
        np.testing.assert_array_equal(pk.matrices, m)
        np.testing.assert_array_equal(pk.gaps, g)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("traceback", ["pointers", "scores", "hirschberg"])
def test_batch_tracebacks_agree(case, traceback):
    rng = np.random.default_rng(5)
    A, B, mats, gaps, tracks, gtracks = random_problem(rng, 80, 4)
    P = to_params(mats, gaps)
    ref = make_batch(case, "_double", A, B, P, tracks, gtracks, fill="striped", traceback="pointers")
    ref.score_and_grad()
    for fill in FILLS:
        b = make_batch(case, "_double", A, B, P, tracks, gtracks, fill=fill, traceback=traceback)
        b.score_and_grad()
        np.testing.assert_array_equal(b.scores(), ref.scores())
        for x, y in zip(b.grads(), ref.grads()):
            np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("prec", ["_double", ""])
def test_weighted_grad_is_the_weighted_sum_of_pair_gradients(case, prec):
    rng = np.random.default_rng(8)
    A, B, mats, gaps, tracks, gtracks = random_problem(rng, 300, 5)
    P = to_params(mats, gaps)
    batch = make_batch(case, prec, A, B, P, tracks, gtracks)
    batch.score_and_grad()
    gm, gg = batch.grads()
    w = rng.normal(size=len(A))
    acc_m = np.zeros(gm.shape[1:])
    acc_g = np.zeros(gg.shape[1:])
    for k in range(len(A)):            # one block (< 4096 pairs): the C++ order exactly
        acc_m += w[k] * gm[k]
        acc_g += w[k] * gg[k]
    G = batch.weighted_grad(w)
    assert G.num_sets == 5
    np.testing.assert_array_equal(G.matrices, acc_m)
    np.testing.assert_array_equal(G.gaps, acc_g)
    C = batch.compute_grad()
    np.testing.assert_array_equal(C.matrices, gm.sum(axis=0))
    np.testing.assert_array_equal(C.gaps, gg.sum(axis=0))
    # reproducible whatever the thread count
    b1 = make_batch(case, prec, A, B, P, tracks, gtracks, threads=1)
    b1.score_and_grad()
    G1 = b1.weighted_grad(w)
    np.testing.assert_array_equal(G1.matrices, G.matrices)
    np.testing.assert_array_equal(G1.gaps, G.gaps)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("prec", ["_double", ""])
@pytest.mark.parametrize("fill", FILLS)
def test_equal_sets_batch_is_the_untracked_batch(case, prec, fill):
    rng = np.random.default_rng(12)
    A, B, mats, gaps, tracks, gtracks = random_problem(rng, 200, 4, integral=True, equal=True)
    PK = to_params(mats, gaps)
    P1 = PK[0]
    bt = make_batch(case, prec, A, B, PK, tracks, gtracks, fill=fill)
    bu = BATCH[prec](A, B, P1, n_threads=4, gap_model=case[0], mode=case[1])
    bu.fill = fill
    bt.score_and_grad(keep_paths=True)
    bu.score_and_grad(keep_paths=True)
    np.testing.assert_array_equal(bt.scores(), bu.scores())
    # per-pair counts are integers: summed over the sets they are the untracked ones
    # exactly (a weighted sum per set, summed over the sets, is another association)
    gmt, ggt = bt.grads()
    gmu, ggu = bu.grads()
    np.testing.assert_array_equal(gmt.sum(axis=1), gmu)
    np.testing.assert_array_equal(ggt.sum(axis=1), ggu)
    for k in range(0, len(A), 17):
        assert bt[k].aligned() == bu[k].aligned()
        assert bt[k].guide_j == bu[k].guide_j


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("prec", ["_double", ""])
def test_banded_grad_fills_agree_and_wide_band_is_full(case, prec):
    rng = np.random.default_rng(13)
    A, B, mats, gaps, tracks, gtracks = random_problem(rng, 150, 4)
    P = to_params(mats, gaps)
    P2 = to_params(mats + rng.normal(0, 0.3, size=mats.shape), gaps * 1.1)
    res = []
    for fill in ["interpair", "striped"]:
        b = make_batch(case, prec, A, B, P, tracks, gtracks, fill=fill)
        b.score_and_grad()
        b.set_params(P2)
        b.banded_grad(3)
        res.append((b.scores(), b.grads()))
        # a band past every B is the full DP
        b.banded_grad(64)
        s_wide = b.scores()
        f = make_batch(case, prec, A, B, P2, tracks, gtracks, fill=fill)
        f.score_and_grad()
        np.testing.assert_array_equal(s_wide, f.scores())
    np.testing.assert_array_equal(res[0][0], res[1][0])
    for x, y in zip(res[0][1], res[1][1]):
        np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("prec", ["_double", ""])
@pytest.mark.parametrize("fill", FILLS)
def test_score_only_batch(case, prec, fill):
    rng = np.random.default_rng(14)
    A, B, mats, gaps, tracks, gtracks = random_problem(rng, 150, 3)
    P = to_params(mats, gaps)
    hard = make_batch(case, prec, A, B, P, tracks, gtracks, fill=fill, traceback="pointers")
    none = make_batch(case, prec, A, B, P, tracks, gtracks, fill=fill, grad_mode="none")
    hard.score_and_grad()
    none.score_and_grad()
    np.testing.assert_array_equal(none.scores(), hard.scores())
    none.banded_grad(4)   # pending guides resolve under the params scored
    hard.banded_grad(4)
    np.testing.assert_array_equal(none.scores(), hard.scores())


def test_tracked_seq_pair_and_views():
    rng = np.random.default_rng(15)
    mats, gaps = rng.normal(size=(3, N, N)), rng.uniform(0.5, 1.5, size=(3, 4))
    P = to_params(mats, gaps)
    sp = nwgrad.SeqPairDouble("ACGTAC", "ACTTAGC", P, track=[0, 1, 2, 2, 1, 0])
    s, g = sp.score_and_grad()
    s2, g2 = nwgrad.nw_affine_grad_double("ACGTAC", "ACTTAGC", P, track=[0, 1, 2, 2, 1, 0])
    assert s == s2
    np.testing.assert_array_equal(g.matrices, g2.matrices)
    np.testing.assert_array_equal(g.gaps, g2.gaps)
    assert sp.track == [0, 1, 2, 2, 1, 0]
    assert sp.gap_track == [0, 0, 1, 2, 2, 1, 0]
    a, b = sp.aligned()
    assert a.replace("-", "") == "ACGTAC"
    # a one-set pair is not tracked
    assert nwgrad.SeqPairDouble("AC", "AC", P[0]).track is None
    batch = nwgrad.SeqPairBatchDouble(["ACGT", "GG"], ["ACT", "GGA"], P,
                                      n_threads=2, gap_model="affine", mode="global", tracks=[[0, 1, 2, 0], None], gap_tracks=[None, [2, 2, 2]])
    assert batch[0].track == [0, 1, 2, 0] and batch[0].gap_track == [0, 0, 1, 2, 0]
    assert batch[1].track == [0, 0] and batch[1].gap_track == [2, 2, 2]


def test_tracked_batch_validation():
    P3 = to_params(np.zeros((3, N, N)), np.ones((3, 4)))
    P2 = to_params(np.zeros((2, N, N)), np.ones((2, 4)))
    kw = dict(n_threads=2, gap_model="linear", mode="local")
    make = nwgrad.SeqPairBatchDouble
    with pytest.raises(ValueError, match="one entry per pair"):
        make(["AC"], ["AC"], P3, tracks=[[0, 1], [0]], **kw)
    with pytest.raises(ValueError, match="pair 0: track has 1 entries"):
        make(["AC"], ["AC"], P3, tracks=[[0]], **kw)
    with pytest.raises(ValueError, match="is not a parameter set"):
        make(["AC"], ["AC"], P3, tracks=[[0, 3]], **kw)
    with pytest.raises(ValueError, match="gap_track has 2 entries"):
        make(["AC"], ["AC"], P3, gap_tracks=[[0, 1]], **kw)
    b = make(["AC"], ["AC"], P3, tracks=[[0, 2]], **kw)
    with pytest.raises(ValueError, match="number of parameter sets"):
        b.set_params(P2)
    b.set_params(P3 * 2.0)


@pytest.mark.filterwarnings("ignore:SeqPairBatch.add_many")   # appends: deprecated, removed in 0.8
def test_tracked_batch_append_keeps_its_set_count():
    P3 = to_params(np.zeros((3, N, N)), np.ones((3, 4)))
    P2 = to_params(np.zeros((2, N, N)), np.ones((2, 4)))
    b = nwgrad.SeqPairBatchDouble(["AC"], ["AC"], P3, n_threads=2, gap_model="linear", mode="local",
                                  tracks=[[0, 2]])
    with pytest.raises(ValueError, match="parameter sets cannot join"):
        b.add_many(["AC"], ["AC"], P2)
    assert len(b) == 1


@pytest.mark.filterwarnings("ignore:SeqPairBatch.add_many")   # appends: deprecated, removed in 0.8
def test_untracked_batch_becomes_tracked():
    rng = np.random.default_rng(16)
    mats, gaps = rng.normal(size=(1, N, N)), rng.uniform(0.5, 1.5, size=(1, 4))
    P = to_params(mats, gaps)
    b = BATCH["_double"](2, gap_model="affine", mode="local")
    b.add_many(["ACGT", "AAC"], ["ACGG", "AC"], P)
    b.score_and_grad()
    s_before = b.scores()
    assert not b.tracked
    b.add_many(["GGT"], ["GCT"], P, tracks=[[0, 0, 0]])
    assert b.tracked and b.num_sets == 1
    with pytest.raises(RuntimeError):   # the earlier results were dropped with the switch
        b.scores()
    b.score_and_grad()
    np.testing.assert_array_equal(b.scores()[:2], s_before)
    assert b.grads()[0].shape == (3, 1, N, N)



@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("traceback", ["pointers", "hirschberg"])
def test_long_gap_runs_replay(case, traceback):
    """Stored paths are replayed by runs of one move, read 28 at a time: very unequal
    lengths force runs far longer than that (gaps in A along a row, gaps in B down many
    rows), which must give the pair's own gradient."""
    rng = np.random.default_rng(23)
    K = 4
    A, B = [], []
    for _ in range(40):
        short, long_ = int(rng.integers(0, 6)), int(rng.integers(29, 160))
        la, lb = (short, long_) if rng.integers(0, 2) else (long_, short)
        A.append("".join(rng.choice(list(ALPHA), size=la)))
        B.append("".join(rng.choice(list(ALPHA), size=lb)))
    mats = rng.normal(0, 1.5, size=(K, N, N))
    gaps = rng.uniform(0.2, 2.0, size=(K, 4))
    tracks = [list(rng.integers(0, K, size=len(a))) for a in A]
    gtracks = [list(rng.integers(0, K, size=len(a) + 1)) for a in A]
    P = to_params(mats, gaps)
    batch = make_batch(case, "_double", A, B, P, tracks, gtracks, traceback=traceback)
    batch.score_and_grad()
    gm, gg = batch.grads()
    for k, (s, m_, g_) in enumerate(per_pair(case, "_double", A, B, P, tracks, gtracks)):
        assert batch.scores()[k] == s
        np.testing.assert_array_equal(gm[k], m_)
        np.testing.assert_array_equal(gg[k], g_)
    w = rng.normal(size=len(A))
    G = batch.weighted_grad(w)
    acc_m, acc_g = np.zeros(gm.shape[1:]), np.zeros(gg.shape[1:])
    for k in range(len(A)):
        acc_m += w[k] * gm[k]
        acc_g += w[k] * gg[k]
    np.testing.assert_array_equal(G.matrices, acc_m)
    np.testing.assert_array_equal(G.gaps, acc_g)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("prec", ["_double", ""])
def test_one_set_tracked_weighted_grad_is_untracked_bit_for_bit(case, prec):
    """With one set, the tracked sum (each pair's touched fields, weight * count, in pair
    order, fixed blocks) is the untracked dense sum to the bit — more than 4096 pairs, so
    several blocks are combined."""
    rng = np.random.default_rng(17)
    A, B, mats, gaps, tracks, gtracks = random_problem(rng, 9000, 1, maxlen=12)
    P = to_params(mats, gaps)
    bt = make_batch(case, prec, A, B, P, tracks, None, threads=8)
    bu = BATCH[prec](A, B, P, n_threads=8, gap_model=case[0], mode=case[1])
    assert bt.tracked and not bu.tracked
    bt.score_and_grad()
    bu.score_and_grad()
    np.testing.assert_array_equal(bt.scores(), bu.scores())
    w = rng.normal(size=len(A))
    Gt, Gu = bt.weighted_grad(w), bu.weighted_grad(w)
    np.testing.assert_array_equal(Gt.matrices, Gu.matrices)
    np.testing.assert_array_equal(Gt.gaps, Gu.gaps)
    Ct, Cu = bt.compute_grad(), bu.compute_grad()
    np.testing.assert_array_equal(Ct.matrices, Cu.matrices)
    np.testing.assert_array_equal(Ct.gaps, Cu.gaps)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("prec", ["_double", ""])
@pytest.mark.parametrize("traceback", ["hirschberg", "hirschberg_pmax"])
def test_tracked_hirschberg_splits_and_stays_optimal(case, prec, traceback):
    """hb_cutoff 2: the recursion really splits (with tracks the reverse sweeps run on
    rewritten rows).  The path's score is the optimum, and the gradient is that path's."""
    rng = np.random.default_rng(19)
    A, B, mats, gaps, tracks, gtracks = random_problem(rng, 120, 4, maxlen=40)
    P = to_params(mats, gaps)
    ref = make_batch(case, prec, A, B, P, tracks, gtracks, fill="striped", traceback="pointers")
    hb = make_batch(case, prec, A, B, P, tracks, gtracks, fill="striped", traceback=traceback)
    hb.hb_cutoff = 2
    ref.score_and_grad()
    hb.score_and_grad()
    tol = 1e-9 if prec == "_double" else 2e-3
    exact = traceback == "hirschberg" or case[0] == "linear"
    np.testing.assert_array_less(hb.scores(), ref.scores() + tol)
    np.testing.assert_array_less(ref.scores() - (tol if exact else 100 * tol), hb.scores() + 1e-12)
    gm, gg = hb.grads()
    dots = (gm * mats).sum(axis=(1, 2, 3)) + (gg * gaps).sum(axis=(1, 2))
    np.testing.assert_allclose(dots, hb.scores(), atol=tol * 50)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("prec", ["_double", ""])
@pytest.mark.parametrize("fill", ["interpair", "striped"])
@pytest.mark.parametrize("grad_mode", ["hard", "soft", "none"])
def test_streaming_align_with_tracks(case, prec, fill, grad_mode):
    """align() (nothing stored per pair) agrees with the batch's own pairs: the same
    scores, and the gradient summed over the problems."""
    rng = np.random.default_rng(23)
    A, B, mats, gaps, tracks, gtracks = random_problem(rng, 150, 4, maxlen=30)
    P = to_params(mats, gaps)
    gtracks = [g if k % 3 else None for k, g in enumerate(gtracks)]
    stored = make_batch(case, prec, A, B, P, tracks, gtracks, fill=fill, grad_mode=grad_mode)
    stored.score_and_grad()
    streamer = BATCH[prec](4, gap_model=case[0], mode=case[1], grad_mode=grad_mode)
    streamer.fill = fill
    res = streamer.align(A, B, P, tracks=tracks, gap_tracks=gtracks)
    if grad_mode == "soft":
        np.testing.assert_allclose(res.scores, stored.scores(), rtol=1e-11, atol=1e-11)
    else:
        np.testing.assert_array_equal(res.scores, stored.scores())
    if grad_mode == "none":
        return
    C = stored.compute_grad()
    assert res.grad.num_sets == 4
    if grad_mode == "hard":
        np.testing.assert_array_equal(res.grad.matrices, C.matrices)
        np.testing.assert_array_equal(res.grad.gaps, C.gaps)
    else:
        np.testing.assert_allclose(res.grad.matrices, C.matrices, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(res.grad.gaps, C.gaps, rtol=1e-10, atol=1e-10)
    # banded streaming: a band past every B is the full DP
    resb = streamer.align(A, B, P, band=64, tracks=tracks, gap_tracks=gtracks)
    if grad_mode == "hard":
        np.testing.assert_array_equal(resb.scores, stored.scores())


def test_streaming_track_validation():
    P3 = to_params(np.zeros((3, N, N)), np.ones((3, 4)))
    s = nwgrad.SeqPairBatchDouble(2, gap_model="affine", mode="global")
    with pytest.raises(ValueError, match="pair 1: track has 1 entries"):
        s.align(["AC", "AC"], ["AC", "AC"], P3, tracks=[[0, 1], [0]])
    with pytest.raises(ValueError, match="is not a parameter set"):
        s.align(["AC"], ["AC"], P3, gap_tracks=[[0, 0, 7]])


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("prec", ["_double", ""])
def test_track_storage_switch(case, prec):
    """track_storage picks paths or counts for a tracked batch's hard gradients: the
    results are identical, and a switch drops the stored gradients (not the scores or
    guides) until the next fill.  An untracked batch keeps the setting and ignores it."""
    rng = np.random.default_rng(55)
    A, B, mats, gaps, tracks, gtracks = random_problem(rng, 300, 3)
    P = to_params(mats, gaps)
    w = rng.normal(size=len(A))
    batch = make_batch(case, prec, A, B, P, tracks, gtracks)
    other = {"paths": "counts", "counts": "paths"}
    first = batch.track_storage
    batch.score_and_grad()
    s0, (gm0, gg0), G0 = batch.scores(), batch.grads(), batch.weighted_grad(w)
    batch.track_storage = first   # no change: nothing dropped
    batch.weighted_grad(w)
    batch.track_storage = other[first]
    assert batch.track_storage == other[first]
    np.testing.assert_array_equal(batch.scores(), s0)
    with pytest.raises(RuntimeError, match="gradient not computed"):
        batch.weighted_grad(w)
    with pytest.raises(RuntimeError, match="gradient not computed"):
        batch.grads()
    batch.banded_grad(64)   # the guides survived the switch
    for storage in [other[first], first, other[first]]:
        batch.track_storage = storage
        batch.score_and_grad()
        for k in range(0, len(A), 11):   # single-pair recomputes, in this store
            batch[k].score_and_grad()
        np.testing.assert_array_equal(batch.scores(), s0)
        gm, gg = batch.grads()
        np.testing.assert_array_equal(gm, gm0)
        np.testing.assert_array_equal(gg, gg0)
        G = batch.weighted_grad(w)
        np.testing.assert_array_equal(G.matrices, G0.matrices)
        np.testing.assert_array_equal(G.gaps, G0.gaps)
    with pytest.raises(ValueError, match="unknown track_storage"):
        batch.track_storage = "dense"
    u = BATCH[prec](A[:20], B[:20], to_params(mats[:1], gaps[:1]),
                    n_threads=2, gap_model=case[0], mode=case[1])
    u.track_storage = other[first]
    assert not u.tracked and u.track_storage == other[first]
    u.score_and_grad()
    u.weighted_grad(w[:20])


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("dtype", [np.uint16, np.int32, np.int64, np.uint8])
def test_flat_track_arrays(case, dtype):
    """tracks / gap_tracks may be one flat 1-D integer array each, every pair's entries
    concatenated: the same batch, and the same streamed result, as per-pair lists."""
    rng = np.random.default_rng(91)
    A, B, mats, gaps, tracks, gtracks = random_problem(rng, 200, 4)
    P = to_params(mats, gaps)
    flat_t = np.concatenate([np.asarray(t, dtype=np.int64) for t in tracks]).astype(dtype)
    flat_g = np.concatenate([np.asarray(g, dtype=np.int64) for g in gtracks]).astype(dtype)
    nested = make_batch(case, "_double", A, B, P, tracks, gtracks)
    flat = make_batch(case, "_double", A, B, P, flat_t, flat_g)
    for b in (nested, flat):
        b.score_and_grad()
    np.testing.assert_array_equal(flat.scores(), nested.scores())
    for x, y in zip(flat.grads(), nested.grads()):
        np.testing.assert_array_equal(x, y)
    assert [flat[k].track for k in range(5)] == [list(map(int, t)) for t in tracks[:5]]
    # only the track flat, then a strided view
    half = make_batch(case, "_double", A, B, P, np.ascontiguousarray(np.repeat(flat_t, 2)[::2]), gtracks)
    half.score_and_grad()
    np.testing.assert_array_equal(half.scores(), nested.scores())
    s = BATCH["_double"](3, gap_model=case[0], mode=case[1])
    r_flat = s.align(A, B, P, tracks=np.repeat(flat_t, 2)[::2], gap_tracks=flat_g)
    r_nested = s.align(A, B, P, tracks=tracks, gap_tracks=gtracks)
    np.testing.assert_array_equal(r_flat.scores, r_nested.scores)
    np.testing.assert_array_equal(r_flat.grad.matrices, r_nested.grad.matrices)
    np.testing.assert_array_equal(r_flat.grad.gaps, r_nested.grad.gaps)


def test_flat_track_array_validation():
    P3 = to_params(np.zeros((3, N, N)), np.ones((3, 4)))
    def make(A, **tk):
        return BATCH["_double"](A, ["AC", "AC"], P3, n_threads=2, gap_model="linear", mode="global", **tk)
    with pytest.raises(ValueError, match="flat tracks has 4 entries, expected 5"):
        make(["AC", "GTA"], tracks=np.zeros(4, dtype=np.int32))
    with pytest.raises(ValueError, match="flat gap_tracks has 5 entries, expected 7"):
        make(["AC", "GTA"], gap_tracks=np.zeros(5, dtype=np.int32))
    with pytest.raises(ValueError, match=r"tracks\[3\] = -1 is not a parameter set"):
        make(["AC", "GTA"], tracks=np.array([0, 1, 2, -1, 0]))
    with pytest.raises(ValueError, match="is not a parameter set"):   # against K, by the batch
        make(["AC", "GTA"], tracks=np.array([0, 1, 2, 3, 0]))
    with pytest.raises(TypeError, match="integer dtype"):
        make(["AC", "GTA"], tracks=np.zeros(5))
    for big in (2**63, 2**64 - 1):   # read as -1 internally; the message names the value
        with pytest.raises(ValueError, match=rf"tracks\[3\] = {big} is not a parameter set"):
            make(["AC", "GTA"], tracks=np.array([0, 1, 2, big, 0], dtype=np.uint64))
    # a 2-D array is still one row per pair (all of A's lengths equal)
    b = make(["AC", "GT"], tracks=np.array([[0, 1], [2, 2]]))
    assert b[1].track == [2, 2]
