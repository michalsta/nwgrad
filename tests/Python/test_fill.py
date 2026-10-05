"""fill="rowwise" and fill="interpair" must be indistinguishable from fill="striped".

Both are simd fills of the full affine DP that write tables bit-identical to the
scalar kernel; they differ only in speed.  Row-wise has no lazy-F fixpoint and wins
on short pairs, striped wins on long ones.  So the contract is identical results:
same score, alignment strings and gradient, ties included.

Integral scores make ties common, and a near-zero gap cost makes long gap runs
common (the case where lazy-F does the most work); both regimes are covered.
"""

import numpy as np
import pytest

import nwgrad

DNA = "ACGT"


def _params(kind):
    if kind == "ties":
        m = np.full((4, 4), -1.0)
        np.fill_diagonal(m, 2.0)
        go, ge = 3.0, 1.0
    elif kind == "cheap_gaps":
        m = np.random.default_rng(7).normal(scale=0.1, size=(4, 4))
        go, ge = 0.0, 1e-4
    else:
        m = np.random.default_rng(8).normal(size=(4, 4))
        go, ge = 2.5, 0.7
    return nwgrad.AlignParams(nwgrad.SubstMatrix(m, alphabet=DNA), gap_open_a=go,
                              gap_extend_a=ge, gap_open_b=go * 1.3, gap_extend_b=ge * 0.8)


def _seqs(n, lo, hi, seed):
    rng = np.random.default_rng(seed)
    return ["".join(rng.choice(list(DNA), int(l))) for l in rng.integers(lo, hi, n)]


# Batch and pair class per precision: float32 interpair (InterJobT<float>, twice the
# lanes) is held to the same bit-identity as double, against the float32 own fill.
_CLS = {"double": (nwgrad.SeqPairBatchDouble, nwgrad.SeqPairDouble),
        "float32": (nwgrad.SeqPairBatch, nwgrad.SeqPair)}


def _run(seqs_a, seqs_b, params, mode, traceback, fill, kernel="auto", prec="double"):
    bcls, pcls = _CLS[prec]
    b = bcls(n_threads=4, traceback=traceback)
    b.fill = fill
    b.add_many(seqs_a, seqs_b, params, gap_model="affine", mode=mode,
               grad_mode="hard", kernel=kernel)
    b.score_and_grad()
    mats, gaps = b.grads()
    aligned = []
    for i in range(0, len(seqs_a), 7):
        sp = pcls(seqs_a[i], seqs_b[i], params, gap_model="affine", mode=mode,
                  grad_mode="hard", kernel=kernel, traceback=traceback)
        sp.fill = fill if fill != "interpair" else "striped"
        sp.alloc_dp()
        sp.align_full()
        aligned.append((sp.score, sp.aligned()))
    return b.scores(), mats, gaps, aligned


def test_default_and_validation():
    b = nwgrad.SeqPairBatchDouble(n_threads=1)
    assert b.fill == "striped"
    b.fill = "rowwise"
    assert b.fill == "rowwise"
    b.fill = "interpair"
    assert b.fill == "interpair"
    with pytest.raises(ValueError, match="unknown fill"):
        b.fill = "diagonal"
    sp = nwgrad.SeqPairDouble("ACG", "ACG", _params("random"), gap_model="affine", mode="local")
    with pytest.raises(ValueError, match="unknown fill"):
        sp.fill = "interpair"


def _fixed_len_b(seqs, n, seed):
    rng = np.random.default_rng(seed)
    return ["".join(rng.choice(list(DNA), n)) for _ in seqs]


@pytest.mark.parametrize("prec,fill", [("double", "rowwise"), ("double", "interpair"),
                                       ("float32", "interpair")])
@pytest.mark.parametrize("kind", ["ties", "cheap_gaps", "random"])
@pytest.mark.parametrize("mode", ["local", "global"])
@pytest.mark.parametrize("traceback", ["pointers", "scores"])
@pytest.mark.parametrize("lengths", [(1, 30, 1, 60), (15, 30, 40, 60), (60, 200, 60, 200),
                                     (15, 30, 50, 51)])
def test_fill_matches_striped(prec, fill, kind, mode, traceback, lengths):
    # (15, 30, 50, 51): every B of length 50, so inter-pair groups are full, as in
    # miRNA x site data; the others mix lengths, so many groups are short.
    a = _seqs(301, lengths[0], lengths[1], 1)
    b = _seqs(301, lengths[2], lengths[3], 2)
    p = _params(kind)
    s0, m0, g0, al0 = _run(a, b, p, mode, traceback, "striped", prec=prec)
    s1, m1, g1, al1 = _run(a, b, p, mode, traceback, fill, prec=prec)
    assert np.array_equal(s0, s1)
    assert np.array_equal(m0, m1)
    assert np.array_equal(g0, g1)
    assert al0 == al1


@pytest.mark.parametrize("prec,fill", [("double", "rowwise"), ("double", "interpair"),
                                       ("float32", "interpair")])
@pytest.mark.parametrize("mode", ["local", "global"])
@pytest.mark.parametrize("level", [l for l in nwgrad.available_isa_levels()])
def test_every_isa_level(level, mode, prec, fill):
    a, b = _seqs(203, 10, 40, 3), _seqs(203, 30, 33, 4)
    p = _params("cheap_gaps")
    ref = _run(a, b, p, mode, "pointers", "striped", kernel="scalar_fallback", prec=prec)
    got = _run(a, b, p, mode, "pointers", fill, kernel=level, prec=prec)
    for x, y in zip(ref[:3], got[:3]):
        assert np.array_equal(x, y)
    assert ref[3] == got[3]


def test_interpair_falls_back_where_it_cannot_run():
    """A protein alphabet (over 8 letters) and a linear-gap pair in the same batch."""
    AA = "ARNDCQEGHILKMFPSTWYV"
    rng = np.random.default_rng(9)
    pa = [ "".join(rng.choice(list(AA), int(l))) for l in rng.integers(5, 40, 50)]
    pb = [ "".join(rng.choice(list(AA), int(l))) for l in rng.integers(5, 40, 50)]
    m = rng.normal(size=(20, 20))
    p = nwgrad.AlignParams(nwgrad.SubstMatrix(m, alphabet=AA), gap_open_a=2.0,
                           gap_extend_a=0.5, gap_open_b=2.0, gap_extend_b=0.5)
    out = []
    for fill in ("striped", "interpair"):
        batch = nwgrad.SeqPairBatchDouble(n_threads=3, traceback="pointers")
        batch.fill = fill
        batch.add_many(pa, pb, p, gap_model="affine", mode="local")
        batch.add_many(pa, pb, p, gap_model="linear", mode="global")
        batch.score_and_grad()
        out.append((batch.scores(), *batch.grads()))
    for x, y in zip(*out):
        assert np.array_equal(x, y)


def test_interpair_mixed_models_in_one_batch():
    a, b = _seqs(97, 10, 30, 11), _fixed_len_b(range(97), 50, 12)
    p = _params("random")
    out = []
    for fill in ("striped", "interpair"):
        batch = nwgrad.SeqPairBatchDouble(n_threads=3, traceback="pointers")
        batch.fill = fill
        batch.add_many(a, b, p, gap_model="affine", mode="local")
        batch.add_many(a, b, p, gap_model="affine", mode="global")
        batch.add_many(a, b, p, gap_model="linear", mode="local")
        batch.score_and_grad()
        out.append((batch.scores(), *batch.grads()))
    for x, y in zip(*out):
        assert np.array_equal(x, y)


def test_setting_fill_applies_to_existing_pairs():
    a, b = _seqs(50, 10, 30, 5), _seqs(50, 30, 60, 6)
    batch = nwgrad.SeqPairBatchDouble(n_threads=2, traceback="pointers")
    batch.add_many(a, b, _params("random"), gap_model="affine", mode="local")
    assert batch[0].fill == "striped"
    batch.fill = "rowwise"
    assert all(batch[i].fill == "rowwise" for i in range(len(batch)))


def test_float32_default_batch_interpair():
    """The Python default: a bare SeqPairBatch (float32, traceback="auto" — which is
    hirschberg_pmax for affine Global, run as Pointers below hb_cutoff)."""
    a, b = _seqs(301, 15, 30, 21), _fixed_len_b(range(301), 50, 22)
    for mode in ("local", "global"):
        for gm in ("affine", "linear"):
            out = []
            for fill in ("striped", "interpair"):
                batch = nwgrad.SeqPairBatch(n_threads=3)
                batch.fill = fill
                batch.add_many(a, b, _params("ties"), gap_model=gm, mode=mode)
                batch.score_and_grad()
                out.append((batch.scores(), *batch.grads(),
                            [list(batch[i].guide_j) for i in range(0, 301, 5)]))
            for x, y in zip(out[0][:3], out[1][:3]):
                assert np.array_equal(x, y)
            assert out[0][3] == out[1][3]


def test_float32_and_linear_ignore_it():
    a, b = _seqs(100, 10, 30, 7), _seqs(100, 30, 60, 8)
    p = _params("random")
    for cls, gm in ((nwgrad.SeqPairBatch, "affine"), (nwgrad.SeqPairBatchDouble, "linear")):
        out = []
        for fill in ("striped", "rowwise"):
            batch = cls(n_threads=2, traceback="pointers")
            batch.fill = fill
            batch.add_many(a, b, p, gap_model=gm, mode="local")
            batch.score_and_grad()
            out.append((batch.scores(), *batch.grads()))
        for x, y in zip(*out):
            assert np.array_equal(x, y)


def test_interpair_plan_follows_set_params_and_add_many():
    """The grouping is cached across calls: it must survive set_params() and be
    rebuilt when pairs are added."""
    a, b = _seqs(150, 10, 30, 13), _fixed_len_b(range(150), 40, 14)
    a2, b2 = _seqs(37, 5, 25, 15), _fixed_len_b(range(37), 40, 16)
    p1, p2 = _params("random"), _params("cheap_gaps")
    res = {}
    for fill in ("striped", "interpair"):
        batch = nwgrad.SeqPairBatchDouble(n_threads=3, traceback="pointers")
        batch.fill = fill
        batch.add_many(a, b, p1, gap_model="affine", mode="local")
        out = []
        batch.score_and_grad(); out.append((batch.scores(), *batch.grads()))
        batch.set_params(p2)
        batch.score_and_grad(); out.append((batch.scores(), *batch.grads()))
        batch.add_many(a2, b2, p2, gap_model="affine", mode="local")
        batch.score_and_grad(); out.append((batch.scores(), *batch.grads()))
        res[fill] = out
    for x, y in zip(res["striped"], res["interpair"]):
        for u, v in zip(x, y):
            assert np.array_equal(u, v)


@pytest.mark.parametrize("traceback", ["hirschberg", "hirschberg_pmax"])
@pytest.mark.parametrize("threads", [1, 3])
@pytest.mark.parametrize("initial_cutoff", [1, 512])
def test_interpair_plan_follows_pair_cutoff_changes(traceback, threads, initial_cutoff):
    """A cached plan must honor cutoff changes on any lane, in both directions."""
    p = nwgrad.AlignParams(np.eye(4) * 3 - 1, alphabet=DNA,
                          gap_open_a=2, gap_extend_a=.5,
                          gap_open_b=2, gap_extend_b=.5)
    a = ["AGGGTTGGACTTACCGACCATGATGAGCCC"] * 17
    b = ["TATATTATACGGAACTCGATTCTCCCATAC"] * 17

    def make(fill, cutoffs):
        batch = nwgrad.SeqPairBatchDouble(n_threads=threads, traceback=traceback)
        batch.fill = fill
        batch.add_many(a, b, p)
        for i, cutoff in enumerate(cutoffs):
            batch[i].hb_cutoff = cutoff
        batch.score_and_grad()
        return batch

    cutoffs = [initial_cutoff] * len(a)
    batch = make("interpair", cutoffs)
    original = batch.grads()
    # Index 1 catches checks limited to a group's first pair; the last index
    # also exercises a partially populated vector group.
    for cutoff in (513 - initial_cutoff, initial_cutoff):
        for i in (1, len(a) - 1):
            cutoffs[i] = cutoff
            batch[i].hb_cutoff = cutoff
        batch.score_and_grad()
        for fill in ("striped", "interpair"):
            ref = make(fill, cutoffs)
            np.testing.assert_array_equal(batch.scores(), ref.scores())
            for got, expected in zip(batch.grads(), ref.grads()):
                np.testing.assert_array_equal(got, expected)
        if cutoff != initial_cutoff:
            # This fixture must expose the changed tie-break; otherwise a stale
            # plan could pass the comparisons without ever taking Hirschberg.
            assert not np.array_equal(batch.grads()[1][1], original[1][1])


@pytest.mark.parametrize("fill", ["rowwise", "interpair"])
@pytest.mark.parametrize("mode", ["local", "global"])
def test_guides_match_striped(fill, mode):
    """score_and_grad() caches each pair's path as the guide for banded_grad(); the
    row-wise and inter-pair paths take it from the gradient's own traceback walk.  A
    narrow band around a different guide would score differently."""
    a, b = _seqs(120, 10, 30, 17), _fixed_len_b(range(120), 45, 18)
    p1, p2 = _params("cheap_gaps"), _params("random")
    out = []
    for f in ("striped", fill):
        batch = nwgrad.SeqPairBatchDouble(n_threads=3, traceback="pointers")
        batch.fill = f
        batch.add_many(a, b, p1, gap_model="affine", mode=mode)
        batch.score_and_grad()
        batch.set_params(p2)
        batch.banded_grad(2)
        out.append((batch.scores(), *batch.grads()))
    for x, y in zip(*out):
        assert np.array_equal(x, y)


# ── Linear gaps under fill="interpair" (InterJob.linear) ─────────────────────
# Each lane must write H bit-identical to the pair's own scalar viterbi_linear, so the
# score, the traceback-derived guide and the hard gradient match striped bit for bit,
# ties included (the "ties" fixture makes them common).

def _run_linear(seqs_a, seqs_b, params, mode, fill, grad_mode="hard", kernel="auto",
                prec="double"):
    b = _CLS[prec][0](n_threads=4, traceback="pointers")
    b.fill = fill
    b.add_many(seqs_a, seqs_b, params, gap_model="linear", mode=mode,
               grad_mode=grad_mode, kernel=kernel)
    b.score_and_grad()
    mats, gaps = b.grads()
    guides = [list(b[i].guide_j) for i in range(len(b))]
    return b.scores(), mats, gaps, guides


@pytest.mark.parametrize("prec", ["double", "float32"])
@pytest.mark.parametrize("kind", ["ties", "cheap_gaps", "random"])
@pytest.mark.parametrize("mode", ["local", "global"])
@pytest.mark.parametrize("lengths", [(1, 30, 1, 60), (15, 30, 50, 51), (60, 200, 60, 200)])
def test_linear_interpair_matches_striped(kind, mode, lengths, prec):
    a = _seqs(301, lengths[0], lengths[1], 11)
    b = _seqs(301, lengths[2], lengths[3], 12)
    p = _params(kind)
    r0 = _run_linear(a, b, p, mode, "striped", prec=prec)
    r1 = _run_linear(a, b, p, mode, "interpair", prec=prec)
    for x, y in zip(r0[:3], r1[:3]):
        assert np.array_equal(x, y)
    assert r0[3] == r1[3]


@pytest.mark.parametrize("prec", ["double", "float32"])
@pytest.mark.parametrize("mode", ["local", "global"])
@pytest.mark.parametrize("level", [l for l in nwgrad.available_isa_levels()])
def test_linear_interpair_every_isa_level(level, mode, prec):
    a, b = _seqs(203, 10, 40, 13), _seqs(203, 30, 33, 14)
    p = _params("ties")
    ref = _run_linear(a, b, p, mode, "striped", kernel="scalar_fallback", prec=prec)
    got = _run_linear(a, b, p, mode, "interpair", kernel=level, prec=prec)
    for x, y in zip(ref[:3], got[:3]):
        assert np.array_equal(x, y)
    assert ref[3] == got[3]


@pytest.mark.parametrize("prec", ["double", "float32"])
@pytest.mark.parametrize("mode", ["local", "global"])
def test_linear_interpair_soft_guides(mode, prec):
    """Eager soft linear pairs now take their guide from the inter-pair Viterbi fill:
    guides bit-identical to striped; scores and gradients tolerance-equal (soft)."""
    a, b = _seqs(150, 10, 40, 15), _seqs(150, 40, 41, 16)
    p = _params("random")
    r0 = _run_linear(a, b, p, mode, "striped", grad_mode="soft", prec=prec)
    r1 = _run_linear(a, b, p, mode, "interpair", grad_mode="soft", prec=prec)
    for x, y in zip(r0[:3], r1[:3]):
        np.testing.assert_allclose(x, y, rtol=1e-11, atol=1e-11)
    assert r0[3] == r1[3]


# ── banded_grad() under fill="interpair" (InterJobT::blo) ─────────────────────
# Each lane is banded around its own guide: every row is computed over the union of the
# lanes' spans with out-of-band cells set to -inf, which must leave each lane's table
# bit-identical to its own banded fill — so scores, gradients and the updated guides
# match the per-pair path exactly, ties included.

def _banded(prec, gm, mode, A, B, p, fill, bws, kernel="auto", p2=None):
    b = _CLS[prec][0](n_threads=3, traceback="pointers")
    b.fill = fill
    b.add_many(A, B, p, gap_model=gm, mode=mode, grad_mode="hard", kernel=kernel)
    b.score_and_grad()
    if p2 is not None:
        b.set_params(p2)
    out = []
    for bw in bws:
        tot = b.banded_grad(bw)
        out.append((tot, b.scores(), *b.grads(), [list(b[i].guide_j) for i in range(len(b))]))
    return out


def _assert_same(r0, r1):
    for x, y in zip(r0, r1):
        assert x[0] == y[0]
        for u, v in zip(x[1:4], y[1:4]):
            assert np.array_equal(u, v)
        assert x[4] == y[4]


@pytest.mark.parametrize("prec", ["double", "float32"])
@pytest.mark.parametrize("gm", ["affine", "linear"])
@pytest.mark.parametrize("mode", ["local", "global"])
@pytest.mark.parametrize("kind", ["ties", "cheap_gaps", "random"])
@pytest.mark.parametrize("lengths", [(1, 30, 1, 60), (15, 30, 50, 51), (40, 120, 40, 120)])
def test_banded_interpair_matches_own(prec, gm, mode, kind, lengths):
    A = _seqs(203, lengths[0], lengths[1], 31)
    B = _seqs(203, lengths[2], lengths[3], 32)
    p = _params(kind)
    p2 = _params("random" if kind != "random" else "ties")
    bws = [1, 3, 8, 3]   # repeated: each call re-bands around the guide the last one left
    for q2 in (None, p2):
        r0 = _banded(prec, gm, mode, A, B, p, "striped", bws, p2=q2)
        r1 = _banded(prec, gm, mode, A, B, p, "interpair", bws, p2=q2)
        _assert_same(r0, r1)


@pytest.mark.parametrize("prec", ["double", "float32"])
@pytest.mark.parametrize("gm", ["affine", "linear"])
@pytest.mark.parametrize("mode", ["local", "global"])
@pytest.mark.parametrize("level", [l for l in nwgrad.available_isa_levels()])
def test_banded_interpair_every_isa_level(level, mode, gm, prec):
    A, B = _seqs(150, 10, 40, 33), _seqs(150, 30, 33, 34)
    p = _params("ties")
    r0 = _banded(prec, gm, mode, A, B, p, "striped", [2, 5], kernel="scalar_fallback")
    r1 = _banded(prec, gm, mode, A, B, p, "interpair", [2, 5], kernel=level)
    _assert_same(r0, r1)


def test_banded_interpair_paths_and_soft_lanes():
    """Alignment strings after a banded interpair pass equal the own path's; soft pairs
    in the batch keep their own banded forward-backward (tolerance-equal: soft)."""
    A, B = _seqs(120, 10, 30, 35), _fixed_len_b(range(120), 40, 36)
    p = _params("ties")
    res = []
    for fill in ("striped", "interpair"):
        b = nwgrad.SeqPairBatchDouble(n_threads=2, traceback="pointers")
        b.fill = fill
        b.add_many(A[:80], B[:80], p, gap_model="affine", mode="local", grad_mode="hard")
        b.add_many(A[80:], B[80:], p, gap_model="affine", mode="local", grad_mode="soft")
        b.score_and_grad()
        b.banded_grad(3)
        res.append((b.scores(), *b.grads()))
        res.append([b[i].guide_j for i in range(120)])
    np.testing.assert_allclose(res[0][0], res[2][0], rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(res[0][1], res[2][1], rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(res[0][2], res[2][2], rtol=1e-11, atol=1e-11)
    assert np.array_equal(res[0][0][:80], res[2][0][:80])
    assert [list(g) for g in res[1]] == [list(g) for g in res[3]]
    # paths: a pair's own banded realign around the same guide gives the same strings
    for i in range(0, 80, 9):
        sp = nwgrad.SeqPairDouble(A[i], B[i], p, gap_model="affine", mode="local",
                                  grad_mode="hard", traceback="pointers")
        sp.alloc_dp(); sp.align_full(); sp.realign_banded(3)
        assert list(sp.guide_j) == list(res[1][i])


def test_banded_interpair_needs_a_guide():
    A, B = _seqs(20, 10, 30, 37), _fixed_len_b(range(20), 40, 38)
    b = nwgrad.SeqPairBatch(n_threads=2)
    b.fill = "interpair"
    b.add_many(A, B, _params("random"), gap_model="affine", mode="local", grad_mode="hard")
    with pytest.raises(Exception, match="guide"):
        b.banded_grad(3)
