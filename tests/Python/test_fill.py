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


def _run(seqs_a, seqs_b, params, mode, traceback, fill, kernel="auto"):
    b = nwgrad.SeqPairBatchDouble(n_threads=4, traceback=traceback)
    b.fill = fill
    b.add_many(seqs_a, seqs_b, params, gap_model="affine", mode=mode,
               grad_mode="hard", kernel=kernel)
    b.score_and_grad()
    mats, gaps = b.grads()
    aligned = []
    for i in range(0, len(seqs_a), 7):
        sp = nwgrad.SeqPairDouble(seqs_a[i], seqs_b[i], params, gap_model="affine", mode=mode,
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


@pytest.mark.parametrize("fill", ["rowwise", "interpair"])
@pytest.mark.parametrize("kind", ["ties", "cheap_gaps", "random"])
@pytest.mark.parametrize("mode", ["local", "global"])
@pytest.mark.parametrize("traceback", ["pointers", "scores"])
@pytest.mark.parametrize("lengths", [(1, 30, 1, 60), (15, 30, 40, 60), (60, 200, 60, 200),
                                     (15, 30, 50, 51)])
def test_fill_matches_striped(fill, kind, mode, traceback, lengths):
    # (15, 30, 50, 51): every B of length 50, so inter-pair groups are full, as in
    # miRNA x site data; the others mix lengths, so many groups are short.
    a = _seqs(301, lengths[0], lengths[1], 1)
    b = _seqs(301, lengths[2], lengths[3], 2)
    p = _params(kind)
    s0, m0, g0, al0 = _run(a, b, p, mode, traceback, "striped")
    s1, m1, g1, al1 = _run(a, b, p, mode, traceback, fill)
    assert np.array_equal(s0, s1)
    assert np.array_equal(m0, m1)
    assert np.array_equal(g0, g1)
    assert al0 == al1


@pytest.mark.parametrize("fill", ["rowwise", "interpair"])
@pytest.mark.parametrize("mode", ["local", "global"])
@pytest.mark.parametrize("level", [l for l in nwgrad.available_isa_levels()])
def test_every_isa_level(level, mode, fill):
    a, b = _seqs(203, 10, 40, 3), _seqs(203, 30, 33, 4)
    p = _params("cheap_gaps")
    ref = _run(a, b, p, mode, "pointers", "striped", kernel="scalar_fallback")
    got = _run(a, b, p, mode, "pointers", fill, kernel=level)
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
