"""fill="rowwise" must be indistinguishable from fill="striped".

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
        sp.fill = fill
        sp.alloc_dp()
        sp.align_full()
        aligned.append((sp.score, sp.aligned()))
    return b.scores(), mats, gaps, aligned


def test_default_and_validation():
    b = nwgrad.SeqPairBatchDouble(n_threads=1)
    assert b.fill == "striped"
    b.fill = "rowwise"
    assert b.fill == "rowwise"
    with pytest.raises(ValueError, match="unknown fill"):
        b.fill = "diagonal"


@pytest.mark.parametrize("kind", ["ties", "cheap_gaps", "random"])
@pytest.mark.parametrize("mode", ["local", "global"])
@pytest.mark.parametrize("traceback", ["pointers", "scores"])
@pytest.mark.parametrize("lengths", [(1, 30, 1, 60), (15, 30, 40, 60), (60, 200, 60, 200)])
def test_rowwise_matches_striped(kind, mode, traceback, lengths):
    a = _seqs(300, lengths[0], lengths[1], 1)
    b = _seqs(300, lengths[2], lengths[3], 2)
    p = _params(kind)
    s0, m0, g0, al0 = _run(a, b, p, mode, traceback, "striped")
    s1, m1, g1, al1 = _run(a, b, p, mode, traceback, "rowwise")
    assert np.array_equal(s0, s1)
    assert np.array_equal(m0, m1)
    assert np.array_equal(g0, g1)
    assert al0 == al1


@pytest.mark.parametrize("level", [l for l in nwgrad.available_isa_levels()])
def test_every_isa_level(level):
    a, b = _seqs(200, 10, 40, 3), _seqs(200, 30, 70, 4)
    p = _params("cheap_gaps")
    ref = _run(a, b, p, "local", "pointers", "striped", kernel="scalar_fallback")
    got = _run(a, b, p, "local", "pointers", "rowwise", kernel=level)
    for x, y in zip(ref[:3], got[:3]):
        assert np.array_equal(x, y)
    assert ref[3] == got[3]


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
