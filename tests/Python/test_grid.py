"""traceback="grid": the same result as traceback="pointers", bit for bit, through the
batch layer (per-pair fills, both precisions, threads) and the standalone SeqPair."""
import numpy as np
import pytest

import nwgrad
from nwgrad.matrices import BLOSUM62

AA = "ACDEFGHIKLMNPQRSTVWY"


def related_pairs(n, lo, hi, seed):
    rng = np.random.default_rng(seed)
    A, B = [], []
    for _ in range(n):
        a = "".join(rng.choice(list(AA), int(rng.integers(lo, hi))))
        b = []
        for c in a:
            r = rng.random()
            if r < 0.03:
                continue
            if r < 0.06:
                b.append(AA[int(rng.integers(20))])
            b.append(c if rng.random() > 0.25 else AA[int(rng.integers(20))])
        A.append(a)
        B.append("".join(b))
    return A, B


PARAMS = [
    nwgrad.AlignParams(BLOSUM62, 11.0, 1.0, 11.0, 1.0),     # integer scores: ties
    nwgrad.AlignParams(BLOSUM62, 10.0, 1.0 / 3.0, 9.0, 1.0) * 0.3,  # non-representable extends
]


@pytest.mark.parametrize("cls", [nwgrad.SeqPairBatch, nwgrad.SeqPairBatchDouble])
@pytest.mark.parametrize("pi", range(len(PARAMS)))
@pytest.mark.parametrize("threads", [1, 3])
def test_batch_grid_equals_pointers(cls, pi, threads):
    A, B = related_pairs(12, 40, 400, seed=pi)
    p = PARAMS[pi]
    res = {}
    for tb in ("pointers", "grid"):
        b = cls(A, B, p, n_threads=threads, traceback=tb)
        b.fill = "striped"
        b.hb_cutoff = 16          # pairs above it take the grid
        b.score_and_grad(keep_paths=True)
        mats, gaps = b.grads()
        res[tb] = (b.scores(), mats, gaps,
                   [b[i].aligned() for i in range(len(b))],
                   [list(b[i].guide_j) for i in range(len(b))])
        assert b.traceback == tb
    ref, got = res["pointers"], res["grid"]
    np.testing.assert_array_equal(got[0], ref[0])
    np.testing.assert_array_equal(got[1], ref[1])
    np.testing.assert_array_equal(got[2], ref[2])
    assert got[3] == ref[3]
    assert got[4] == ref[4]


def test_seqpair_grid_equals_pointers():
    A, B = related_pairs(4, 600, 900, seed=5)
    for a, b in zip(A, B):
        sp = {tb: nwgrad.SeqPair(a, b, PARAMS[0], traceback=tb) for tb in ("pointers", "grid")}
        s_ref, g_ref = sp["pointers"].score_and_grad()
        s_got, g_got = sp["grid"].score_and_grad()
        assert s_got == s_ref
        assert g_got.to_dict()["gap_open_a"] == g_ref.to_dict()["gap_open_a"]
        np.testing.assert_array_equal(g_got.to_dict()["matrix"], g_ref.to_dict()["matrix"])
        assert sp["grid"].aligned() == sp["pointers"].aligned()


def test_grid_unknown_name_lists_grid():
    with pytest.raises(ValueError, match="grid"):
        nwgrad.SeqPairBatch(["ACD"], ["ACD"], PARAMS[0], traceback="gridd")


@pytest.mark.parametrize("cls", [nwgrad.SeqPairBatch, nwgrad.SeqPairBatchDouble])
@pytest.mark.parametrize("fill", ["rowwise", "striped", "interpair"])
def test_auto_grid_fallback_sizes_its_tables(cls, fill):
    # `auto` resolves to grid, which declines a negative gap open: the run takes the
    # fallback mode.  The buffers must be sized for that mode — rowwise fill used to run
    # into unallocated score tables (null store, CI 2026-10-27).
    A, B = related_pairs(8, 20, 120, seed=11)
    p = nwgrad.AlignParams(BLOSUM62, -1.0, 1.0, -1.0, 1.0)
    res = {}
    for tb in ("auto", "pointers"):
        b = cls(A, B, p, n_threads=3, traceback=tb)
        b.fill = fill
        b.hb_cutoff = 4
        b.score_and_grad(keep_paths=True)
        res[tb] = b.scores()
    np.testing.assert_array_equal(res["auto"], res["pointers"])


BATCHES = {"float32": nwgrad.SeqPairBatch, "float64": nwgrad.SeqPairBatchDouble,
           "int16": nwgrad.SeqPairBatchInt16, "int32": nwgrad.SeqPairBatchInt32,
           "int64": nwgrad.SeqPairBatchInt64}


def _run(cls, A, B, p, tb, mode, threads, cutoff=16, kernel=None):
    b = cls(A, B, p, n_threads=threads, traceback=tb, mode=mode)
    b.fill = "striped"
    b.hb_cutoff = cutoff
    if kernel is not None:
        b.kernel = kernel
    b.score_and_grad(keep_paths=True)
    mats, gaps = b.grads()
    return (b.scores(), mats, gaps, [b[i].aligned() for i in range(len(b))],
            [list(b[i].guide_j) for i in range(len(b))])


def _same(got, ref):
    np.testing.assert_array_equal(got[0], ref[0])
    np.testing.assert_array_equal(got[1], ref[1])
    np.testing.assert_array_equal(got[2], ref[2])
    assert got[3] == ref[3]
    assert got[4] == ref[4]


@pytest.mark.parametrize("tname", list(BATCHES))
@pytest.mark.parametrize("mode", ["global", "local", "glocal"])
@pytest.mark.parametrize("pi", range(len(PARAMS)))
def test_batch_grid_equals_pointers_every_mode_and_type(tname, mode, pi):
    if tname.startswith("int") and pi == 1:
        pytest.skip("integer types take integral parameters only")
    # int16 BLOSUM62 11/1 crosses the quarter rule at ~L300: the saturating regime
    A, B = related_pairs(10, 40, 600 if tname == "int16" else 400, seed=pi + 17)
    if mode == "local":   # unrelated flanks, so the local path starts and ends inside
        rng = np.random.default_rng(pi)
        fl = lambda: "".join(rng.choice(list(AA), int(rng.integers(0, 60))))
        A = [fl() + a + fl() for a in A]
        B = [fl() + b + fl() for b in B]
    cls, p = BATCHES[tname], PARAMS[pi]
    _same(_run(cls, A, B, p, "grid", mode, 2), _run(cls, A, B, p, "pointers", mode, 2))


@pytest.mark.parametrize("tname", ["float32", "int32"])
@pytest.mark.parametrize("mode", ["global", "local", "glocal"])
def test_auto_takes_grid_where_it_split(tname, mode):
    # pairs past hb_cutoff that `auto` used to split (hirschberg_pmax): now Grid, i.e.
    # Pointers' result (Global float32 was already Grid)
    A, B = related_pairs(6, 200, 400, seed=3)
    cls, p = BATCHES[tname], PARAMS[0]
    _same(_run(cls, A, B, p, "auto", mode, 2, cutoff=64), _run(cls, A, B, p, "pointers", mode, 2, cutoff=64))
