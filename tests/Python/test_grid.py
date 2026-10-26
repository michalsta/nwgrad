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
