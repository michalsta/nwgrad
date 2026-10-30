"""int16 past the old quarter-range bound: saturating arithmetic and the value-range bound.

int16 computes with saturating adds and subtracts and a sentinel of lowest(), so its
a-priori bound asks only that every reachable value stays in range and that no
sentinel-derived value can climb up to a reachable one (scalar_traits.hpp, aligner.hpp
nw_int_bound_check).  BLOSUM62 pairs of ~1000 residues now fit where ~370 was the limit.
Every result must still equal double's bit for bit: scores, paths, gradients, guides,
banded steps, score-only, streaming, at every ISA level.
"""
import numpy as np
import pytest

import nwgrad
from nwgrad import matrices


def related_pairs(rng, alpha, n, length, mut=0.3, indel=0.05):
    syms = np.array(list(alpha))
    seqs_a, seqs_b = [], []
    for _ in range(n):
        a = syms[rng.integers(0, len(syms), length)]
        b = a.copy()
        m = rng.random(length) < mut
        b[m] = syms[rng.integers(0, len(syms), int(m.sum()))]
        # a few indels so gap runs appear on the path
        keep = rng.random(length) >= indel
        b = b[keep]
        ins = rng.random(len(b)) < indel
        b = np.concatenate([b, syms[rng.integers(0, len(syms), int(ins.sum()))]])
        seqs_a.append("".join(a)); seqs_b.append("".join(b))
    return seqs_a, seqs_b


def blosum_params(go=11.0, ge=1.0):
    return nwgrad.AlignParams(matrices.BLOSUM62, go, ge, go, ge)


def same_params(a, b):
    return np.array_equal(a.matrices, b.matrices) and np.array_equal(a.gaps, b.gaps)


@pytest.mark.parametrize("mode", ["global", "local"])
@pytest.mark.parametrize("tb", ["auto", "pointers", "scores", "hirschberg", "hirschberg_pmax"])
def test_long_protein_pairs_match_double(mode, tb):
    rng = np.random.default_rng(len(mode) * 31 + len(tb))
    p = blosum_params()
    sa, sb = related_pairs(rng, matrices.BLOSUM62.alphabet.symbols, 6, 1000)
    kw = dict(gap_model="affine", mode=mode, n_threads=2, traceback=tb)
    bd = nwgrad.SeqPairBatchDouble(sa, sb, p, **kw)
    bi = nwgrad.SeqPairBatchInt16(sa, sb, p, **kw)
    assert bd.score_and_grad(keep_paths=True) == bi.score_and_grad(keep_paths=True)
    assert np.array_equal(bd.scores(), bi.scores())
    hb = tb in ("hirschberg", "hirschberg_pmax") or tb == "auto"
    if not hb:
        assert same_params(bd.compute_grad(), bi.compute_grad())
        for i in range(len(sa)):
            assert bd[i].aligned() == bi[i].aligned()
    # banded re-alignment around the path, with the guides just computed
    assert np.array_equal(bd.realign_banded(16), bi.realign_banded(16))
    assert np.array_equal(bd.scores(), bi.scores())
    # score-only and streaming
    sd = nwgrad.SeqPairBatchDouble(sa, sb, p, grad_mode="none", **kw)
    si = nwgrad.SeqPairBatchInt16(sa, sb, p, grad_mode="none", **kw)
    sd.score_and_grad(); si.score_and_grad()
    assert np.array_equal(sd.scores(), si.scores())
    rd = nwgrad.SeqPairBatchDouble(n_threads=2, gap_model="affine", mode=mode).align(sa, sb, p)
    ri = nwgrad.SeqPairBatchInt16(n_threads=2, gap_model="affine", mode=mode).align(sa, sb, p)
    assert np.array_equal(rd.scores, ri.scores)
    assert same_params(rd.grad, ri.grad)


@pytest.mark.parametrize("mode", ["global", "local"])
def test_long_linear_pairs_match_double(mode):
    rng = np.random.default_rng(7 + len(mode))
    p = nwgrad.AlignParams(matrices.BLOSUM62, 0.0, 4.0, 0.0, 4.0)
    sa, sb = related_pairs(rng, matrices.BLOSUM62.alphabet.symbols, 4, 900)
    for tb in ["auto", "pointers", "hirschberg"]:
        kw = dict(gap_model="linear", mode=mode, n_threads=2, traceback=tb)
        bd = nwgrad.SeqPairBatchDouble(sa, sb, p, **kw)
        bi = nwgrad.SeqPairBatchInt16(sa, sb, p, **kw)
        assert bd.score_and_grad(keep_paths=True) == bi.score_and_grad(keep_paths=True)
        assert np.array_equal(bd.scores(), bi.scores())
        if tb == "pointers":
            assert same_params(bd.compute_grad(), bi.compute_grad())


def test_every_isa_level_agrees_on_long_pairs():
    rng = np.random.default_rng(2026)
    p = blosum_params()
    sa, sb = related_pairs(rng, matrices.BLOSUM62.alphabet.symbols, 4, 1100)
    ref = None
    for level in nwgrad.available_isa_levels():
        for tb in ["pointers", "hirschberg_pmax"]:
            b = nwgrad.SeqPairBatchInt16(sa, sb, p, mode="global", kernel=level, traceback=tb, n_threads=2)
            b.score_and_grad(keep_paths=True)
            got = (list(b.scores()), b.compute_grad().matrices.tolist())
            if ref is None:
                ref = got
            else:
                assert got == ref, (level, tb)


def test_bound_is_the_value_range():
    """The int16 bound follows the values a path can collect, not the move count."""
    p = blosum_params()
    alpha = matrices.BLOSUM62.alphabet.symbols
    rng = np.random.default_rng(5)
    sa, sb = related_pairs(rng, alpha, 1, 1400)      # 1400 * 11 * 2 + 11 <= 32767: fits
    assert nwgrad.nw_score_affine_int16(sa[0], sb[0], p) == nwgrad.nw_score_affine_double(sa[0], sb[0], p)
    sa, sb = related_pairs(rng, alpha, 1, 1600)      # 1600 * 11 * 2 > 32767: refused
    with pytest.raises(OverflowError, match="int16"):
        nwgrad.nw_score_affine_int16(sa[0], sb[0], p)
    # the same pair in int32 is fine
    assert nwgrad.nw_score_affine_int32(sa[0], sb[0], p) == nwgrad.nw_score_affine_double(sa[0], sb[0], p)
    # deep negatives: a Global pair whose all-gap floor leaves the range is refused
    m = np.where(np.eye(4, dtype=bool), 1.0, -1.0)
    q = nwgrad.AlignParams(nwgrad.SubstMatrix(m, alphabet="ACGT"), 100.0, 40.0, 100.0, 40.0)
    a, b = "A" * 500, "C" * 500                      # floor ~ -(200 + 500*40 + 564*40) < -32767
    with pytest.raises(OverflowError, match="int16"):
        nwgrad.nw_score_affine_int16(a, b, q)
    assert nwgrad.nw_score_affine_int32(a, b, q) == nwgrad.nw_score_affine_double(a, b, q)
