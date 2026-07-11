"""Tests for the API added in response to review feedback:

- AlignParams accepting a SubstMatrix directly (carrying its own alphabet)
- AlignParams.to_dict()
- SeqPair.aligned() / SeqPair.formatted()
- SeqPair.score_and_grad()
- gradient exported in the matrix's alphabet (not the default canonical order)
- params/SeqPair kept alive automatically (no dangling reference footgun)
"""

import gc

import numpy as np
import pytest

import nwgrad
from nwgrad.matrices import BLOSUM62


def _affine_params(matrix):
    return nwgrad.AlignParams(matrix, gap_open_a=11.0, gap_extend_a=1.0,
                              gap_open_b=11.0, gap_extend_b=1.0)


# ── AlignParams accepts a SubstMatrix ────────────────────────────────────────

def test_alignparams_from_subst_matrix_matches_array_form():
    arr = BLOSUM62.to_matrix()
    alpha = BLOSUM62.alphabet

    p_sm = _affine_params(BLOSUM62)                              # SubstMatrix form
    p_arr = nwgrad.AlignParams(arr, alphabet=alpha,             # array + alphabet form
                               gap_open_a=11.0, gap_extend_a=1.0,
                               gap_open_b=11.0, gap_extend_b=1.0)

    assert p_sm.matrix.alphabet == alpha
    np.testing.assert_array_equal(p_sm.matrix.to_matrix(), p_arr.matrix.to_matrix())


def test_alignparams_subst_matrix_carries_alphabet_into_scores():
    # B/Z/X ambiguity codes only exist in the 23-symbol alphabet; aligning them
    # only works if the alphabet travelled with the matrix.
    p = _affine_params(BLOSUM62)
    sp = nwgrad.SeqPair("BZX", "BZX", p, gap_model="affine",
                        mode="global", grad_mode="hard")
    score, _ = sp.score_and_grad()
    expected = sum(BLOSUM62.score(c, c) for c in "BZX")
    assert score == pytest.approx(expected)


# ── to_dict() ────────────────────────────────────────────────────────────────

def test_to_dict_shape_and_keys():
    p = _affine_params(BLOSUM62)
    d = p.to_dict()
    assert set(d) == {"matrix", "alphabet", "gap_open_a", "gap_extend_a",
                      "gap_open_b", "gap_extend_b"}
    assert d["alphabet"] == BLOSUM62.alphabet
    assert d["matrix"].shape == (BLOSUM62.size, BLOSUM62.size)
    assert d["gap_open_a"] == 11.0 and d["gap_extend_b"] == 1.0
    np.testing.assert_array_equal(d["matrix"], BLOSUM62.to_matrix())


# ── aligned() / formatted() ──────────────────────────────────────────────────

@pytest.mark.parametrize("gap_model", ["linear", "affine"])
@pytest.mark.parametrize("mode", ["global", "local"])
def test_aligned_strings_are_consistent(gap_model, mode):
    p = _affine_params(BLOSUM62)
    sp = nwgrad.SeqPair("PLEASANTLY", "MEANLY", p, gap_model=gap_model,
                        mode=mode, grad_mode="hard")
    sp.alloc_dp()
    sp.align_full()
    a, b = sp.aligned()

    assert len(a) == len(b)                          # equal-length columns
    assert not any(ca == "-" and cb == "-"           # no all-gap column
                   for ca, cb in zip(a, b))
    # ungapped projections are substrings of the originals
    assert a.replace("-", "") in "PLEASANTLY"
    assert b.replace("-", "") in "MEANLY"
    if mode == "global":
        assert a.replace("-", "") == "PLEASANTLY"
        assert b.replace("-", "") == "MEANLY"
        # round-trips through the inverse guide_j helper
        assert nwgrad.guide_j_from_aligned(a, b) == sp.guide_j


def test_global_affine_aligned_exact():
    p = _affine_params(BLOSUM62)
    sp = nwgrad.SeqPair("PLEASANTLY", "MEANLY", p, gap_model="affine",
                        mode="global", grad_mode="hard")
    sp.alloc_dp()
    sp.align_full()
    assert sp.aligned() == ("PLEASANTLY", "-MEAN---LY")


def test_formatted_three_lines_and_match_markers():
    p = _affine_params(BLOSUM62)
    sp = nwgrad.SeqPair("PLEASANTLY", "MEANLY", p, gap_model="affine",
                        mode="global", grad_mode="hard")
    sp.alloc_dp()
    sp.align_full()
    a, mid, b = sp.formatted(width=0).split("\n")
    assert (a, b) == ("PLEASANTLY", "-MEAN---LY")
    for ca, m, cb in zip(a, mid, b):
        if ca == "-" or cb == "-":
            assert m == " "
        elif ca == cb:
            assert m == "|"
        else:
            assert m == "."


def test_formatted_wraps_at_width():
    p = _affine_params(BLOSUM62)
    sp = nwgrad.SeqPair("ACDEFGHIKLMNPQRSTVWY", "ACDEFGHIKLMNPQRSTVWY", p,
                        gap_model="affine", mode="global", grad_mode="hard")
    sp.alloc_dp()
    sp.align_full()
    blocks = sp.formatted(width=8).split("\n\n")
    assert len(blocks) == 3            # 20 columns / 8 -> 3 blocks
    assert all(len(line) <= 8 for blk in blocks for line in blk.split("\n"))


def test_aligned_requires_dp_tables():
    p = _affine_params(BLOSUM62)
    sp = nwgrad.SeqPair("ACDE", "ACDF", p, gap_model="affine",
                        mode="global", grad_mode="hard")
    sp.alloc_dp()
    sp.align_full()
    sp.drop_dp()
    with pytest.raises(Exception):
        sp.aligned()


# ── score_and_grad() ─────────────────────────────────────────────────────────

def test_score_and_grad_matches_separate_calls():
    p = _affine_params(BLOSUM62)
    sp1 = nwgrad.SeqPair("PLEASANTLY", "MEANLY", p, gap_model="affine",
                         mode="global", grad_mode="hard")
    score, grad = sp1.score_and_grad()

    sp2 = nwgrad.SeqPair("PLEASANTLY", "MEANLY", p, gap_model="affine",
                         mode="global", grad_mode="hard")
    sp2.alloc_dp()
    sp2.align_full()
    sp2.compute_grad()
    assert score == pytest.approx(sp2.score)
    np.testing.assert_array_equal(grad.matrix.to_matrix(),
                                  sp2.grad.matrix.to_matrix())


def test_score_and_grad_none_raises():
    p = _affine_params(BLOSUM62)
    sp = nwgrad.SeqPair("ACDE", "ACDF", p, gap_model="affine",
                        mode="global", grad_mode="none")
    with pytest.raises(Exception):
        sp.score_and_grad()


# ── gradient alphabet correctness ────────────────────────────────────────────

def test_gradient_uses_matrix_alphabet():
    p = _affine_params(BLOSUM62)
    sp = nwgrad.SeqPair("PLEASANTLY", "MEANLY", p, gap_model="affine",
                        mode="global", grad_mode="hard")
    _, grad = sp.score_and_grad()
    assert grad.matrix.alphabet == BLOSUM62.alphabet
    assert grad.matrix.to_matrix().shape == (BLOSUM62.size, BLOSUM62.size)


def test_batch_gradient_uses_matrix_alphabet():
    p = _affine_params(BLOSUM62)
    pairs = [nwgrad.SeqPair(a, b, p, gap_model="affine", mode="global",
                            grad_mode="hard")
             for a, b in [("PLEASANTLY", "MEANLY"), ("ACDE", "ACDF")]]
    batch = nwgrad.SeqPairBatch(n_threads=2)
    for sp in pairs:
        batch.add(sp)
    batch.score_and_grad()
    grad = batch.compute_grad()
    assert grad.matrix.alphabet == BLOSUM62.alphabet
    assert grad.matrix.to_matrix().shape == (BLOSUM62.size, BLOSUM62.size)


# ── lifetime: no dangling-params footgun ─────────────────────────────────────

def _build_pair_with_local_params():
    # `p` is a local that goes out of scope on return; the SeqPair must keep it alive.
    p = _affine_params(BLOSUM62)
    return nwgrad.SeqPair("PLEASANTLY", "MEANLY", p, gap_model="affine",
                          mode="global", grad_mode="hard")


def test_seqpair_keeps_params_alive():
    sp = _build_pair_with_local_params()
    gc.collect()
    score, grad = sp.score_and_grad()      # would crash if params were freed
    assert score == pytest.approx(-3.0)


def _build_batch_with_local_everything():
    p = _affine_params(BLOSUM62)
    batch = nwgrad.SeqPairBatch(n_threads=2)
    for a, b in [("PLEASANTLY", "MEANLY"), ("ACDE", "ACDF")]:
        batch.add(nwgrad.SeqPair(a, b, p, gap_model="affine",
                                 mode="global", grad_mode="hard"))
    return batch


def test_batch_keeps_pairs_and_params_alive():
    batch = _build_batch_with_local_everything()
    gc.collect()
    total = batch.score_and_grad()
    grad = batch.compute_grad()
    assert np.isfinite(total)
    assert grad.matrix.to_matrix().shape == (BLOSUM62.size, BLOSUM62.size)
