"""SeqPairBatch.grads() and SeqPairBatch.alphabet: every pair's cached gradient as
two arrays, without one AlignParams object per pair."""

import numpy as np
import pytest
import nwgrad

from test_weighted_grad import (BATCHES, FIELDS, PROBLEMS, SEQS_A, SEQS_B, dna_params,
                                make_batch)


def per_pair(batch):
    """(matrices, gaps) assembled from batch[i].grad."""
    grads = [batch[i].grad.to_dict() for i in range(len(batch))]
    return (np.array([g["matrix"] for g in grads]),
            np.array([[g[f] for f in FIELDS] for g in grads]))


@pytest.mark.parametrize("cls", BATCHES)
@pytest.mark.parametrize("gap_model, mode", PROBLEMS)
def test_grads_match_per_pair(cls, gap_model, mode):
    batch, _ = make_batch(cls, gap_model, mode)
    batch.score_and_grad()
    matrices, gaps = batch.grads()
    for a in (matrices, gaps):
        assert isinstance(a, np.ndarray) and a.dtype == np.float64
    assert matrices.shape == (len(SEQS_A), 4, 4)
    assert gaps.shape == (len(SEQS_A), 4)
    expected_matrices, expected_gaps = per_pair(batch)
    np.testing.assert_array_equal(matrices, expected_matrices)
    np.testing.assert_array_equal(gaps, expected_gaps)


def test_grads_with_soft_gradients():
    batch, _ = make_batch(nwgrad.SeqPairBatchDouble, grad_mode="soft")
    batch.score_and_grad()
    matrices, gaps = batch.grads()
    expected_matrices, expected_gaps = per_pair(batch)
    np.testing.assert_array_equal(matrices, expected_matrices)
    np.testing.assert_array_equal(gaps, expected_gaps)


def test_grads_protein_alphabet():
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    rng = np.random.default_rng(0)
    M = rng.normal(size=(20, 20))   # asymmetric, so an orientation error would show
    params = nwgrad.AlignParams(nwgrad.SubstMatrix(M, alphabet=alphabet),
                                gap_open_a=3.0, gap_extend_a=0.5,
                                gap_open_b=2.0, gap_extend_b=1.0)
    seqs = ["".join(rng.choice(list(alphabet), size=n)) for n in (30, 25, 40, 1, 33)]
    batch = nwgrad.SeqPairBatchDouble(n_threads=2)
    batch.add_many(seqs, seqs[::-1], params, gap_model="affine", mode="global")
    batch.score_and_grad()
    assert batch.alphabet == alphabet
    matrices, gaps = batch.grads()
    assert matrices.shape == (5, 20, 20)
    expected_matrices, expected_gaps = per_pair(batch)
    np.testing.assert_array_equal(matrices, expected_matrices)
    np.testing.assert_array_equal(gaps, expected_gaps)


def test_grads_sum_to_weighted_grad():
    batch, _ = make_batch(nwgrad.SeqPairBatchDouble)
    batch.score_and_grad()
    w = np.array([0.5, -1.25, 2.0, 0.0, -0.75, 1.0])
    matrices, gaps = batch.grads()
    got = batch.weighted_grad(w).to_dict()
    np.testing.assert_allclose(np.tensordot(w, matrices, axes=1), got["matrix"], atol=1e-12)
    np.testing.assert_allclose(w @ gaps, [got[f] for f in FIELDS], atol=1e-12)


def test_grads_are_copies_and_follow_new_params():
    batch, params = make_batch(nwgrad.SeqPairBatchDouble, "affine", "global")
    batch.score_and_grad()
    matrices, gaps = batch.grads()
    kept = matrices.copy(), gaps.copy()
    matrices[:] = -7.0
    gaps[:] = -7.0
    again = batch.grads()
    np.testing.assert_array_equal(again[0], kept[0])
    np.testing.assert_array_equal(again[1], kept[1])

    # A large gap cost changes the optimal paths, and the arrays follow.
    expensive = nwgrad.AlignParams(params.matrix, gap_open_a=50.0, gap_extend_a=50.0,
                                   gap_open_b=50.0, gap_extend_b=50.0)
    batch.set_params(expensive)
    batch.score_and_grad()
    new_matrices, new_gaps = batch.grads()
    expected_matrices, expected_gaps = per_pair(batch)
    np.testing.assert_array_equal(new_matrices, expected_matrices)
    np.testing.assert_array_equal(new_gaps, expected_gaps)
    assert not np.array_equal(new_gaps, kept[1])


@pytest.mark.parametrize("cls", BATCHES)
def test_alphabet(cls):
    batch, _ = make_batch(cls)
    assert batch.alphabet == "ACGT"


def test_grads_and_alphabet_of_empty_batch_raise():
    batch = nwgrad.SeqPairBatchDouble()
    with pytest.raises(RuntimeError, match="empty batch"):
        batch.grads()
    with pytest.raises(RuntimeError, match="empty batch"):
        batch.alphabet


def test_grads_before_alignment_raise():
    batch, _ = make_batch(nwgrad.SeqPairBatchDouble)
    with pytest.raises(RuntimeError, match="gradient not computed"):
        batch.grads()


def test_grads_after_set_params_raise():
    batch, params = make_batch(nwgrad.SeqPairBatchDouble)
    batch.score_and_grad()
    batch.set_params(params)
    with pytest.raises(RuntimeError, match="gradient not computed"):
        batch.grads()
