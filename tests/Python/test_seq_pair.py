"""Tests for SeqPair and SeqPairBatch, comparing against old single-pair and
BatchAligner APIs.

Coverage:
  - SeqPairBatch.score_and_grad() vs old API; dp_valid stays False; compute_grad() after
  - SeqPair.drop_dp() / SeqPairBatch.drop_dp() lifecycle and memory release
  - SeqPair.align_full() score vs nw_score / sw_score / nw_score_affine / sw_score_affine
  - SeqPair hard grad vs nw_grad / sw_grad / nw_affine_grad / sw_affine_grad
  - SeqPair soft score (log Z) and grad vs nw_soft_grad / sw_soft_grad / ...
  - SeqPair.realign_banded() with wide band around exact guide matches full DP
  - State-machine flags: path_valid / score_valid / grad_valid lifecycle
  - set_matrix() preserves path_valid, clears score_valid + grad_valid
  - SeqPairBatch sums match individual SeqPair results
  - SeqPairBatch matches BatchAligner (hard grad mode)
  - SeqPairBatch multi-thread matches single-thread
  - Error handling: premature calls raise; align_full without alloc_dp raises
"""

import numpy as np
import pytest
import nwgrad
from test_blosum import BLOSUM62


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def blosum():
    return nwgrad.BlosumMatrix(BLOSUM62)


@pytest.fixture(scope="module")
def blosum2():
    """A slightly perturbed matrix for set_matrix() tests."""
    arr = BLOSUM62 + 0.5
    return nwgrad.BlosumMatrix(arr)


@pytest.fixture(scope="module")
def unit():
    mat = np.zeros((20, 20), dtype=np.float64)
    np.fill_diagonal(mat, 1.0)
    return nwgrad.BlosumMatrix(mat)


PAIRS = [
    ("A",          "A"),
    ("ACDE",       "ACDE"),
    ("ACDE",       "ACDF"),
    ("A",          "AC"),
    ("PLEASANTLY", "MEANLY"),
    ("ACDEFGHIKL", "CDEFGHIKLM"),
    ("MADEEKLF",   "MADEEKLF"),
    ("ACDEFG",     "ACDE"),
    ("MADEEKLF",   "ACDEFGHIKL"),
]

ALL_CONFIGS = [
    ("linear", "global", 0.0,  1.0),
    ("linear", "local",  0.0,  1.0),
    ("affine", "global", 11.0, 1.0),
    ("affine", "local",  11.0, 1.0),
]


def old_score(a, b, mat, gap_model, mode, gap_open, gap_extend):
    if gap_model == "linear" and mode == "global":
        return nwgrad.nw_score(a, b, mat, gap_extend)
    if gap_model == "linear" and mode == "local":
        return nwgrad.sw_score(a, b, mat, gap_extend)
    if gap_model == "affine" and mode == "global":
        return nwgrad.nw_score_affine(a, b, mat, gap_open, gap_extend)
    return nwgrad.sw_score_affine(a, b, mat, gap_open, gap_extend)


def old_hard_grad(a, b, mat, gap_model, mode, gap_open, gap_extend):
    if gap_model == "linear" and mode == "global":
        return nwgrad.nw_grad(a, b, mat, gap_extend)
    if gap_model == "linear" and mode == "local":
        return nwgrad.sw_grad(a, b, mat, gap_extend)
    if gap_model == "affine" and mode == "global":
        return nwgrad.nw_affine_grad(a, b, mat, gap_open, gap_extend)
    return nwgrad.sw_affine_grad(a, b, mat, gap_open, gap_extend)


def old_soft_grad(a, b, mat, gap_model, mode, gap_open, gap_extend):
    if gap_model == "linear" and mode == "global":
        return nwgrad.nw_soft_grad(a, b, mat, gap_extend)
    if gap_model == "linear" and mode == "local":
        return nwgrad.sw_soft_grad(a, b, mat, gap_extend)
    if gap_model == "affine" and mode == "global":
        return nwgrad.nw_affine_soft_grad(a, b, mat, gap_open, gap_extend)
    return nwgrad.sw_affine_soft_grad(a, b, mat, gap_open, gap_extend)


# ── SeqPair: explicit allocation requirement ──────────────────────────────────

class TestExplicitAlloc:
    """align_full / realign_banded require alloc_dp() first; exception if missing."""

    def test_align_full_without_alloc_raises(self, blosum):
        sp = nwgrad.SeqPair("ACDE", "ACDE", blosum)
        with pytest.raises(Exception, match="alloc_dp"):
            sp.align_full()

    def test_realign_banded_without_alloc_raises(self, blosum):
        sp = nwgrad.SeqPair("ACDE", "ACDE", blosum)
        sp.alloc_dp()
        sp.align_full()
        sp.drop_dp()
        with pytest.raises(Exception, match="alloc_dp"):
            sp.realign_banded(10)

    def test_align_full_after_alloc_succeeds(self, blosum):
        sp = nwgrad.SeqPair("ACDE", "ACDE", blosum)
        sp.alloc_dp()
        sp.align_full()  # must not raise
        assert sp.score_valid

    def test_batch_align_full_without_alloc_raises(self, blosum):
        batch = nwgrad.SeqPairBatch(n_threads=1)
        batch.add(nwgrad.SeqPair("ACDE", "ACDE", blosum))
        with pytest.raises(Exception, match="alloc_dp"):
            batch.align_full()

    def test_batch_align_full_after_alloc_succeeds(self, blosum):
        batch = nwgrad.SeqPairBatch(n_threads=1)
        batch.add(nwgrad.SeqPair("ACDE", "ACDE", blosum))
        batch.alloc_dp()
        batch.align_full()  # must not raise


# ── SeqPair: align_full score ─────────────────────────────────────────────────

class TestSeqPairScore:
    @pytest.mark.parametrize("gap_model,mode,gap_open,gap_extend", ALL_CONFIGS)
    @pytest.mark.parametrize("a,b", PAIRS)
    def test_align_full_score_matches_old_api(
            self, a, b, gap_model, mode, gap_open, gap_extend, blosum):
        sp = nwgrad.SeqPair(a, b, blosum,
                             gap_open=gap_open, gap_extend=gap_extend,
                             gap_model=gap_model, mode=mode, grad_mode="none")
        sp.alloc_dp()
        sp.align_full()
        expected = old_score(a, b, blosum, gap_model, mode, gap_open, gap_extend)
        assert sp.score == pytest.approx(expected), (
            f"{gap_model}/{mode} pair ({a!r},{b!r}): got {sp.score}, expected {expected}"
        )

    def test_score_none_before_align(self, blosum):
        sp = nwgrad.SeqPair("ACDE", "ACDE", blosum)
        assert sp.score is None

    def test_score_valid_after_align(self, blosum):
        sp = nwgrad.SeqPair("ACDE", "ACDE", blosum)
        assert not sp.score_valid
        sp.alloc_dp()
        sp.align_full()
        assert sp.score_valid


# ── SeqPair: hard gradient ────────────────────────────────────────────────────

class TestSeqPairHardGrad:
    @pytest.mark.parametrize("gap_model,mode,gap_open,gap_extend", ALL_CONFIGS)
    @pytest.mark.parametrize("a,b", PAIRS)
    def test_hard_grad_matches_old_api(
            self, a, b, gap_model, mode, gap_open, gap_extend, blosum):
        sp = nwgrad.SeqPair(a, b, blosum,
                             gap_open=gap_open, gap_extend=gap_extend,
                             gap_model=gap_model, mode=mode, grad_mode="hard")
        sp.alloc_dp()
        sp.align_full()
        sp.compute_grad()

        exp_score, exp_grad = old_hard_grad(
            a, b, blosum, gap_model, mode, gap_open, gap_extend)

        assert sp.score == pytest.approx(exp_score)
        np.testing.assert_allclose(
            np.array(sp.grad), np.array(exp_grad), atol=1e-12,
            err_msg=f"hard grad mismatch for {gap_model}/{mode} ({a!r},{b!r})"
        )

    def test_grad_none_before_compute(self, blosum):
        sp = nwgrad.SeqPair("ACDE", "ACDE", blosum)
        sp.alloc_dp()
        sp.align_full()
        assert sp.grad is None

    def test_grad_valid_after_compute(self, blosum):
        sp = nwgrad.SeqPair("ACDE", "ACDE", blosum)
        sp.alloc_dp()
        sp.align_full()
        sp.compute_grad()
        assert sp.grad_valid
        assert sp.grad is not None

    def test_grad_shape(self, blosum):
        sp = nwgrad.SeqPair("ACDE", "ACDE", blosum)
        sp.alloc_dp()
        sp.align_full()
        sp.compute_grad()
        assert np.array(sp.grad).shape == (20, 20)


# ── SeqPair: soft gradient ────────────────────────────────────────────────────

class TestSeqPairSoftGrad:
    @pytest.mark.parametrize("gap_model,mode,gap_open,gap_extend", ALL_CONFIGS)
    @pytest.mark.parametrize("a,b", PAIRS)
    def test_soft_score_and_grad_match_old_api(
            self, a, b, gap_model, mode, gap_open, gap_extend, blosum):
        sp = nwgrad.SeqPair(a, b, blosum,
                             gap_open=gap_open, gap_extend=gap_extend,
                             gap_model=gap_model, mode=mode, grad_mode="soft")
        sp.alloc_dp()
        sp.align_full()
        sp.compute_grad()

        exp_score, exp_grad = old_soft_grad(
            a, b, blosum, gap_model, mode, gap_open, gap_extend)

        assert sp.score == pytest.approx(exp_score, rel=1e-10), (
            f"soft score mismatch for {gap_model}/{mode} ({a!r},{b!r}): "
            f"got {sp.score}, expected {exp_score}"
        )
        np.testing.assert_allclose(
            np.array(sp.grad), np.array(exp_grad), atol=1e-10,
            err_msg=f"soft grad mismatch for {gap_model}/{mode} ({a!r},{b!r})"
        )


# ── SeqPair: banded realignment ───────────────────────────────────────────────

class TestSeqPairBanded:
    @pytest.mark.parametrize("gap_model,mode,gap_open,gap_extend", ALL_CONFIGS)
    @pytest.mark.parametrize("a,b", PAIRS)
    def test_realign_banded_wide_band_matches_full(
            self, a, b, gap_model, mode, gap_open, gap_extend, blosum):
        sp = nwgrad.SeqPair(a, b, blosum,
                             gap_open=gap_open, gap_extend=gap_extend,
                             gap_model=gap_model, mode=mode, grad_mode="hard")
        sp.alloc_dp()
        sp.align_full()
        full_score = sp.score

        # Wide band: should recover same score
        bw = max(len(a), len(b)) + 5
        sp.realign_banded(bw)
        assert sp.score == pytest.approx(full_score, rel=1e-10), (
            f"realign_banded(wide) score mismatch for {gap_model}/{mode} ({a!r},{b!r})"
        )

    @pytest.mark.parametrize("gap_model,mode,gap_open,gap_extend", ALL_CONFIGS)
    @pytest.mark.parametrize("a,b", PAIRS)
    def test_realign_banded_hard_grad_matches_full(
            self, a, b, gap_model, mode, gap_open, gap_extend, blosum):
        sp = nwgrad.SeqPair(a, b, blosum,
                             gap_open=gap_open, gap_extend=gap_extend,
                             gap_model=gap_model, mode=mode, grad_mode="hard")
        sp.alloc_dp()
        sp.align_full()
        sp.compute_grad()
        full_grad = np.array(sp.grad).copy()

        bw = max(len(a), len(b)) + 5
        sp.realign_banded(bw)
        sp.compute_grad()
        np.testing.assert_allclose(
            np.array(sp.grad), full_grad, atol=1e-12,
            err_msg=f"realign_banded grad mismatch for {gap_model}/{mode} ({a!r},{b!r})"
        )

    def test_realign_banded_requires_path(self, blosum):
        sp = nwgrad.SeqPair("ACDE", "ACDE", blosum)
        with pytest.raises(Exception, match="align_full"):
            sp.realign_banded(5)

    def test_realign_banded_clears_grad_valid(self, blosum):
        sp = nwgrad.SeqPair("ACDE", "ACDE", blosum)
        sp.alloc_dp()
        sp.align_full()
        sp.compute_grad()
        assert sp.grad_valid
        sp.realign_banded(10)
        assert not sp.grad_valid

    def test_realign_banded_sets_score_valid(self, blosum):
        sp = nwgrad.SeqPair("ACDE", "ACDE", blosum)
        sp.alloc_dp()
        sp.align_full()
        sp.realign_banded(10)
        assert sp.score_valid


# ── SeqPair: state-machine flags ──────────────────────────────────────────────

class TestSeqPairStateMachine:
    def test_initial_state_all_invalid(self, blosum):
        sp = nwgrad.SeqPair("ACDE", "ACDE", blosum)
        assert not sp.path_valid
        assert not sp.score_valid
        assert not sp.grad_valid

    def test_align_full_sets_path_and_score(self, blosum):
        sp = nwgrad.SeqPair("ACDE", "ACDE", blosum)
        sp.alloc_dp()
        sp.align_full()
        assert sp.path_valid
        assert sp.score_valid
        assert not sp.grad_valid

    def test_compute_grad_sets_grad_valid(self, blosum):
        sp = nwgrad.SeqPair("ACDE", "ACDE", blosum)
        sp.alloc_dp()
        sp.align_full()
        sp.compute_grad()
        assert sp.grad_valid

    def test_set_matrix_preserves_path_clears_score_and_grad(self, blosum, blosum2):
        sp = nwgrad.SeqPair("ACDE", "ACDE", blosum)
        sp.alloc_dp()
        sp.align_full()
        sp.compute_grad()
        sp.set_matrix(blosum2)
        assert sp.path_valid
        assert not sp.score_valid
        assert not sp.grad_valid

    def test_set_matrix_allows_realign_banded(self, blosum, blosum2):
        sp = nwgrad.SeqPair("ACDE", "ACDE", blosum)
        sp.alloc_dp()
        sp.align_full()
        sp.set_matrix(blosum2)
        # Must not raise — buffers are still allocated after set_matrix
        sp.realign_banded(10)
        assert sp.score_valid

    def test_set_matrix_score_updates_after_realign(self, blosum, blosum2):
        sp = nwgrad.SeqPair("ACDE", "ACDE", blosum)
        sp.alloc_dp()
        sp.align_full()
        score1 = sp.score
        sp.set_matrix(blosum2)
        sp.realign_banded(50)
        score2 = sp.score
        # Different matrix → different score (matrices differ by 0.5 per entry)
        assert score2 != pytest.approx(score1)

    def test_compute_grad_requires_score_valid(self, blosum):
        sp = nwgrad.SeqPair("ACDE", "ACDE", blosum)
        with pytest.raises(Exception):
            sp.compute_grad()

    def test_compute_grad_none_mode_raises(self, blosum):
        sp = nwgrad.SeqPair("ACDE", "ACDE", blosum, grad_mode="none")
        sp.alloc_dp()
        sp.align_full()
        with pytest.raises(Exception, match="grad_mode"):
            sp.compute_grad()

    def test_guide_j_none_before_align(self, blosum):
        sp = nwgrad.SeqPair("ACDE", "ACDE", blosum)
        assert sp.guide_j is None

    def test_guide_j_set_after_align(self, blosum):
        sp = nwgrad.SeqPair("ACDE", "ACDE", blosum)
        sp.alloc_dp()
        sp.align_full()
        gj = sp.guide_j
        assert gj is not None
        assert len(gj) == len("ACDE") + 1

    def test_guide_j_preserved_after_set_matrix(self, blosum, blosum2):
        sp = nwgrad.SeqPair("ACDE", "ACDE", blosum)
        sp.alloc_dp()
        sp.align_full()
        gj_before = list(sp.guide_j)
        sp.set_matrix(blosum2)
        assert list(sp.guide_j) == gj_before


# ── SeqPairBatch: sums match per-pair ─────────────────────────────────────────

class TestSeqPairBatchSums:
    @pytest.mark.parametrize("gap_model,mode,gap_open,gap_extend", ALL_CONFIGS)
    def test_align_full_sum_matches_individual(
            self, gap_model, mode, gap_open, gap_extend, blosum):
        batch = nwgrad.SeqPairBatch(n_threads=1)
        pairs_obj = []
        for a, b in PAIRS:
            sp = nwgrad.SeqPair(a, b, blosum,
                                 gap_open=gap_open, gap_extend=gap_extend,
                                 gap_model=gap_model, mode=mode, grad_mode="hard")
            pairs_obj.append(sp)
            batch.add(sp)

        batch.alloc_dp()
        total = batch.align_full()
        expected = sum(sp.score for sp in pairs_obj)
        assert total == pytest.approx(expected, rel=1e-10)

    @pytest.mark.parametrize("gap_model,mode,gap_open,gap_extend", ALL_CONFIGS)
    def test_compute_grad_sum_matches_individual(
            self, gap_model, mode, gap_open, gap_extend, blosum):
        batch = nwgrad.SeqPairBatch(n_threads=1)
        pairs_obj = []
        for a, b in PAIRS:
            sp = nwgrad.SeqPair(a, b, blosum,
                                 gap_open=gap_open, gap_extend=gap_extend,
                                 gap_model=gap_model, mode=mode, grad_mode="hard")
            pairs_obj.append(sp)
            batch.add(sp)

        batch.alloc_dp()
        batch.align_full()
        batch_grad = np.array(batch.compute_grad())

        expected = np.zeros((20, 20))
        for sp in pairs_obj:
            expected += np.array(sp.grad)

        np.testing.assert_allclose(batch_grad, expected, atol=1e-12)

    def test_realign_banded_sum_matches_individual(self, blosum):
        batch = nwgrad.SeqPairBatch(n_threads=1)
        pairs_obj = []
        for a, b in PAIRS:
            sp = nwgrad.SeqPair(a, b, blosum,
                                 gap_open=11.0, gap_extend=1.0,
                                 gap_model="affine", mode="global", grad_mode="hard")
            pairs_obj.append(sp)
            batch.add(sp)

        batch.alloc_dp()
        batch.align_full()
        bw = 20
        total = batch.realign_banded(bw)
        expected = sum(sp.score for sp in pairs_obj)
        assert total == pytest.approx(expected, rel=1e-10)


# ── SeqPairBatch vs BatchAligner ──────────────────────────────────────────────

class TestSeqPairBatchVsBatchAligner:
    @pytest.mark.parametrize("gap_model,mode,gap_open,gap_extend", ALL_CONFIGS)
    def test_scores_match_batch_aligner(
            self, gap_model, mode, gap_open, gap_extend, blosum):
        seqs_a = [p[0] for p in PAIRS]
        seqs_b = [p[1] for p in PAIRS]

        old_aligner = nwgrad.BatchAligner(
            matrix=blosum,
            gap_open=gap_open, gap_extend=gap_extend,
            gap_model=gap_model, mode=mode,
            grad_mode="hard", n_threads=1,
        )
        old_result = old_aligner.align(seqs_a, seqs_b)
        old_scores = np.array(old_result.scores)

        batch = nwgrad.SeqPairBatch(n_threads=1)
        pairs_obj = []
        for a, b in PAIRS:
            sp = nwgrad.SeqPair(a, b, blosum,
                                 gap_open=gap_open, gap_extend=gap_extend,
                                 gap_model=gap_model, mode=mode, grad_mode="hard")
            pairs_obj.append(sp)
            batch.add(sp)
        batch.alloc_dp()
        batch.align_full()

        new_scores = np.array([sp.score for sp in pairs_obj])
        np.testing.assert_allclose(new_scores, old_scores, atol=1e-10)

    @pytest.mark.parametrize("gap_model,mode,gap_open,gap_extend", ALL_CONFIGS)
    def test_grad_sum_matches_batch_aligner(
            self, gap_model, mode, gap_open, gap_extend, blosum):
        seqs_a = [p[0] for p in PAIRS]
        seqs_b = [p[1] for p in PAIRS]

        old_aligner = nwgrad.BatchAligner(
            matrix=blosum,
            gap_open=gap_open, gap_extend=gap_extend,
            gap_model=gap_model, mode=mode,
            grad_mode="hard", n_threads=1,
        )
        old_result = old_aligner.align(seqs_a, seqs_b)
        old_grad = np.array(old_result.grad)

        batch = nwgrad.SeqPairBatch(n_threads=1)
        for a, b in PAIRS:
            sp = nwgrad.SeqPair(a, b, blosum,
                                 gap_open=gap_open, gap_extend=gap_extend,
                                 gap_model=gap_model, mode=mode, grad_mode="hard")
            batch.add(sp)
        batch.alloc_dp()
        batch.align_full()
        new_grad = np.array(batch.compute_grad())

        np.testing.assert_allclose(new_grad, old_grad, atol=1e-10)


# ── SeqPairBatch: multi-thread matches single-thread ─────────────────────────

class TestSeqPairBatchThreading:
    @pytest.mark.parametrize("n_threads", [2, 4])
    def test_align_full_multithread_matches_single(self, n_threads, blosum):
        def make_batch(nt):
            b = nwgrad.SeqPairBatch(n_threads=nt)
            for a, seq_b in PAIRS * 3:
                sp = nwgrad.SeqPair(a, seq_b, blosum,
                                     gap_open=11.0, gap_extend=1.0,
                                     gap_model="affine", mode="global", grad_mode="hard")
                b.add(sp)
            return b

        b1 = make_batch(1)
        bm = make_batch(n_threads)

        b1.alloc_dp()
        bm.alloc_dp()
        total1 = b1.align_full()
        totalm = bm.align_full()

        assert totalm == pytest.approx(total1, rel=1e-10)

    @pytest.mark.parametrize("n_threads", [2, 4])
    def test_compute_grad_multithread_matches_single(self, n_threads, blosum):
        def make_batch(nt):
            b = nwgrad.SeqPairBatch(n_threads=nt)
            for a, seq_b in PAIRS * 3:
                sp = nwgrad.SeqPair(a, seq_b, blosum,
                                     gap_open=11.0, gap_extend=1.0,
                                     gap_model="affine", mode="global", grad_mode="hard")
                b.add(sp)
            return b

        b1 = make_batch(1)
        bm = make_batch(n_threads)
        b1.alloc_dp()
        bm.alloc_dp()
        b1.align_full()
        bm.align_full()

        g1 = np.array(b1.compute_grad())
        gm = np.array(bm.compute_grad())

        np.testing.assert_allclose(gm, g1, atol=1e-10)

    def test_set_matrix_then_realign_multithread(self, blosum, blosum2):
        batch = nwgrad.SeqPairBatch(n_threads=2)
        for a, b in PAIRS:
            sp = nwgrad.SeqPair(a, b, blosum,
                                 gap_open=11.0, gap_extend=1.0,
                                 gap_model="affine", mode="global", grad_mode="hard")
            batch.add(sp)

        batch.alloc_dp()
        batch.align_full()
        batch.set_matrix(blosum2)
        total = batch.realign_banded(30)
        assert isinstance(total, float)

        grad = np.array(batch.compute_grad())
        assert grad.shape == (20, 20)
        assert grad.sum() > 0


# ── SeqPairBatch.score_and_grad ───────────────────────────────────────────────

class TestScoreAndGrad:
    """score_and_grad uses thread-owned DpBuffers; pairs' dp_valid stays False."""

    def _make_batch(self, blosum, grad_mode="hard", n_threads=1):
        batch = nwgrad.SeqPairBatch(n_threads=n_threads)
        pairs_obj = []
        for a, b in PAIRS:
            sp = nwgrad.SeqPair(a, b, blosum,
                                 gap_open=11.0, gap_extend=1.0,
                                 gap_model="affine", mode="global",
                                 grad_mode=grad_mode)
            pairs_obj.append(sp)
            batch.add(sp)
        return batch, pairs_obj

    # ── score correctness ──────────────────────────────────────────────────────

    @pytest.mark.parametrize("gap_model,mode,gap_open,gap_extend", ALL_CONFIGS)
    def test_score_matches_align_full(self, gap_model, mode, gap_open, gap_extend, blosum):
        batch_ref = nwgrad.SeqPairBatch(n_threads=1)
        batch_new = nwgrad.SeqPairBatch(n_threads=1)
        for a, b in PAIRS:
            batch_ref.add(nwgrad.SeqPair(a, b, blosum,
                gap_open=gap_open, gap_extend=gap_extend,
                gap_model=gap_model, mode=mode, grad_mode="hard"))
            batch_new.add(nwgrad.SeqPair(a, b, blosum,
                gap_open=gap_open, gap_extend=gap_extend,
                gap_model=gap_model, mode=mode, grad_mode="hard"))

        batch_ref.alloc_dp()
        ref_total = batch_ref.align_full()
        new_total = batch_new.score_and_grad()
        assert new_total == pytest.approx(ref_total, rel=1e-10)

    def test_banded_score_matches_wide_realign(self, blosum):
        batch_ref = nwgrad.SeqPairBatch(n_threads=1)
        batch_new = nwgrad.SeqPairBatch(n_threads=1)
        bw = max(len(a) for a, _ in PAIRS) + max(len(b) for _, b in PAIRS) + 5
        for a, b in PAIRS:
            batch_ref.add(nwgrad.SeqPair(a, b, blosum,
                gap_open=11.0, gap_extend=1.0,
                gap_model="affine", mode="global", grad_mode="hard"))
            batch_new.add(nwgrad.SeqPair(a, b, blosum,
                gap_open=11.0, gap_extend=1.0,
                gap_model="affine", mode="global", grad_mode="hard"))

        batch_ref.alloc_dp()
        batch_ref.align_full()
        ref_total = batch_ref.realign_banded(bw)
        new_total = batch_new.score_and_grad(bandwidth=bw)
        assert new_total == pytest.approx(ref_total, rel=1e-10)

    # ── gradient correctness ───────────────────────────────────────────────────

    @pytest.mark.parametrize("gap_model,mode,gap_open,gap_extend", ALL_CONFIGS)
    def test_grad_sum_matches_old_api(self, gap_model, mode, gap_open, gap_extend, blosum):
        old_aligner = nwgrad.BatchAligner(
            matrix=blosum, gap_open=gap_open, gap_extend=gap_extend,
            gap_model=gap_model, mode=mode, grad_mode="hard", n_threads=1)
        old_grad = np.array(old_aligner.align(
            [p[0] for p in PAIRS], [p[1] for p in PAIRS]).grad)

        batch = nwgrad.SeqPairBatch(n_threads=1)
        for a, b in PAIRS:
            batch.add(nwgrad.SeqPair(a, b, blosum,
                gap_open=gap_open, gap_extend=gap_extend,
                gap_model=gap_model, mode=mode, grad_mode="hard"))
        batch.score_and_grad()
        new_grad = np.array(batch.compute_grad())

        np.testing.assert_allclose(new_grad, old_grad, atol=1e-10)

    def test_banded_grad_matches_wide_realign(self, blosum):
        bw = max(len(a) for a, _ in PAIRS) + max(len(b) for _, b in PAIRS) + 5

        batch_ref = nwgrad.SeqPairBatch(n_threads=1)
        batch_new = nwgrad.SeqPairBatch(n_threads=1)
        for a, b in PAIRS:
            batch_ref.add(nwgrad.SeqPair(a, b, blosum,
                gap_open=11.0, gap_extend=1.0,
                gap_model="affine", mode="global", grad_mode="hard"))
            batch_new.add(nwgrad.SeqPair(a, b, blosum,
                gap_open=11.0, gap_extend=1.0,
                gap_model="affine", mode="global", grad_mode="hard"))

        batch_ref.alloc_dp()
        batch_ref.align_full()
        batch_ref.compute_grad()
        ref_grad = np.array(batch_ref.compute_grad())

        batch_new.score_and_grad(bandwidth=bw)
        new_grad = np.array(batch_new.compute_grad())

        np.testing.assert_allclose(new_grad, ref_grad, atol=1e-10)

    def test_soft_grad_matches_old_api(self, blosum):
        old_aligner = nwgrad.BatchAligner(
            matrix=blosum, gap_open=11.0, gap_extend=1.0,
            gap_model="affine", mode="global", grad_mode="soft", n_threads=1)
        old_result = old_aligner.align(
            [p[0] for p in PAIRS], [p[1] for p in PAIRS])
        old_grad = np.array(old_result.grad)
        old_total = float(np.array(old_result.scores).sum())

        batch = nwgrad.SeqPairBatch(n_threads=1)
        for a, b in PAIRS:
            batch.add(nwgrad.SeqPair(a, b, blosum,
                gap_open=11.0, gap_extend=1.0,
                gap_model="affine", mode="global", grad_mode="soft"))
        new_total = batch.score_and_grad()
        new_grad = np.array(batch.compute_grad())

        assert new_total == pytest.approx(old_total, rel=1e-10)
        np.testing.assert_allclose(new_grad, old_grad, atol=1e-10)

    # ── state flags ───────────────────────────────────────────────────────────

    def test_dp_valid_stays_false(self, blosum):
        batch, pairs_obj = self._make_batch(blosum)
        batch.score_and_grad()
        assert not any(sp.dp_valid for sp in pairs_obj)

    def test_score_valid_set(self, blosum):
        batch, pairs_obj = self._make_batch(blosum)
        batch.score_and_grad()
        assert all(sp.score_valid for sp in pairs_obj)

    def test_path_valid_set(self, blosum):
        batch, pairs_obj = self._make_batch(blosum)
        batch.score_and_grad()
        assert all(sp.path_valid for sp in pairs_obj)

    def test_grad_valid_set_for_hard_mode(self, blosum):
        batch, pairs_obj = self._make_batch(blosum, grad_mode="hard")
        batch.score_and_grad()
        assert all(sp.grad_valid for sp in pairs_obj)

    def test_grad_valid_false_for_none_mode(self, blosum):
        batch, pairs_obj = self._make_batch(blosum, grad_mode="none")
        batch.score_and_grad()
        assert not any(sp.grad_valid for sp in pairs_obj)

    def test_compute_grad_after_score_and_grad_sums_cached(self, blosum):
        """compute_grad() after score_and_grad() reuses cached grads (no DP needed)."""
        batch, pairs_obj = self._make_batch(blosum)
        batch.score_and_grad()
        # All grads already cached — compute_grad must not throw despite dp_valid=False.
        grad_sum = np.array(batch.compute_grad())
        expected = sum(np.array(sp.grad) for sp in pairs_obj)
        np.testing.assert_allclose(grad_sum, expected, atol=1e-12)

    # ── threading ─────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("n_threads", [2, 4])
    def test_multithread_score_matches_single(self, n_threads, blosum):
        def make(nt):
            b = nwgrad.SeqPairBatch(n_threads=nt)
            for a, seq_b in PAIRS * 3:
                b.add(nwgrad.SeqPair(a, seq_b, blosum,
                    gap_open=11.0, gap_extend=1.0,
                    gap_model="affine", mode="global", grad_mode="hard"))
            return b

        assert make(1).score_and_grad() == pytest.approx(
            make(n_threads).score_and_grad(), rel=1e-10)

    @pytest.mark.parametrize("n_threads", [2, 4])
    def test_multithread_grad_matches_single(self, n_threads, blosum):
        def make_and_run(nt):
            b = nwgrad.SeqPairBatch(n_threads=nt)
            for a, seq_b in PAIRS * 3:
                b.add(nwgrad.SeqPair(a, seq_b, blosum,
                    gap_open=11.0, gap_extend=1.0,
                    gap_model="affine", mode="global", grad_mode="hard"))
            b.score_and_grad()
            return np.array(b.compute_grad())

        np.testing.assert_allclose(
            make_and_run(n_threads), make_and_run(1), atol=1e-10)

    # ── guide_j is set from full DP even when banded ──────────────────────────

    def test_guide_j_set_after_score_and_grad(self, blosum):
        batch, pairs_obj = self._make_batch(blosum)
        batch.score_and_grad(bandwidth=15)
        for sp in pairs_obj:
            gj = sp.guide_j
            assert gj is not None
            assert len(gj) == len(sp.seq_a) + 1

    def test_guide_j_matches_align_full(self, blosum):
        """guide_j from score_and_grad should equal guide_j from align_full."""
        sp_new = nwgrad.SeqPair("PLEASANTLY", "MEANLY", blosum,
                                 gap_open=11.0, gap_extend=1.0,
                                 gap_model="affine", mode="global", grad_mode="hard")
        sp_ref = nwgrad.SeqPair("PLEASANTLY", "MEANLY", blosum,
                                 gap_open=11.0, gap_extend=1.0,
                                 gap_model="affine", mode="global", grad_mode="hard")

        batch = nwgrad.SeqPairBatch(n_threads=1)
        batch.add(sp_new)
        batch.score_and_grad()
        sp_ref.alloc_dp()
        sp_ref.align_full()

        assert list(sp_new.guide_j) == list(sp_ref.guide_j)


# ── drop_dp: SeqPair ─────────────────────────────────────────────────────────

class TestDropDp:
    def test_dp_invalid_initially(self, blosum):
        sp = nwgrad.SeqPair("ACDE", "ACDE", blosum)
        assert not sp.dp_valid

    def test_dp_valid_after_align_full(self, blosum):
        sp = nwgrad.SeqPair("ACDE", "ACDE", blosum)
        sp.alloc_dp()
        sp.align_full()
        assert sp.dp_valid

    def test_dp_valid_after_realign_banded(self, blosum):
        sp = nwgrad.SeqPair("ACDE", "ACDE", blosum)
        sp.alloc_dp()
        sp.align_full()
        sp.realign_banded(10)
        assert sp.dp_valid

    def test_drop_dp_clears_dp_valid(self, blosum):
        sp = nwgrad.SeqPair("ACDE", "ACDE", blosum)
        sp.alloc_dp()
        sp.align_full()
        sp.drop_dp()
        assert not sp.dp_valid

    def test_drop_dp_preserves_score(self, blosum):
        sp = nwgrad.SeqPair("PLEASANTLY", "MEANLY", blosum,
                             gap_open=11.0, gap_extend=1.0,
                             gap_model="affine", mode="global")
        sp.alloc_dp()
        sp.align_full()
        score_before = sp.score
        sp.drop_dp()
        assert sp.score_valid
        assert sp.score == pytest.approx(score_before)

    def test_drop_dp_preserves_grad(self, blosum):
        sp = nwgrad.SeqPair("PLEASANTLY", "MEANLY", blosum,
                             gap_open=11.0, gap_extend=1.0,
                             gap_model="affine", mode="global", grad_mode="hard")
        sp.alloc_dp()
        sp.align_full()
        sp.compute_grad()
        grad_before = np.array(sp.grad).copy()
        sp.drop_dp()
        assert sp.grad_valid
        np.testing.assert_array_equal(np.array(sp.grad), grad_before)

    def test_drop_dp_preserves_guide_j(self, blosum):
        sp = nwgrad.SeqPair("PLEASANTLY", "MEANLY", blosum)
        sp.alloc_dp()
        sp.align_full()
        gj_before = list(sp.guide_j)
        sp.drop_dp()
        assert sp.path_valid
        assert list(sp.guide_j) == gj_before

    def test_drop_dp_blocks_compute_grad(self, blosum):
        sp = nwgrad.SeqPair("ACDE", "ACDE", blosum)
        sp.alloc_dp()
        sp.align_full()
        sp.drop_dp()
        with pytest.raises(Exception, match="dropped"):
            sp.compute_grad()

    def test_drop_dp_then_align_full_reenables_compute_grad(self, blosum):
        sp = nwgrad.SeqPair("ACDE", "ACDE", blosum)
        sp.alloc_dp()
        sp.align_full()
        sp.drop_dp()
        sp.alloc_dp()
        sp.align_full()
        assert sp.dp_valid
        sp.compute_grad()  # must not raise
        assert sp.grad_valid

    def test_drop_dp_then_realign_banded_reenables_compute_grad(self, blosum):
        sp = nwgrad.SeqPair("ACDE", "ACDE", blosum)
        sp.alloc_dp()
        sp.align_full()
        sp.drop_dp()
        sp.alloc_dp()
        sp.realign_banded(10)
        assert sp.dp_valid
        sp.compute_grad()
        assert sp.grad_valid

    def test_drop_dp_before_compute_grad_score_still_accessible(self, blosum):
        """drop_dp without ever calling compute_grad — score and path remain."""
        sp = nwgrad.SeqPair("ACDE", "ACDE", blosum, grad_mode="none")
        sp.alloc_dp()
        sp.align_full()
        sp.drop_dp()
        assert sp.score is not None
        assert sp.guide_j is not None

    def test_drop_dp_idempotent(self, blosum):
        sp = nwgrad.SeqPair("ACDE", "ACDE", blosum)
        sp.alloc_dp()
        sp.align_full()
        sp.drop_dp()
        sp.drop_dp()  # second call must not crash
        assert not sp.dp_valid

    @pytest.mark.parametrize("gap_model,mode,gap_open,gap_extend", ALL_CONFIGS)
    def test_drop_dp_then_realign_score_matches_fresh(
            self, gap_model, mode, gap_open, gap_extend, blosum):
        """After drop, realign_banded score matches a fresh SeqPair."""
        a, b = "PLEASANTLY", "MEANLY"
        sp = nwgrad.SeqPair(a, b, blosum,
                             gap_open=gap_open, gap_extend=gap_extend,
                             gap_model=gap_model, mode=mode, grad_mode="hard")
        sp.alloc_dp()
        sp.align_full()
        sp.drop_dp()
        bw = max(len(a), len(b)) + 5
        sp.alloc_dp()
        sp.realign_banded(bw)
        sp.compute_grad()

        sp2 = nwgrad.SeqPair(a, b, blosum,
                              gap_open=gap_open, gap_extend=gap_extend,
                              gap_model=gap_model, mode=mode, grad_mode="hard")
        sp2.alloc_dp()
        sp2.align_full()
        sp2.compute_grad()

        assert sp.score == pytest.approx(sp2.score, rel=1e-10)
        np.testing.assert_allclose(np.array(sp.grad), np.array(sp2.grad), atol=1e-12)


# ── drop_dp: SeqPairBatch ─────────────────────────────────────────────────────

class TestDropDpBatch:
    def _make_batch(self, blosum, n_threads=1):
        batch = nwgrad.SeqPairBatch(n_threads=n_threads)
        pairs_obj = []
        for a, b in PAIRS:
            sp = nwgrad.SeqPair(a, b, blosum,
                                 gap_open=11.0, gap_extend=1.0,
                                 gap_model="affine", mode="global", grad_mode="hard")
            pairs_obj.append(sp)
            batch.add(sp)
        return batch, pairs_obj

    def test_batch_drop_dp_clears_dp_valid_on_all(self, blosum):
        batch, pairs_obj = self._make_batch(blosum)
        batch.alloc_dp()
        batch.align_full()
        assert all(sp.dp_valid for sp in pairs_obj)
        batch.drop_dp()
        assert not any(sp.dp_valid for sp in pairs_obj)

    def test_batch_drop_dp_preserves_scores(self, blosum):
        batch, pairs_obj = self._make_batch(blosum)
        batch.alloc_dp()
        batch.align_full()
        scores_before = [sp.score for sp in pairs_obj]
        batch.drop_dp()
        for sp, s in zip(pairs_obj, scores_before):
            assert sp.score == pytest.approx(s)

    def test_batch_drop_dp_preserves_grads(self, blosum):
        batch, pairs_obj = self._make_batch(blosum)
        batch.alloc_dp()
        batch.align_full()
        batch.compute_grad()
        grads_before = [np.array(sp.grad).copy() for sp in pairs_obj]
        batch.drop_dp()
        for sp, g in zip(pairs_obj, grads_before):
            np.testing.assert_array_equal(np.array(sp.grad), g)

    def test_batch_drop_dp_blocks_compute_grad(self, blosum):
        batch, pairs_obj = self._make_batch(blosum)
        batch.alloc_dp()
        batch.align_full()
        batch.drop_dp()
        with pytest.raises(Exception, match="dropped"):
            batch.compute_grad()

    def test_batch_drop_dp_then_realign_restores(self, blosum):
        batch, pairs_obj = self._make_batch(blosum)
        batch.alloc_dp()
        batch.align_full()
        batch.drop_dp()
        batch.alloc_dp()
        batch.realign_banded(30)
        assert all(sp.dp_valid for sp in pairs_obj)
        grad = np.array(batch.compute_grad())
        assert grad.shape == (20, 20)
        assert grad.sum() > 0

    def test_batch_drop_dp_multithreaded(self, blosum):
        batch, pairs_obj = self._make_batch(blosum, n_threads=4)
        batch.alloc_dp()
        batch.align_full()
        batch.drop_dp()
        assert not any(sp.dp_valid for sp in pairs_obj)


# ── SeqPairBatch: API surface ─────────────────────────────────────────────────

class TestSeqPairBatchAPI:
    def test_len(self, blosum):
        batch = nwgrad.SeqPairBatch()
        assert len(batch) == 0
        for a, b in PAIRS:
            batch.add(nwgrad.SeqPair(a, b, blosum))
        assert len(batch) == len(PAIRS)

    def test_getitem(self, blosum):
        batch = nwgrad.SeqPairBatch()
        sp = nwgrad.SeqPair("ACDE", "ACDE", blosum)
        batch.add(sp)
        assert batch[0].seq_a == "ACDE"

    def test_getitem_negative(self, blosum):
        batch = nwgrad.SeqPairBatch()
        for a, b in PAIRS:
            batch.add(nwgrad.SeqPair(a, b, blosum))
        assert batch[-1].seq_a == PAIRS[-1][0]

    def test_getitem_out_of_range(self, blosum):
        batch = nwgrad.SeqPairBatch()
        with pytest.raises(Exception):
            _ = batch[0]

    def test_n_threads_default_positive(self, blosum):
        batch = nwgrad.SeqPairBatch()
        assert batch.n_threads >= 1

    def test_n_threads_explicit(self, blosum):
        batch = nwgrad.SeqPairBatch(n_threads=3)
        assert batch.n_threads == 3

    def test_grad_shape(self, blosum):
        batch = nwgrad.SeqPairBatch(n_threads=1)
        batch.add(nwgrad.SeqPair("ACDE", "ACDE", blosum))
        batch.alloc_dp()
        batch.align_full()
        g = np.array(batch.compute_grad())
        assert g.shape == (20, 20)

    def test_empty_batch_align_full_returns_zero(self, blosum):
        batch = nwgrad.SeqPairBatch(n_threads=1)
        assert batch.align_full() == pytest.approx(0.0)

    def test_empty_batch_realign_banded_returns_zero(self, blosum):
        batch = nwgrad.SeqPairBatch(n_threads=1)
        assert batch.realign_banded(10) == pytest.approx(0.0)

    def test_empty_batch_compute_grad_returns_zeros(self, blosum):
        batch = nwgrad.SeqPairBatch(n_threads=1)
        g = np.array(batch.compute_grad())
        assert g.shape == (20, 20)
        assert g.sum() == pytest.approx(0.0)
