"""Tests for SeqPair and SeqPairBatch, comparing against single-pair and
streaming (SeqPairBatch.align, formerly BatchAligner) APIs.

Coverage:
  - SeqPairBatch.score_and_grad() vs ref API; dp_valid stays False; compute_grad() after
  - SeqPair.drop_dp() / SeqPairBatch.drop_dp() lifecycle and memory release
  - SeqPair.align_full() score vs nw_score / sw_score / nw_score_affine / sw_score_affine
  - SeqPair hard grad vs nw_grad / sw_grad / nw_affine_grad / sw_affine_grad
  - SeqPair soft score (log Z) and grad vs nw_soft_grad / sw_soft_grad / ...
  - SeqPair.realign_banded() with wide band around exact guide matches full DP
  - State-machine flags: path_valid / score_valid / grad_valid lifecycle
  - set_params() preserves path_valid, clears score_valid + grad_valid
  - SeqPairBatch sums match individual SeqPair results
  - SeqPairBatch matches BatchAligner (hard grad mode)
  - SeqPairBatch multi-thread matches single-thread
  - Error handling: premature calls raise; alloc_dp is a deprecated no-op
"""

import numpy as np
import pytest
import nwgrad
from conftest import StreamAligner, StreamAlignerDouble
from test_subst_matrix import BLOSUM62


# ── helpers ───────────────────────────────────────────────────────────────────

def make_params(gap_extend, gap_open=0.0, matrix_arr=None):
    arr = BLOSUM62 if matrix_arr is None else matrix_arr
    return nwgrad.AlignParams(arr, gap_open_a=gap_open, gap_extend_a=gap_extend,
                                   gap_open_b=gap_open, gap_extend_b=gap_extend)


def grad_matrix(g):
    return g.matrix.to_matrix()


def ref_score(a, b, gap_model, mode, gap_open, gap_extend):
    p = make_params(gap_extend, gap_open)
    if gap_model == "linear" and mode == "global":
        return nwgrad.nw_score(a, b, p)
    if gap_model == "linear" and mode == "local":
        return nwgrad.sw_score(a, b, p)
    if gap_model == "affine" and mode == "global":
        return nwgrad.nw_score_affine(a, b, p)
    return nwgrad.sw_score_affine(a, b, p)


def ref_hard_grad(a, b, gap_model, mode, gap_open, gap_extend):
    p = make_params(gap_extend, gap_open)
    if gap_model == "linear" and mode == "global":
        return nwgrad.nw_grad(a, b, p)
    if gap_model == "linear" and mode == "local":
        return nwgrad.sw_grad(a, b, p)
    if gap_model == "affine" and mode == "global":
        return nwgrad.nw_affine_grad(a, b, p)
    return nwgrad.sw_affine_grad(a, b, p)


def ref_soft_grad(a, b, gap_model, mode, gap_open, gap_extend):
    p = make_params(gap_extend, gap_open)
    if gap_model == "linear" and mode == "global":
        return nwgrad.nw_soft_grad(a, b, p)
    if gap_model == "linear" and mode == "local":
        return nwgrad.sw_soft_grad(a, b, p)
    if gap_model == "affine" and mode == "global":
        return nwgrad.nw_affine_soft_grad(a, b, p)
    return nwgrad.sw_affine_soft_grad(a, b, p)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def blosum():
    return nwgrad.AlignParams(BLOSUM62,
                               gap_open_a=0.0, gap_extend_a=1.0,
                               gap_open_b=0.0, gap_extend_b=1.0)


@pytest.fixture(scope="module")
def blosum2():
    """A slightly perturbed matrix for set_params() tests."""
    return nwgrad.AlignParams(BLOSUM62 + 0.5,
                               gap_open_a=0.0, gap_extend_a=1.0,
                               gap_open_b=0.0, gap_extend_b=1.0)


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


# ── SeqPair: allocation lifecycle ─────────────────────────────────────────────

class TestExplicitAlloc:
    """alloc_dp() was required before align_full(); since 0.6 there are no pair-owned
    tables, and it is a deprecated no-op."""

    def test_align_full_without_alloc_works(self, blosum):
        sp = nwgrad.SeqPair("ACDE", "ACDE", blosum)
        sp.align_full()
        assert sp.score_valid

    def test_alloc_dp_is_deprecated(self, blosum):
        sp = nwgrad.SeqPair("ACDE", "ACDE", blosum)
        with pytest.warns(DeprecationWarning):
            sp.alloc_dp()

    def test_realign_banded_after_drop_dp_requires_alloc(self, blosum):
        sp = nwgrad.SeqPair("ACDE", "ACDE", blosum)
        sp.align_full()
        sp.drop_dp()
        sp.realign_banded(10)
        assert sp.score_valid

    def test_align_full_after_alloc_succeeds(self, blosum):
        sp = nwgrad.SeqPair("ACDE", "ACDE", blosum)
        sp.align_full()
        assert sp.score_valid

    def test_batch_deprecated_alloc_and_align_full_still_work(self, blosum):
        batch = nwgrad.SeqPairBatch(n_threads=1, gap_model="linear")
        batch.add_many(["ACDE"], ["ACDE"], blosum)
        with pytest.warns(DeprecationWarning):
            batch.alloc_dp()
        with pytest.warns(DeprecationWarning):
            batch.align_full()
        assert batch[0].dp_valid and not batch[0].grad_valid   # held, as before
        batch.compute_grad()
        assert batch[0].grad_valid


# ── SeqPair: align_full score ─────────────────────────────────────────────────

class TestSeqPairScore:
    @pytest.mark.parametrize("gap_model,mode,gap_open,gap_extend", ALL_CONFIGS)
    @pytest.mark.parametrize("a,b", PAIRS)
    def test_align_full_score_matches_ref(self, a, b, gap_model, mode, gap_open, gap_extend):
        p = make_params(gap_extend, gap_open)
        sp = nwgrad.SeqPair(a, b, p, gap_model=gap_model, mode=mode, grad_mode="none")
        sp.align_full()
        expected = ref_score(a, b, gap_model, mode, gap_open, gap_extend)
        assert sp.score == pytest.approx(expected), (
            f"{gap_model}/{mode} pair ({a!r},{b!r}): got {sp.score}, expected {expected}"
        )

    def test_score_none_before_align(self, blosum):
        sp = nwgrad.SeqPair("ACDE", "ACDE", blosum)
        assert sp.score is None

    def test_score_valid_after_align(self, blosum):
        sp = nwgrad.SeqPair("ACDE", "ACDE", blosum)
        assert not sp.score_valid
        sp.align_full()
        assert sp.score_valid


# ── SeqPair: hard gradient ────────────────────────────────────────────────────

class TestSeqPairHardGrad:
    @pytest.mark.parametrize("gap_model,mode,gap_open,gap_extend", ALL_CONFIGS)
    @pytest.mark.parametrize("a,b", PAIRS)
    def test_hard_grad_matches_ref(self, a, b, gap_model, mode, gap_open, gap_extend):
        p = make_params(gap_extend, gap_open)
        sp = nwgrad.SeqPair(a, b, p, gap_model=gap_model, mode=mode, grad_mode="hard")
        sp.align_full()
        sp.compute_grad()

        exp_score, exp_g = ref_hard_grad(a, b, gap_model, mode, gap_open, gap_extend)

        assert sp.score == pytest.approx(exp_score)
        np.testing.assert_allclose(
            grad_matrix(sp.grad), grad_matrix(exp_g), atol=1e-12,
            err_msg=f"hard grad mismatch for {gap_model}/{mode} ({a!r},{b!r})"
        )

    def test_grad_none_before_compute(self, blosum):
        sp = nwgrad.SeqPair("ACDE", "ACDE", blosum)
        sp.align_full()
        assert sp.grad is None

    def test_grad_valid_after_compute(self, blosum):
        sp = nwgrad.SeqPair("ACDE", "ACDE", blosum)
        sp.align_full()
        sp.compute_grad()
        assert sp.grad_valid
        assert sp.grad is not None

    def test_grad_shape(self, blosum):
        sp = nwgrad.SeqPair("ACDE", "ACDE", blosum)
        sp.align_full()
        sp.compute_grad()
        assert grad_matrix(sp.grad).shape == (20, 20)


# ── SeqPair: soft gradient ────────────────────────────────────────────────────

class TestSeqPairSoftGrad:
    @pytest.mark.parametrize("gap_model,mode,gap_open,gap_extend", ALL_CONFIGS)
    @pytest.mark.parametrize("a,b", PAIRS)
    def test_soft_score_and_grad_match_ref(self, a, b, gap_model, mode, gap_open, gap_extend):
        p = make_params(gap_extend, gap_open)
        sp = nwgrad.SeqPair(a, b, p, gap_model=gap_model, mode=mode, grad_mode="soft")
        sp.align_full()
        sp.compute_grad()

        exp_score, exp_g = ref_soft_grad(a, b, gap_model, mode, gap_open, gap_extend)

        assert sp.score == pytest.approx(exp_score, rel=1e-10), (
            f"soft score mismatch for {gap_model}/{mode} ({a!r},{b!r}): "
            f"got {sp.score}, expected {exp_score}"
        )
        np.testing.assert_allclose(
            grad_matrix(sp.grad), grad_matrix(exp_g), atol=1e-10,
            err_msg=f"soft grad mismatch for {gap_model}/{mode} ({a!r},{b!r})"
        )


# ── SeqPair: banded realignment ───────────────────────────────────────────────

class TestSeqPairBanded:
    @pytest.mark.parametrize("gap_model,mode,gap_open,gap_extend", ALL_CONFIGS)
    @pytest.mark.parametrize("a,b", PAIRS)
    def test_realign_banded_wide_band_matches_full(
            self, a, b, gap_model, mode, gap_open, gap_extend):
        p = make_params(gap_extend, gap_open)
        sp = nwgrad.SeqPair(a, b, p, gap_model=gap_model, mode=mode, grad_mode="hard")
        sp.align_full()
        full_score = sp.score

        bw = max(len(a), len(b)) + 5
        sp.realign_banded(bw)
        assert sp.score == pytest.approx(full_score, rel=1e-10), (
            f"realign_banded(wide) score mismatch for {gap_model}/{mode} ({a!r},{b!r})"
        )

    @pytest.mark.parametrize("gap_model,mode,gap_open,gap_extend", ALL_CONFIGS)
    @pytest.mark.parametrize("a,b", PAIRS)
    def test_realign_banded_hard_grad_matches_full(
            self, a, b, gap_model, mode, gap_open, gap_extend):
        p = make_params(gap_extend, gap_open)
        sp = nwgrad.SeqPair(a, b, p, gap_model=gap_model, mode=mode, grad_mode="hard")
        sp.align_full()
        sp.compute_grad()
        full_grad = grad_matrix(sp.grad).copy()

        bw = max(len(a), len(b)) + 5
        sp.realign_banded(bw)
        sp.compute_grad()
        np.testing.assert_allclose(
            grad_matrix(sp.grad), full_grad, atol=1e-12,
            err_msg=f"realign_banded grad mismatch for {gap_model}/{mode} ({a!r},{b!r})"
        )

    def test_realign_banded_requires_path(self, blosum):
        sp = nwgrad.SeqPair("ACDE", "ACDE", blosum)
        with pytest.raises(Exception, match="align_full"):
            sp.realign_banded(5)

    def test_realign_banded_clears_grad_valid(self, blosum):
        sp = nwgrad.SeqPair("ACDE", "ACDE", blosum)
        sp.align_full()
        sp.compute_grad()
        assert sp.grad_valid
        sp.realign_banded(10)
        assert not sp.grad_valid

    def test_realign_banded_sets_score_valid(self, blosum):
        sp = nwgrad.SeqPair("ACDE", "ACDE", blosum)
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
        sp.align_full()
        assert sp.path_valid
        assert sp.score_valid
        assert not sp.grad_valid

    def test_compute_grad_sets_grad_valid(self, blosum):
        sp = nwgrad.SeqPair("ACDE", "ACDE", blosum)
        sp.align_full()
        sp.compute_grad()
        assert sp.grad_valid

    def test_set_params_preserves_path_clears_score_and_grad(self, blosum, blosum2):
        sp = nwgrad.SeqPair("ACDE", "ACDE", blosum)
        sp.align_full()
        sp.compute_grad()
        sp.set_params(blosum2)
        assert sp.path_valid
        assert not sp.score_valid
        assert not sp.grad_valid

    def test_set_params_allows_realign_banded(self, blosum, blosum2):
        sp = nwgrad.SeqPair("ACDE", "ACDE", blosum)
        sp.align_full()
        sp.set_params(blosum2)
        sp.realign_banded(10)
        assert sp.score_valid

    def test_set_params_score_updates_after_realign(self, blosum, blosum2):
        sp = nwgrad.SeqPair("ACDE", "ACDE", blosum)
        sp.align_full()
        score1 = sp.score
        sp.set_params(blosum2)
        sp.realign_banded(50)
        score2 = sp.score
        assert score2 != pytest.approx(score1)

    def test_compute_grad_requires_score_valid(self, blosum):
        sp = nwgrad.SeqPair("ACDE", "ACDE", blosum)
        with pytest.raises(Exception):
            sp.compute_grad()

    def test_compute_grad_none_mode_raises(self, blosum):
        sp = nwgrad.SeqPair("ACDE", "ACDE", blosum, grad_mode="none")
        sp.align_full()
        with pytest.raises(Exception, match="grad_mode"):
            sp.compute_grad()

    def test_guide_j_none_before_align(self, blosum):
        sp = nwgrad.SeqPair("ACDE", "ACDE", blosum)
        assert sp.guide_j is None

    def test_guide_j_set_after_align(self, blosum):
        sp = nwgrad.SeqPair("ACDE", "ACDE", blosum)
        sp.align_full()
        gj = sp.guide_j
        assert gj is not None
        assert len(gj) == len("ACDE") + 1

    def test_guide_j_preserved_after_set_params(self, blosum, blosum2):
        sp = nwgrad.SeqPair("ACDE", "ACDE", blosum)
        sp.align_full()
        gj_before = list(sp.guide_j)
        sp.set_params(blosum2)
        assert list(sp.guide_j) == gj_before


# ── SeqPairBatch: sums match per-pair ─────────────────────────────────────────

class TestSeqPairBatchSums:
    @pytest.mark.parametrize("gap_model,mode,gap_open,gap_extend", ALL_CONFIGS)
    def test_align_full_sum_matches_individual(self, gap_model, mode, gap_open, gap_extend):
        p = make_params(gap_extend, gap_open)
        batch = nwgrad.SeqPairBatch(n_threads=1, gap_model=gap_model, mode=mode,
                                  grad_mode="hard")
        batch.add_many([x for x, _ in PAIRS], [y for _, y in PAIRS], p)
        pairs_obj = list(batch)

        total = batch.align_full()
        expected = sum(sp.score for sp in pairs_obj)
        assert total == pytest.approx(expected, rel=1e-10)

    @pytest.mark.parametrize("gap_model,mode,gap_open,gap_extend", ALL_CONFIGS)
    def test_compute_grad_sum_matches_individual(self, gap_model, mode, gap_open, gap_extend):
        p = make_params(gap_extend, gap_open)
        batch = nwgrad.SeqPairBatch(n_threads=1, gap_model=gap_model, mode=mode,
                                  grad_mode="hard")
        batch.add_many([x for x, _ in PAIRS], [y for _, y in PAIRS], p)
        pairs_obj = list(batch)

        batch.align_full()
        batch_grad = grad_matrix(batch.compute_grad())

        expected = np.zeros((20, 20))
        for sp in pairs_obj:
            expected += grad_matrix(sp.grad)

        np.testing.assert_allclose(batch_grad, expected, atol=1e-12)

    def test_realign_banded_sum_matches_individual(self):
        p = make_params(1.0, 11.0)
        batch = nwgrad.SeqPairBatch(n_threads=1, gap_model="affine", mode="global",
                                  grad_mode="hard")
        batch.add_many([x for x, _ in PAIRS], [y for _, y in PAIRS], p)
        pairs_obj = list(batch)

        batch.align_full()
        bw = 20
        total = batch.realign_banded(bw)
        expected = sum(sp.score for sp in pairs_obj)
        assert total == pytest.approx(expected, rel=1e-10)


# ── SeqPairBatch vs BatchAligner ──────────────────────────────────────────────

class TestSeqPairBatchVsBatchAligner:
    @pytest.mark.parametrize("gap_model,mode,gap_open,gap_extend", ALL_CONFIGS)
    def test_scores_match_batch_aligner(self, gap_model, mode, gap_open, gap_extend):
        seqs_a = [pr[0] for pr in PAIRS]
        seqs_b = [pr[1] for pr in PAIRS]
        p = make_params(gap_extend, gap_open)

        ref_aligner = StreamAligner(
            params=p, gap_model=gap_model, mode=mode, grad_mode="hard", n_threads=1,
        )
        ref_result = ref_aligner.align(seqs_a, seqs_b)
        ref_scores = np.array(ref_result.scores)

        batch = nwgrad.SeqPairBatch(n_threads=1, gap_model=gap_model, mode=mode,
                                  grad_mode="hard")
        batch.add_many([x for x, _ in PAIRS], [y for _, y in PAIRS], p)
        pairs_obj = list(batch)
        batch.align_full()

        new_scores = np.array([sp.score for sp in pairs_obj])
        np.testing.assert_allclose(new_scores, ref_scores, atol=1e-10)

    @pytest.mark.parametrize("gap_model,mode,gap_open,gap_extend", ALL_CONFIGS)
    def test_grad_sum_matches_batch_aligner(self, gap_model, mode, gap_open, gap_extend):
        seqs_a = [pr[0] for pr in PAIRS]
        seqs_b = [pr[1] for pr in PAIRS]
        p = make_params(gap_extend, gap_open)

        ref_aligner = StreamAligner(
            params=p, gap_model=gap_model, mode=mode, grad_mode="hard", n_threads=1,
        )
        ref_result = ref_aligner.align(seqs_a, seqs_b)
        ref_grad = ref_result.grad.matrix.to_matrix()

        batch = nwgrad.SeqPairBatch(n_threads=1, gap_model=gap_model, mode=mode,
                                  grad_mode="hard")
        batch.add_many([x for x, _ in PAIRS], [y for _, y in PAIRS], p)
        batch.align_full()
        new_grad = grad_matrix(batch.compute_grad())

        np.testing.assert_allclose(new_grad, ref_grad, atol=1e-10)


# ── SeqPairBatch: multi-thread matches single-thread ─────────────────────────

class TestSeqPairBatchThreading:
    @pytest.mark.parametrize("n_threads", [2, 4])
    def test_align_full_multithread_matches_single(self, n_threads):
        def make_batch(nt):
            p = make_params(1.0, 11.0)
            b = nwgrad.SeqPairBatch(n_threads=nt, gap_model="affine", mode="global",
                                      grad_mode="hard")
            b.add_many([x for x, _ in PAIRS * 3], [y for _, y in PAIRS * 3], p)
            return b

        b1 = make_batch(1)
        bm = make_batch(n_threads)

        total1 = b1.align_full()
        totalm = bm.align_full()

        assert totalm == pytest.approx(total1, rel=1e-10)

    @pytest.mark.parametrize("n_threads", [2, 4])
    def test_compute_grad_multithread_matches_single(self, n_threads):
        def make_batch(nt):
            p = make_params(1.0, 11.0)
            b = nwgrad.SeqPairBatch(n_threads=nt, gap_model="affine", mode="global",
                                      grad_mode="hard")
            b.add_many([x for x, _ in PAIRS * 3], [y for _, y in PAIRS * 3], p)
            return b

        b1 = make_batch(1)
        bm = make_batch(n_threads)
        b1.align_full()
        bm.align_full()

        g1 = grad_matrix(b1.compute_grad())
        gm = grad_matrix(bm.compute_grad())

        np.testing.assert_allclose(gm, g1, atol=1e-10)

    def test_set_params_then_realign_multithread(self, blosum2):
        p = make_params(1.0, 11.0)
        batch = nwgrad.SeqPairBatch(n_threads=2, gap_model="affine", mode="global",
                                  grad_mode="hard")
        batch.add_many([x for x, _ in PAIRS], [y for _, y in PAIRS], p)

        batch.align_full()
        batch.set_params(blosum2)
        total = batch.realign_banded(30)
        assert isinstance(total, float)

        grad = grad_matrix(batch.compute_grad())
        assert grad.shape == (20, 20)
        assert grad.sum() > 0


# ── SeqPairBatch.score_and_grad ───────────────────────────────────────────────

def typed_batch(pairs, p, gap_model="affine", mode="global", grad_mode="hard", n_threads=1):
    b = nwgrad.SeqPairBatch(n_threads=n_threads, gap_model=gap_model, mode=mode,
                            grad_mode=grad_mode)
    b.add_many([x for x, _ in pairs], [y for _, y in pairs], p)
    return b


def standalone(pairs, p, **kw):
    """The same pairs as independent SeqPairs (one-pair batches, own fill)."""
    return [nwgrad.SeqPair(a, b, p, **kw) for a, b in pairs]


class TestScoreAndGrad:
    """score_and_grad uses thread-owned DpBuffers; no path is stored unless asked."""

    def _make_batch(self, grad_mode="hard", n_threads=1):
        batch = typed_batch(PAIRS, make_params(1.0, 11.0), grad_mode=grad_mode,
                            n_threads=n_threads)
        return batch, list(batch)

    # ── score correctness ──────────────────────────────────────────────────────

    @pytest.mark.parametrize("gap_model,mode,gap_open,gap_extend", ALL_CONFIGS)
    def test_score_matches_align_full(self, gap_model, mode, gap_open, gap_extend):
        p = make_params(gap_extend, gap_open)
        ref_total = 0.0
        for sp in standalone(PAIRS, p, gap_model=gap_model, mode=mode, grad_mode="hard"):
            sp.align_full()
            ref_total += sp.score
        new_total = typed_batch(PAIRS, p, gap_model, mode).score_and_grad()
        assert new_total == pytest.approx(ref_total, rel=1e-10)

    def test_banded_score_matches_wide_realign(self, blosum):
        bw = max(len(a) for a, _ in PAIRS) + max(len(b) for _, b in PAIRS) + 5
        p = make_params(1.0, 11.0)
        ref_total = 0.0
        for sp in standalone(PAIRS, p, gap_model="affine", mode="global", grad_mode="hard"):
            sp.align_full()
            sp.realign_banded(bw)
            ref_total += sp.score
        # Two calls, not one: banding around the full DP's OWN optimal path can never
        # change the answer, so score_and_grad() establishes the guide and banded_grad()
        # re-aligns around it.
        batch_new = typed_batch(PAIRS, p)
        batch_new.score_and_grad()
        new_total = batch_new.banded_grad(bw)
        assert new_total == pytest.approx(ref_total, rel=1e-10)

    # ── gradient correctness ───────────────────────────────────────────────────

    @pytest.mark.parametrize("gap_model,mode,gap_open,gap_extend", ALL_CONFIGS)
    def test_grad_sum_matches_ref_api(self, gap_model, mode, gap_open, gap_extend):
        p = make_params(gap_extend, gap_open)
        ref_aligner = StreamAligner(
            params=p, gap_model=gap_model, mode=mode, grad_mode="hard", n_threads=1)
        ref_grad = ref_aligner.align(
            [pr[0] for pr in PAIRS], [pr[1] for pr in PAIRS]).grad.matrix.to_matrix()

        batch = typed_batch(PAIRS, p, gap_model, mode)
        batch.score_and_grad()
        new_grad = grad_matrix(batch.compute_grad())

        np.testing.assert_allclose(new_grad, ref_grad, atol=1e-10)

    def test_banded_grad_matches_wide_realign(self):
        bw = max(len(a) for a, _ in PAIRS) + max(len(b) for _, b in PAIRS) + 5
        p = make_params(1.0, 11.0)

        ref_grad = np.zeros((20, 20))
        for sp in standalone(PAIRS, p, gap_model="affine", mode="global", grad_mode="hard"):
            sp.align_full()
            sp.compute_grad()
            ref_grad += grad_matrix(sp.grad)

        batch_new = typed_batch(PAIRS, p)
        batch_new.score_and_grad()
        batch_new.banded_grad(bw)
        new_grad = grad_matrix(batch_new.compute_grad())

        np.testing.assert_allclose(new_grad, ref_grad, atol=1e-10)

    def test_soft_grad_matches_ref(self):
        p = make_params(1.0, 11.0)
        ref_aligner = StreamAligner(
            params=p, gap_model="affine", mode="global", grad_mode="soft", n_threads=1)
        ref_result = ref_aligner.align(
            [pr[0] for pr in PAIRS], [pr[1] for pr in PAIRS])
        ref_grad = ref_result.grad.matrix.to_matrix()
        ref_total = float(np.array(ref_result.scores).sum())

        batch = typed_batch(PAIRS, p, grad_mode="soft")
        new_total = batch.score_and_grad()
        new_grad = grad_matrix(batch.compute_grad())

        assert new_total == pytest.approx(ref_total, rel=1e-10)
        np.testing.assert_allclose(new_grad, ref_grad, atol=1e-10)

    # ── state flags ───────────────────────────────────────────────────────────

    def test_dp_valid_stays_false(self):
        batch, pairs_obj = self._make_batch()
        batch.score_and_grad()
        assert not any(sp.dp_valid for sp in pairs_obj)

    def test_keep_paths_stores_paths(self):
        batch, pairs_obj = self._make_batch()
        batch.score_and_grad(keep_paths=True)
        assert all(sp.dp_valid for sp in pairs_obj)
        assert all(len(sp.aligned()[0]) == len(sp.aligned()[1]) for sp in pairs_obj)

    def test_score_valid_set(self):
        batch, pairs_obj = self._make_batch()
        batch.score_and_grad()
        assert all(sp.score_valid for sp in pairs_obj)

    def test_path_valid_set(self):
        batch, pairs_obj = self._make_batch()
        batch.score_and_grad()
        assert all(sp.path_valid for sp in pairs_obj)

    def test_grad_valid_set_for_hard_mode(self):
        batch, pairs_obj = self._make_batch(grad_mode="hard")
        batch.score_and_grad()
        assert all(sp.grad_valid for sp in pairs_obj)

    def test_grad_valid_false_for_none_mode(self):
        batch, pairs_obj = self._make_batch(grad_mode="none")
        batch.score_and_grad()
        assert not any(sp.grad_valid for sp in pairs_obj)

    def test_compute_grad_after_score_and_grad_sums_cached(self):
        """compute_grad() after score_and_grad() reuses cached grads (no DP needed)."""
        batch, pairs_obj = self._make_batch()
        batch.score_and_grad()
        grad_sum = grad_matrix(batch.compute_grad())
        expected = sum(grad_matrix(sp.grad) for sp in pairs_obj)
        np.testing.assert_allclose(grad_sum, expected, atol=1e-12)

    # ── threading ─────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("n_threads", [2, 4])
    def test_multithread_score_matches_single(self, n_threads):
        def make(nt):
            return typed_batch(PAIRS * 3, make_params(1.0, 11.0), n_threads=nt)

        assert make(1).score_and_grad() == make(n_threads).score_and_grad()

    @pytest.mark.parametrize("n_threads", [2, 4])
    def test_multithread_grad_matches_single(self, n_threads):
        def make_and_run(nt):
            b = typed_batch(PAIRS * 3, make_params(1.0, 11.0), n_threads=nt)
            b.score_and_grad()
            return grad_matrix(b.compute_grad())

        np.testing.assert_array_equal(make_and_run(n_threads), make_and_run(1))

    # ── guide_j is set from full DP even when banded ──────────────────────────

    def test_guide_j_set_after_score_and_grad(self):
        batch, pairs_obj = self._make_batch()
        batch.score_and_grad()
        for sp in pairs_obj:
            gj = sp.guide_j
            assert gj is not None
            assert len(gj) == len(sp.seq_a) + 1

    def test_guide_j_matches_align_full(self):
        """guide_j from score_and_grad should equal guide_j from align_full."""
        p = make_params(1.0, 11.0)
        sp_ref = nwgrad.SeqPair("PLEASANTLY", "MEANLY", p,
                                gap_model="affine", mode="global", grad_mode="hard")
        batch = typed_batch([("PLEASANTLY", "MEANLY")], p)
        batch.score_and_grad()
        sp_ref.align_full()

        assert list(batch[0].guide_j) == list(sp_ref.guide_j)


# ── drop_dp: SeqPair ─────────────────────────────────────────────────────────

class TestDropDp:
    def test_dp_invalid_initially(self, blosum):
        sp = nwgrad.SeqPair("ACDE", "ACDE", blosum)
        assert not sp.dp_valid

    def test_dp_valid_after_align_full(self, blosum):
        sp = nwgrad.SeqPair("ACDE", "ACDE", blosum)
        sp.align_full()
        assert sp.dp_valid

    def test_dp_valid_after_realign_banded(self, blosum):
        sp = nwgrad.SeqPair("ACDE", "ACDE", blosum)
        sp.align_full()
        sp.realign_banded(10)
        assert sp.dp_valid

    def test_drop_dp_clears_dp_valid(self, blosum):
        sp = nwgrad.SeqPair("ACDE", "ACDE", blosum)
        sp.align_full()
        sp.drop_dp()
        assert not sp.dp_valid

    def test_drop_dp_preserves_score(self):
        p = make_params(1.0, 11.0)
        sp = nwgrad.SeqPair("PLEASANTLY", "MEANLY", p, gap_model="affine", mode="global")
        sp.align_full()
        score_before = sp.score
        sp.drop_dp()
        assert sp.score_valid
        assert sp.score == pytest.approx(score_before)

    def test_drop_dp_preserves_grad(self):
        p = make_params(1.0, 11.0)
        sp = nwgrad.SeqPair("PLEASANTLY", "MEANLY", p,
                             gap_model="affine", mode="global", grad_mode="hard")
        sp.align_full()
        sp.compute_grad()
        grad_before = grad_matrix(sp.grad).copy()
        sp.drop_dp()
        assert sp.grad_valid
        np.testing.assert_array_equal(grad_matrix(sp.grad), grad_before)

    def test_drop_dp_preserves_guide_j(self, blosum):
        sp = nwgrad.SeqPair("PLEASANTLY", "MEANLY", blosum)
        sp.align_full()
        gj_before = list(sp.guide_j)
        sp.drop_dp()
        assert sp.path_valid
        assert list(sp.guide_j) == gj_before

    def test_drop_dp_blocks_compute_grad(self, blosum):
        sp = nwgrad.SeqPair("ACDE", "ACDE", blosum)
        sp.align_full()
        sp.drop_dp()
        with pytest.raises(Exception, match="dropped"):
            sp.compute_grad()

    def test_drop_dp_then_align_full_reenables_compute_grad(self, blosum):
        sp = nwgrad.SeqPair("ACDE", "ACDE", blosum)
        sp.align_full()
        sp.drop_dp()
        sp.align_full()
        assert sp.dp_valid
        sp.compute_grad()
        assert sp.grad_valid

    def test_drop_dp_then_realign_banded_reenables_compute_grad(self, blosum):
        sp = nwgrad.SeqPair("ACDE", "ACDE", blosum)
        sp.align_full()
        sp.drop_dp()
        sp.realign_banded(10)
        assert sp.dp_valid
        sp.compute_grad()
        assert sp.grad_valid

    def test_drop_dp_before_compute_grad_score_still_accessible(self, blosum):
        sp = nwgrad.SeqPair("ACDE", "ACDE", blosum, grad_mode="none")
        sp.align_full()
        sp.drop_dp()
        assert sp.score is not None
        assert sp.guide_j is not None

    def test_drop_dp_idempotent(self, blosum):
        sp = nwgrad.SeqPair("ACDE", "ACDE", blosum)
        sp.align_full()
        sp.drop_dp()
        sp.drop_dp()
        assert not sp.dp_valid

    @pytest.mark.parametrize("gap_model,mode,gap_open,gap_extend", ALL_CONFIGS)
    def test_drop_dp_then_realign_score_matches_fresh(
            self, gap_model, mode, gap_open, gap_extend):
        a, b = "PLEASANTLY", "MEANLY"
        p = make_params(gap_extend, gap_open)
        sp = nwgrad.SeqPair(a, b, p, gap_model=gap_model, mode=mode, grad_mode="hard")
        sp.align_full()
        sp.drop_dp()
        bw = max(len(a), len(b)) + 5
        sp.realign_banded(bw)
        sp.compute_grad()

        sp2 = nwgrad.SeqPair(a, b, p, gap_model=gap_model, mode=mode, grad_mode="hard")
        sp2.align_full()
        sp2.compute_grad()

        assert sp.score == pytest.approx(sp2.score, rel=1e-10)
        np.testing.assert_allclose(grad_matrix(sp.grad), grad_matrix(sp2.grad), atol=1e-12)


# ── drop_dp: SeqPairBatch ─────────────────────────────────────────────────────

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestDropDpBatch:
    """The deprecated pair-table batch API (alloc_dp / align_full / realign_banded /
    drop_dp), kept for DiscrimAlign: same observable lifecycle, over stored paths."""
    def _make_batch(self, n_threads=1):
        p = make_params(1.0, 11.0)
        batch = nwgrad.SeqPairBatch(n_threads=n_threads, gap_model="affine", mode="global",
                                  grad_mode="hard")
        batch.add_many([x for x, _ in PAIRS], [y for _, y in PAIRS], p)
        pairs_obj = list(batch)
        return batch, pairs_obj

    def test_batch_drop_dp_clears_dp_valid_on_all(self):
        batch, pairs_obj = self._make_batch()
        batch.align_full()
        assert all(sp.dp_valid for sp in pairs_obj)
        batch.drop_dp()
        assert not any(sp.dp_valid for sp in pairs_obj)

    def test_batch_drop_dp_preserves_scores(self):
        batch, pairs_obj = self._make_batch()
        batch.align_full()
        scores_before = [sp.score for sp in pairs_obj]
        batch.drop_dp()
        for sp, s in zip(pairs_obj, scores_before):
            assert sp.score == pytest.approx(s)

    def test_batch_drop_dp_preserves_grads(self):
        batch, pairs_obj = self._make_batch()
        batch.align_full()
        batch.compute_grad()
        grads_before = [grad_matrix(sp.grad).copy() for sp in pairs_obj]
        batch.drop_dp()
        for sp, g in zip(pairs_obj, grads_before):
            np.testing.assert_array_equal(grad_matrix(sp.grad), g)

    def test_batch_drop_dp_blocks_compute_grad(self):
        batch, pairs_obj = self._make_batch()
        batch.align_full()
        batch.drop_dp()
        with pytest.raises(Exception, match="dropped"):
            batch.compute_grad()

    def test_batch_drop_dp_then_realign_restores(self):
        batch, pairs_obj = self._make_batch()
        batch.align_full()
        batch.drop_dp()
        batch.realign_banded(30)
        assert all(sp.dp_valid for sp in pairs_obj)
        grad = grad_matrix(batch.compute_grad())
        assert grad.shape == (20, 20)
        assert grad.sum() > 0

    def test_batch_drop_dp_multithreaded(self):
        batch, pairs_obj = self._make_batch(n_threads=4)
        batch.align_full()
        batch.drop_dp()
        assert not any(sp.dp_valid for sp in pairs_obj)


# ── SeqPairBatch: API surface ─────────────────────────────────────────────────

class TestSeqPairBatchAPI:
    def test_len(self, blosum):
        batch = nwgrad.SeqPairBatch(grad_mode="hard")
        assert len(batch) == 0
        batch.add_many([a for a, _ in PAIRS], [b for _, b in PAIRS], blosum)
        assert len(batch) == len(PAIRS)

    def test_getitem(self, blosum):
        batch = typed_batch([("ACDE", "ACDE")], blosum)
        assert batch[0].seq_a == "ACDE"

    def test_getitem_negative(self, blosum):
        batch = typed_batch(PAIRS, blosum)
        assert batch[-1].seq_a == PAIRS[-1][0]

    def test_getitem_out_of_range(self):
        batch = nwgrad.SeqPairBatch(grad_mode="hard")
        with pytest.raises(IndexError):
            _ = batch[0]

    def test_type_is_the_batchs(self, blosum):
        batch = typed_batch(PAIRS, blosum, gap_model="linear", mode="local", grad_mode="soft")
        assert (batch.gap_model, batch.mode, batch.grad_mode) == ("linear", "local", "soft")
        assert (batch[0].gap_model, batch[0].mode, batch[0].grad_mode) == ("linear", "local", "soft")

    def test_n_threads_default_positive(self):
        batch = nwgrad.SeqPairBatch(grad_mode="hard")
        assert batch.n_threads >= 1

    def test_n_threads_explicit(self):
        batch = nwgrad.SeqPairBatch(n_threads=3, grad_mode="hard")
        assert batch.n_threads == 3

    def test_grad_shape(self, blosum):
        batch = typed_batch([("ACDE", "ACDE")], blosum)
        batch.score_and_grad()
        g = grad_matrix(batch.compute_grad())
        assert g.shape == (20, 20)

    def test_empty_batch_score_and_grad_returns_zero(self):
        batch = nwgrad.SeqPairBatch(n_threads=1, grad_mode="hard")
        assert batch.score_and_grad(keep_paths=True) == pytest.approx(0.0)

    def test_empty_batch_banded_grad_returns_zero(self):
        batch = nwgrad.SeqPairBatch(n_threads=1, grad_mode="hard")
        assert batch.banded_grad(10) == pytest.approx(0.0)

    def test_empty_batch_compute_grad_raises(self):
        # The sum of no gradients has no alphabet, and there is nothing in scope
        # to infer one from.  This used to return a zero gradient shaped (20, 20)
        # -- the protein default -- so an empty DNA batch silently handed back a
        # protein-shaped answer.
        batch = nwgrad.SeqPairBatch(n_threads=1)
        with pytest.raises(RuntimeError, match="empty batch"):
            batch.compute_grad()
