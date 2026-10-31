"""Tests for building a SeqPairBatch — bulk C++-side construction of its pairs,
SeqPairBatch(seqs_a, seqs_b, params, ...).

Bulk construction exists purely for speed: building N SeqPairs one at a time from
Python costs more than the DP they feed.  So the first thing it must earn is the
right to be trusted, and every test here is really the same test — that a batch
built in bulk is *indistinguishable* from one built by hand.

Coverage:
  - the batch == the pairs aligned one by one (standalone SeqPairs) for per-pair
    score, per-pair grad, and summed grad,
    across every gap_model x mode x grad_mode combination
  - per-pair gradients survive: batch[i].grad is the whole point of the API
  - len / indexing / negative indexing / out-of-range
  - the pairs are usable through the full SeqPair lifecycle (
    score_and_grad, compute_grad, banded_grad, aligned, drop_paths)
  - single-thread == multi-thread (construction is parallel)
  - params lifetime: the batch owns a copy of its params
  - length mismatch and a bad character raise; an empty batch
"""

import gc
import sys

import numpy as np
import pytest
import nwgrad

from test_subst_matrix import BLOSUM62


# ── helpers ───────────────────────────────────────────────────────────────────

PROT_A = ["ACGWYT", "MKLPQR", "WWYYAA", "ACDEFG", "K", "HHHHPPPP"]
PROT_B = ["ACWYT", "MKLPQ", "WYYAAC", "ACDEF", "KKK", "HHPPP"]

DNA_A = ["ACGTACGT", "TTGCA", "GGGG", "ACACACAC"]
DNA_B = ["ACGTTCGT", "TTGGA", "GGAGG", "ACACTCAC"]

ALL_MODES = [
    (gm, md, gd)
    for gm in ("linear", "affine")
    for md in ("global", "local")
    for gd in ("hard", "soft")
]


def prot_params(gap_open=2.0, gap_extend=1.0):
    return nwgrad.AlignParams(BLOSUM62,
                              gap_open_a=gap_open, gap_extend_a=gap_extend,
                              gap_open_b=gap_open, gap_extend_b=gap_extend)


def dna_params(gap_open=3.0, gap_extend=1.0):
    mat = nwgrad.SubstMatrix(np.eye(4) * 2.0 - 1.0, alphabet="ACGT")
    return nwgrad.AlignParams(mat,
                              gap_open_a=gap_open, gap_extend_a=gap_extend,
                              gap_open_b=gap_open, gap_extend_b=gap_extend)


class HandBatch:
    """The pairs aligned one by one as standalone SeqPairs — what the batch must be
    indistinguishable from (a batch used to be built from such pairs with add())."""

    def __init__(self, pairs, gd):
        self.pairs, self.gd = pairs, gd

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        return self.pairs[i]

    def score_and_grad(self):
        total = 0.0
        for sp in self.pairs:
            sp.align()
            total += sp.score
        return total

    def compute_grad(self):
        g = self.pairs[0].grad
        for sp in self.pairs[1:]:
            g = g + sp.grad
        return g


def build_by_hand(seqs_a, seqs_b, params, gm, md, gd, n_threads=1):
    pairs = [nwgrad.SeqPair(a, b, params, gap_model=gm, mode=md, grad_mode=gd)
             for a, b in zip(seqs_a, seqs_b)]
    return HandBatch(pairs, gd), pairs


def build_bulk(seqs_a, seqs_b, params, gm, md, gd, n_threads=1):
    batch = nwgrad.SeqPairBatch(seqs_a, seqs_b, params,
                                n_threads=n_threads, gap_model=gm, mode=md, grad_mode=gd)
    return batch


def grad_arrays(g):
    d = g.to_dict()
    return (d["matrix"], d["gap_open_a"], d["gap_extend_a"],
            d["gap_open_b"], d["gap_extend_b"])


def assert_grads_equal(g1, g2, tol=1e-12):
    m1, *s1 = grad_arrays(g1)
    m2, *s2 = grad_arrays(g2)
    np.testing.assert_allclose(m1, m2, rtol=tol, atol=tol)
    np.testing.assert_allclose(s1, s2, rtol=tol, atol=tol)


# ── equivalence with the hand-built batch ─────────────────────────────────────

@pytest.mark.parametrize("gm,md,gd", ALL_MODES)
def test_batch_matches_pairs(gm, md, gd):
    """The load-bearing test: same scores, same per-pair grads, same sum."""
    params = prot_params()
    ref_batch, _keep = build_by_hand(PROT_A, PROT_B, params, gm, md, gd)
    bulk_batch = build_bulk(PROT_A, PROT_B, params, gm, md, gd)

    ref_sum = ref_batch.score_and_grad()
    bulk_sum = bulk_batch.score_and_grad()
    # Soft: the two batches may fill differently (shared / own); a float32 batch's soft
    # pass runs in float32, so its results agree to ~1e-6 relative, not bit for bit.
    tol = 2e-4 if gd == "soft" else 1e-12
    assert bulk_sum == pytest.approx(ref_sum, rel=tol)

    assert len(bulk_batch) == len(ref_batch) == len(PROT_A)
    for i in range(len(PROT_A)):
        assert bulk_batch[i].score == pytest.approx(ref_batch[i].score, rel=tol)
        assert_grads_equal(bulk_batch[i].grad, ref_batch[i].grad, tol)

    assert_grads_equal(bulk_batch.compute_grad(), ref_batch.compute_grad(), tol)


def test_per_pair_grads_are_distinct():
    """Guard against every pair being handed the same gradient object."""
    params = prot_params()
    batch = build_bulk(PROT_A, PROT_B, params, "affine", "local", "hard")
    batch.score_and_grad()
    mats = [batch[i].grad.to_dict()["matrix"] for i in range(len(PROT_A))]
    # These pairs are different enough that not all gradients can coincide.
    assert any(not np.allclose(mats[0], m) for m in mats[1:])


def test_batch_dna():
    params = dna_params()
    ref_batch, _keep = build_by_hand(DNA_A, DNA_B, params, "affine", "local", "hard")
    bulk_batch = build_bulk(DNA_A, DNA_B, params, "affine", "local", "hard")
    assert bulk_batch.score_and_grad() == pytest.approx(ref_batch.score_and_grad())
    assert_grads_equal(bulk_batch.compute_grad(), ref_batch.compute_grad())


def test_seq_roundtrip():
    params = dna_params()
    batch = build_bulk(DNA_A, DNA_B, params, "affine", "local", "hard")
    for i, (a, b) in enumerate(zip(DNA_A, DNA_B)):
        assert batch[i].seq_a == a
        assert batch[i].seq_b == b


# ── the owned pairs are ordinary SeqPairs ─────────────────────────────────────

def test_full_seqpair_lifecycle_on_owned_pairs():
    params = prot_params()
    batch = build_bulk(PROT_A, PROT_B, params, "affine", "local", "hard")

    total = batch.score_and_grad(keep_paths=True)
    assert total == pytest.approx(sum(batch[i].score for i in range(len(batch))))

    for i in range(len(batch)):
        assert batch[i].path_valid and batch[i].score_valid and batch[i].stored_path_valid
        a_al, b_al = batch[i].aligned()
        assert len(a_al) == len(b_al)

    g = batch.compute_grad()
    assert g.to_dict()["matrix"].shape == (20, 20)

    batch.banded_grad(8, keep_paths=True)
    batch.drop_paths()
    for i in range(len(batch)):
        assert not batch[i].stored_path_valid


def test_set_params_after_construction():
    """A gradient-descent step: swap params, re-run, get a different answer."""
    batch = build_bulk(PROT_A, PROT_B, prot_params(gap_extend=1.0),
                       "affine", "local", "hard")
    s1 = batch.score_and_grad()
    batch.set_params(prot_params(gap_extend=20.0))
    s2 = batch.score_and_grad()
    assert s1 != pytest.approx(s2)


def test_indexing():
    params = dna_params()
    batch = build_bulk(DNA_A, DNA_B, params, "affine", "local", "hard")
    batch.score_and_grad()
    assert batch[-1].seq_a == DNA_A[-1]
    assert batch[len(DNA_A) - 1].score == batch[-1].score
    with pytest.raises(IndexError):
        batch[len(DNA_A)]
    with pytest.raises(IndexError):
        batch[-len(DNA_A) - 1]


# ── threading ─────────────────────────────────────────────────────────────────

def test_threads_do_not_change_the_answer():
    """Construction is parallel; order and content must not depend on that."""
    params = prot_params()
    seqs_a = PROT_A * 40
    seqs_b = PROT_B * 40
    b1 = build_bulk(seqs_a, seqs_b, params, "affine", "local", "hard", n_threads=1)
    b8 = build_bulk(seqs_a, seqs_b, params, "affine", "local", "hard", n_threads=8)
    assert b1.score_and_grad() == pytest.approx(b8.score_and_grad(), rel=1e-12)
    for i in range(len(seqs_a)):
        assert b1[i].seq_a == b8[i].seq_a      # order preserved
        assert b1[i].score == pytest.approx(b8[i].score)
        assert_grads_equal(b1[i].grad, b8[i].grad)


# ── params lifetime ───────────────────────────────────────────────────────────

def test_dropping_params_after_set_params():
    """Sanitizer bait.  Its job is to *commit the crime*, not to detect it.

    On a normal build this proves almost nothing: if the batch read the caller's
    params instead of its own copy, the workers would read freed memory, which is
    undefined behaviour rather than a crash -- freed heap usually still holds its
    old bytes, the scores come out right, and the test goes green over a live
    use-after-free.  test_batch_copies_params is the real guarantee; under the CI
    ASan job this walks the workers into the access.
    """
    p1 = prot_params(gap_extend=1.0)
    p2 = prot_params(gap_extend=9.0)
    batch = nwgrad.SeqPairBatch(PROT_A, PROT_B, p1,
                                n_threads=2, gap_model="affine", mode="local", grad_mode="hard")
    batch.set_params(p2)
    del p1, p2
    gc.collect()

    batch.score_and_grad()      # workers read the params here
    assert batch[0].score_valid
    ref, _keep = build_by_hand(PROT_A, PROT_B, prot_params(gap_extend=9.0), "affine", "local", "hard")
    assert batch.score_and_grad() == pytest.approx(ref.score_and_grad())


def test_batch_copies_params():
    """The batch owns a COPY of its params: it holds no reference to the caller's
    object, and an in-place change to that object does not reach the batch (the
    scan's B3: cached results stayed "valid" under params that had changed).
    set_params() is how new params reach it."""
    params = prot_params()
    before = sys.getrefcount(params)
    batch = nwgrad.SeqPairBatch(PROT_A, PROT_B, params,
                                n_threads=2, gap_model="affine", mode="local")
    assert sys.getrefcount(params) == before
    first = batch.score_and_grad()

    params.gap_open_a = 50.0              # in place: the batch must not see it
    params *= 3.0
    assert batch[0].score_valid
    assert batch.score_and_grad() == first

    batch.set_params(params)              # ... until it is handed over
    assert not batch[0].score_valid
    assert batch.score_and_grad() != first


def test_standalone_pair_copies_params():
    params = prot_params()
    sp = nwgrad.SeqPairDouble(PROT_A[0], PROT_B[0], params, traceback="scores")
    s0 = sp.score_and_grad()[0]
    params *= 3.0
    assert sp.score_valid and sp.score == s0
    assert nwgrad.SeqPairDouble(PROT_A[0], PROT_B[0], params).score_and_grad()[0] != s0


# ── failure modes ─────────────────────────────────────────────────────────────

def test_length_mismatch_raises():
    with pytest.raises(Exception, match="equal length"):
        nwgrad.SeqPairBatch(PROT_A, PROT_B[:-1], prot_params(), n_threads=2)


def test_bad_character_raises_from_a_worker():
    """The throw happens on a worker thread, mid-construction: it must surface as an
    exception naming the character, with nothing leaked behind it (ASan)."""
    params = dna_params()
    bad_a = list(DNA_A) + ["ACGT"] * 50 + ["ACGZ"]     # 'Z' is not in ACGT
    bad_b = list(DNA_B) + ["ACGT"] * 50 + ["ACGT"]
    with pytest.raises(Exception, match="not in alphabet"):
        nwgrad.SeqPairBatch(bad_a, bad_b, params, n_threads=4, gap_model="affine", mode="local")


def test_empty_batch():
    batch = nwgrad.SeqPairBatch([], [], prot_params(), n_threads=2)
    assert len(batch) == 0
    # An empty batch still has no alphabet to sum a gradient over.
    with pytest.raises(Exception):
        batch.compute_grad()


def test_grad_mode_none_still_scores():
    params = prot_params()
    batch = build_bulk(PROT_A, PROT_B, params, "affine", "local", "none")
    batch.score_and_grad()
    assert batch[0].score_valid
    assert not batch[0].grad_valid
    # The binding reports an uncomputed gradient as None rather than raising.
    assert batch[0].grad is None
