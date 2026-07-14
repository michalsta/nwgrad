"""Tests for SeqPairBatch.add_many() — bulk C++-side construction of SeqPairs.

add_many() exists purely for speed: building N SeqPairs one at a time from
Python costs more than the DP they feed.  So the first thing it must earn is the
right to be trusted, and every test here is really the same test — that a batch
built in bulk is *indistinguishable* from one built by hand.

Coverage:
  - add_many() == add() for per-pair score, per-pair grad, and summed grad,
    across every gap_model x mode x grad_mode combination
  - per-pair gradients survive: batch[i].grad is the whole point of the API
  - len / indexing / negative indexing / out-of-range
  - the pairs are usable through the full SeqPair lifecycle (alloc_dp,
    align_full, compute_grad, realign_banded, aligned, drop_dp)
  - mixing add() and add_many() in one batch
  - single-thread == multi-thread (construction is parallel)
  - params lifetime: the batch pins params, including across two add_many()
    calls with *different* params objects
  - failure modes leave the batch UNCHANGED: length mismatch, out-of-alphabet
    character, alphabet mismatch with the existing batch
  - empty add_many() is a no-op
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


def build_by_hand(seqs_a, seqs_b, params, gm, md, gd, n_threads=1):
    """The batch add_many() must be indistinguishable from."""
    batch = nwgrad.SeqPairBatch(n_threads=n_threads)
    pairs = [nwgrad.SeqPair(a, b, params, gap_model=gm, mode=md, grad_mode=gd)
             for a, b in zip(seqs_a, seqs_b)]
    for sp in pairs:
        batch.add(sp)
    # The caller must keep `pairs` alive: batch.add() borrows.
    return batch, pairs


def build_bulk(seqs_a, seqs_b, params, gm, md, gd, n_threads=1):
    batch = nwgrad.SeqPairBatch(n_threads=n_threads)
    batch.add_many(seqs_a, seqs_b, params, gap_model=gm, mode=md, grad_mode=gd)
    return batch


def grad_arrays(g):
    d = g.to_dict()
    return (d["matrix"], d["gap_open_a"], d["gap_extend_a"],
            d["gap_open_b"], d["gap_extend_b"])


def assert_grads_equal(g1, g2):
    m1, *s1 = grad_arrays(g1)
    m2, *s2 = grad_arrays(g2)
    np.testing.assert_allclose(m1, m2, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(s1, s2, rtol=1e-12, atol=1e-12)


# ── equivalence with the hand-built batch ─────────────────────────────────────

@pytest.mark.parametrize("gm,md,gd", ALL_MODES)
def test_add_many_matches_add(gm, md, gd):
    """The load-bearing test: same scores, same per-pair grads, same sum."""
    params = prot_params()
    ref_batch, _keep = build_by_hand(PROT_A, PROT_B, params, gm, md, gd)
    bulk_batch = build_bulk(PROT_A, PROT_B, params, gm, md, gd)

    ref_sum = ref_batch.score_and_grad()
    bulk_sum = bulk_batch.score_and_grad()
    assert bulk_sum == pytest.approx(ref_sum, rel=1e-12)

    assert len(bulk_batch) == len(ref_batch) == len(PROT_A)
    for i in range(len(PROT_A)):
        assert bulk_batch[i].score == pytest.approx(ref_batch[i].score, rel=1e-12)
        assert_grads_equal(bulk_batch[i].grad, ref_batch[i].grad)

    assert_grads_equal(bulk_batch.compute_grad(), ref_batch.compute_grad())


def test_per_pair_grads_are_distinct():
    """Guard against every pair being handed the same gradient object."""
    params = prot_params()
    batch = build_bulk(PROT_A, PROT_B, params, "affine", "local", "hard")
    batch.score_and_grad()
    mats = [batch[i].grad.to_dict()["matrix"] for i in range(len(PROT_A))]
    # These pairs are different enough that not all gradients can coincide.
    assert any(not np.allclose(mats[0], m) for m in mats[1:])


def test_add_many_dna():
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

    batch.alloc_dp()
    total = batch.align_full()
    assert total == pytest.approx(sum(batch[i].score for i in range(len(batch))))

    for i in range(len(batch)):
        assert batch[i].path_valid and batch[i].score_valid and batch[i].dp_valid
        a_al, b_al = batch[i].aligned()
        assert len(a_al) == len(b_al)

    g = batch.compute_grad()
    assert g.to_dict()["matrix"].shape == (20, 20)

    batch.realign_banded(8)
    batch.drop_dp()
    for i in range(len(batch)):
        assert not batch[i].dp_valid


def test_set_params_after_add_many():
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


# ── mixing, threading ─────────────────────────────────────────────────────────

def test_mixed_add_and_add_many():
    params = prot_params()
    batch = nwgrad.SeqPairBatch(n_threads=2)
    hand = nwgrad.SeqPair(PROT_A[0], PROT_B[0], params,
                          gap_model="affine", mode="local", grad_mode="hard")
    batch.add(hand)
    batch.add_many(PROT_A[1:], PROT_B[1:], params,
                   gap_model="affine", mode="local", grad_mode="hard")
    assert len(batch) == len(PROT_A)

    ref, _keep = build_by_hand(PROT_A, PROT_B, params, "affine", "local", "hard")
    assert batch.score_and_grad() == pytest.approx(ref.score_and_grad())
    for i in range(len(PROT_A)):
        assert batch[i].score == pytest.approx(ref[i].score)


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

def test_dropping_params_after_two_add_many_calls():
    """Sanitizer bait.  Its job is to *commit the crime*, not to detect it.

    On a normal build this proves almost nothing: if the batch failed to pin
    params, the first half's pairs would read freed memory, which is undefined
    behaviour rather than a crash -- freed heap usually still holds its old bytes,
    the scores come out right, and the test goes green over a live
    use-after-free.  The refcount tests above are the real guarantee.

    What this buys is the *access*: it must be two add_many() calls with
    different params (one call is pinned even by a single `_params` slot, so it
    frees nothing and there is nothing to trap).  Verified: against a
    single-slot binding under the CI ASan job this raises
    "heap-use-after-free ... READ of size 8 ... thread T3" from a worker thread.
    A sanitizer finds no bug it is not walked into; this walks it in.
    """
    batch = nwgrad.SeqPairBatch(n_threads=2)
    p1 = prot_params(gap_extend=1.0)
    p2 = prot_params(gap_extend=9.0)
    batch.add_many(PROT_A, PROT_B, p1, gap_model="affine", mode="local", grad_mode="hard")
    batch.add_many(PROT_A, PROT_B, p2, gap_model="affine", mode="local", grad_mode="hard")
    del p1, p2
    gc.collect()

    batch.score_and_grad()      # workers read both params objects here
    assert batch[0].score_valid
    assert batch[len(PROT_A)].score_valid


def test_add_many_holds_a_reference_to_params():
    """The pinning contract, asserted directly.

    The pairs are C++-owned and hold a *bare pointer* to params; none of them
    holds a Python reference of its own.  So the batch must hold one for them.

    This is checked on the refcount rather than by aligning after `del params`
    and seeing whether the numbers look right: reading freed memory is undefined
    behaviour, not a crash, and freed heap usually still holds its old bytes.
    Such a test passes whether or not the reference is held, which makes it
    worthless as a regression test.  The refcount either goes up or it does not.
    """
    params = prot_params()
    before = sys.getrefcount(params)

    batch = nwgrad.SeqPairBatch(n_threads=2)
    batch.add_many(PROT_A, PROT_B, params,
                   gap_model="affine", mode="local", grad_mode="hard")

    assert sys.getrefcount(params) > before      # the batch took a reference
    assert any(p is params for p in batch._owned_params)


def test_two_add_many_calls_pin_both_params():
    """A second add_many() must not evict the first call's params.

    Pinning into a single `_params` slot (as set_params() does) would overwrite
    it, drop the last reference to p1, and leave the first call's pairs pointing
    at a freed object.  Again asserted on references, not on whether the wrong
    answer happens to look right.
    """
    p1 = prot_params(gap_extend=1.0)
    p2 = prot_params(gap_extend=9.0)
    p1_before, p2_before = sys.getrefcount(p1), sys.getrefcount(p2)

    batch = nwgrad.SeqPairBatch(n_threads=2)
    batch.add_many(PROT_A, PROT_B, p1, gap_model="affine", mode="local", grad_mode="hard")
    batch.add_many(PROT_A, PROT_B, p2, gap_model="affine", mode="local", grad_mode="hard")

    # BOTH are still referenced -- the second call did not evict the first.
    assert sys.getrefcount(p1) > p1_before
    assert sys.getrefcount(p2) > p2_before
    pinned = batch._owned_params
    assert any(p is p1 for p in pinned)
    assert any(p is p2 for p in pinned)

    # And the values are intact and distinct, so the halves really did use
    # different params -- guarding against both being pinned but one ignored.
    assert len(batch) == 2 * len(PROT_A)
    ref1, _k1 = build_by_hand(PROT_A, PROT_B, p1, "affine", "local", "hard")
    ref2, _k2 = build_by_hand(PROT_A, PROT_B, p2, "affine", "local", "hard")
    ref1.score_and_grad()
    ref2.score_and_grad()
    batch.score_and_grad()
    n = len(PROT_A)
    for i in range(n):
        assert batch[i].score == pytest.approx(ref1[i].score)
        assert batch[n + i].score == pytest.approx(ref2[i].score)
    assert [batch[i].score for i in range(n)] != \
           pytest.approx([batch[n + i].score for i in range(n)])


# ── failure modes leave the batch unchanged ───────────────────────────────────

def test_length_mismatch_raises():
    batch = nwgrad.SeqPairBatch(n_threads=2)
    with pytest.raises(Exception, match="equal length"):
        batch.add_many(PROT_A, PROT_B[:-1], prot_params())
    assert len(batch) == 0


def test_bad_character_raises_and_leaves_batch_unchanged():
    """The throw happens on a worker thread, mid-construction.  The batch must
    not be left half-filled behind it."""
    params = dna_params()
    batch = nwgrad.SeqPairBatch(n_threads=4)
    batch.add_many(DNA_A, DNA_B, params, gap_model="affine", mode="local",
                   grad_mode="hard")
    assert len(batch) == len(DNA_A)

    bad_a = list(DNA_A) + ["ACGT"] * 50 + ["ACGZ"]     # 'Z' is not in ACGT
    bad_b = list(DNA_B) + ["ACGT"] * 50 + ["ACGT"]
    with pytest.raises(Exception, match="not in alphabet"):
        batch.add_many(bad_a, bad_b, params, gap_model="affine", mode="local",
                       grad_mode="hard")

    assert len(batch) == len(DNA_A)          # nothing from the failed call stuck
    batch.score_and_grad()                   # and the survivors still work
    assert batch[0].score_valid


def test_alphabet_mismatch_raises_and_leaves_batch_unchanged():
    batch = nwgrad.SeqPairBatch(n_threads=2)
    batch.add_many(PROT_A, PROT_B, prot_params(),
                   gap_model="affine", mode="local", grad_mode="hard")
    with pytest.raises(Exception, match="cannot join a batch over alphabet"):
        batch.add_many(DNA_A, DNA_B, dna_params(),
                       gap_model="affine", mode="local", grad_mode="hard")
    assert len(batch) == len(PROT_A)
    batch.score_and_grad()


def test_empty_add_many_is_a_noop():
    batch = nwgrad.SeqPairBatch(n_threads=2)
    batch.add_many([], [], prot_params())
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
