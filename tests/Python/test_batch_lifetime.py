"""Object lifetimes across SeqPairBatch: pairs that outlive their batch, params that
outlive their Python name.

The C++ side holds raw pointers everywhere — a SeqPair points at its AlignParams, a
batch points at its pairs — and Python keep-alives are what make that safe.  Each test
here builds a situation where one of those keep-alives is missing, deletes the Python
name that was (accidentally) keeping the memory alive, churns the allocator so freed
storage is reused, and then uses the survivor.  The correct answer is computed from a
freshly built reference pair over the same values.

Every scenario runs in a child interpreter (conftest.run_isolated): the failure mode is
a use-after-free, which in-process would crash pytest or corrupt its heap and fail some
unrelated later test.  Observed on the unfixed tree: a segfault (exit 139) for an
indexed pair, and a silently wrong score (396 where 8 is correct) for stale params.

Tests marked xfail(strict=True) document a bug not yet fixed.  When the fix lands they
XPASS, strict turns that into a failure, and the marker must be removed — so the test
cannot quietly keep "expecting" a bug that no longer exists.
"""

import pytest

from conftest import describe, run_isolated


PRECISIONS = [("SeqPair", "SeqPairBatch"), ("SeqPairDouble", "SeqPairBatchDouble")]
PRECISION_IDS = ["float32", "double"]

# Shared by every child.  `expect` compares a survivor against a fresh pair built over
# params with the same VALUES, so a pair reading freed or reused storage cannot match.
PRELUDE = """
import gc
import numpy as np
import nwgrad as n

SP, SPB = n.{sp}, n.{spb}
A, B = "ACGTTGCA", "ACGATGCA"

def params(k):
    return n.AlignParams(k * np.eye(4) - np.ones((4, 4)), alphabet="ACGT",
                         gap_open_a=2.0, gap_extend_a=1.0,
                         gap_open_b=2.0, gap_extend_b=1.0)

def churn():
    # Reuse freed storage: same-sized AlignParams and assorted small blocks.
    gc.collect()
    return ([params(99) for _ in range(200)] +
            [np.full(n_, 7.0) for n_ in (4, 16, 64, 256) for _ in range(500)])

def expect(sp, k, a=A, b=B):
    s_ref, g_ref = SP(a, b, params(k)).score_and_grad()
    s, g = sp.score_and_grad()
    assert s == s_ref, f"score {{s}} != {{s_ref}}"
    assert np.array_equal(g.matrix.to_matrix(), g_ref.matrix.to_matrix()), "matrix grad differs"
    for f in ("gap_open_a", "gap_extend_a", "gap_open_b", "gap_extend_b"):
        assert getattr(g, f) == getattr(g_ref, f), f"{{f}} grad differs"
"""


def run(sp, spb, body):
    proc = run_isolated(PRELUDE.format(sp=sp, spb=spb) + body + "\nprint('OK')\n")
    assert proc.returncode == 0 and "OK" in proc.stdout, describe(proc)


# --- Issue 2: a pair indexed out of a batch must keep the batch alive -------------------
#
# add_many() pairs are owned by the batch (unique_ptrs in owned_); __getitem__ hands them
# out with rv_policy::reference, so nothing stops the batch dying under the wrapper.

ISSUE2 = pytest.mark.xfail(strict=True, reason="issue 2: indexed C++-owned pair does not "
                                               "retain its batch (rv_policy::reference)")


@ISSUE2
@pytest.mark.parametrize("sp,spb", PRECISIONS, ids=PRECISION_IDS)
def test_indexed_pair_outlives_batch(sp, spb):
    run(sp, spb, """
p = params(2)
b = SPB(n_threads=1)
b.add_many([A], [B], p)
s = b[0]
del b, p
trash = churn()
expect(s, 2)
""")


@ISSUE2
@pytest.mark.parametrize("sp,spb", PRECISIONS, ids=PRECISION_IDS)
def test_every_indexed_pair_outlives_batch(sp, spb):
    """Index every pair, several times over (exercises nanobind handing back an existing
    wrapper as well as creating a new one), then drop the batch."""
    run(sp, spb, """
seqs = [(A, B), ("ACGT", "AGT"), ("TTTT", "TTAT"), ("GATTACA", "GATACA")]
b = SPB(n_threads=2)
b.add_many([x for x, _ in seqs], [y for _, y in seqs], params(2))
held = [b[i] for _ in range(3) for i in range(len(seqs))] + [b[-1]]
del b
trash = churn()
for k, s in enumerate(held[:len(seqs)]):
    expect(s, 2, *seqs[k])
expect(held[-1], 2, *seqs[-1])
""")


@ISSUE2
@pytest.mark.parametrize("sp,spb", PRECISIONS, ids=PRECISION_IDS)
def test_owned_pair_borrowed_by_second_batch(sp, spb):
    """A pair owned by batch 1, added to batch 2: batch 2's keep-alive holds the wrapper,
    so the wrapper must be what keeps batch 1 (the real owner) alive."""
    run(sp, spb, """
p = params(2)
b1 = SPB(n_threads=1)
b1.add_many([A], [B], p)
b2 = SPB(n_threads=1)
b2.add(b1[0])
del b1
trash = churn()
total = b2.score_and_grad()
ref = SP(A, B, params(2)).score_and_grad()[0]
assert total == ref, f"batch total {total} != {ref}"
""")


@pytest.mark.parametrize("sp,spb", PRECISIONS, ids=PRECISION_IDS)
def test_python_pair_outlives_batch(sp, spb):
    """The already-safe half of the model, pinned so a fix for issue 2 cannot break it:
    a Python-constructed pair added to a batch, then the batch dropped."""
    run(sp, spb, """
p = params(2)
s = SP(A, B, p)
b = SPB(n_threads=1)
b.add(s)
b.score_and_grad()
del b
trash = churn()
expect(s, 2)
""")


# --- Issue 3: batch.set_params() must pin the new params for every surviving pair -------
#
# The batch re-points every pair at the new params but pins them only in the BATCH's
# _params attribute.  A borrowed pair outliving the batch then points at freed params.

ISSUE3 = pytest.mark.xfail(strict=True, reason="issue 3: batch.set_params() does not pin "
                                               "new params for borrowed pairs")


@ISSUE3
@pytest.mark.parametrize("sp,spb", PRECISIONS, ids=PRECISION_IDS)
def test_borrowed_pair_keeps_batch_params(sp, spb):
    run(sp, spb, """
s = SP(A, B, params(1))
b = SPB(n_threads=1)
b.add(s)
q = params(2)
b.set_params(q)
del b, q
trash = churn()
expect(s, 2)
""")


@ISSUE3
@pytest.mark.parametrize("sp,spb", PRECISIONS, ids=PRECISION_IDS)
def test_borrowed_pair_after_repeated_set_params(sp, spb):
    """A training loop: many replacements, each new params' only name dropped at once.
    The pair must end on the LAST params — and the earlier ones need not be retained
    (that is the O(1) replacement the bindings' keep_current_params() exists for)."""
    run(sp, spb, """
s = SP(A, B, params(1))
b = SPB(n_threads=1)
b.add(s)
for k in range(2, 12):
    b.set_params(params(k))
    b.score_and_grad()
del b
trash = churn()
expect(s, 11)
""")


@ISSUE3
@pytest.mark.parametrize("sp,spb", PRECISIONS, ids=PRECISION_IDS)
def test_pair_shared_by_two_batches(sp, spb):
    """Last set_params() wins, whichever batch made it, and the pair outlives both."""
    run(sp, spb, """
s = SP(A, B, params(1))
b1, b2 = SPB(n_threads=1), SPB(n_threads=1)
b1.add(s); b2.add(s)
b1.set_params(params(2))
b2.set_params(params(3))
del b2
trash = churn()
expect(s, 3)
del b1
trash = churn()
expect(s, 3)
""")


@pytest.mark.parametrize("sp,spb", PRECISIONS, ids=PRECISION_IDS)
def test_pair_set_params_after_batch_set_params(sp, spb):
    """pair.set_params() after batch.set_params(): the pair's own pin must win.  Passes
    today; pinned so that the issue 3 fix does not leave a stale batch pin in charge."""
    run(sp, spb, """
s = SP(A, B, params(1))
b = SPB(n_threads=1)
b.add(s)
b.set_params(params(2))
s.set_params(params(4))
del b
trash = churn()
expect(s, 4)
""")


@pytest.mark.parametrize("sp,spb", PRECISIONS, ids=PRECISION_IDS)
def test_add_many_pairs_follow_batch_set_params(sp, spb):
    """Batch-owned pairs re-pointed by set_params(), original params dropped.  Passes
    today (the batch's own _params pins them); pinned for the same reason."""
    run(sp, spb, """
b = SPB(n_threads=2)
b.add_many([A, A], [B, B], params(1))
b.set_params(params(3))
trash = churn()
total = b.score_and_grad()
ref = SP(A, B, params(3)).score_and_grad()[0]
assert total == 2 * ref, f"batch total {total} != {2 * ref}"
""")
