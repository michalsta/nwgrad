"""Object lifetimes across SeqPairBatch: pairs that outlive their batch, params that
outlive their Python name.

The C++ side holds raw pointers — a batch points at its params, a pair taken out of a
batch (a view) points at the batch — and Python keep-alives are what make that safe.  Each test
here builds a situation where one of those keep-alives is missing, deletes the Python
name that was (accidentally) keeping the memory alive, churns the allocator so freed
storage is reused, and then uses the survivor.  The correct answer is computed from a
freshly built reference pair over the same values.

Every scenario runs in a child interpreter (conftest.run_isolated): the failure mode is
a use-after-free, which in-process would crash pytest or corrupt its heap and fail some
unrelated later test.  Observed on the unfixed tree: a segfault (exit 139) for an
indexed pair, and a silently wrong score (396 where 8 is correct) for stale params.

The fix: an indexed pair (a view) pins its batch through an `_owner` attribute, and the
batch pins its params (`_owned_params` per add_many(), `_params` for set_params()).
Attributes rather than nanobind keep_alive, so any cycle stays visible to the cyclic GC.
(Since 0.6 there is no add(): the borrowed-pair and shared-pair scenarios that stood
here have no subject any more.)
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
# add_many() pairs are owned by the batch (unique_ptrs in owned_).  __getitem__ used to
# hand them out with rv_policy::reference, so nothing stopped the batch dying under the
# wrapper; the wrapper now pins it through an `_owner` attribute.


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


# --- Issue 3: batch.set_params() must pin the new params for every surviving pair -------
#
# The batch re-points every pair at the new params.  It used to pin them only in the
# BATCH's _params, so a borrowed pair outliving the batch pointed at freed params; it
# now re-pins each borrowed pair's own _params as well.


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


# --- No leaks: the pins above must not keep batches alive forever ------------------------

def _live_batches():
    return """
def live_batches():
    gc.collect()
    return sum(type(o).__name__ in ("SeqPairBatch", "SeqPairBatchDouble")
               for o in gc.get_objects())
"""


@pytest.mark.parametrize("sp,spb", PRECISIONS, ids=PRECISION_IDS)
def test_view_keeps_batch_and_current_params(sp, spb):
    """A view outlives its batch AFTER set_params() with the new params' name dropped:
    view -> batch -> current params must all hold."""
    run(sp, spb, """
b = SPB(n_threads=1, gap_model="affine", mode="global", grad_mode="hard")
b.add_many([A], [B], params(1))
b.set_params(params(3))
v = b[0]
del b
trash = churn()
expect(v, 3)
""")


@pytest.mark.parametrize("sp,spb", PRECISIONS, ids=PRECISION_IDS)
def test_views_do_not_leak_their_batch(sp, spb):
    """The view's pin is an attribute: once view and batch are both gone, the batch is
    collected — no keep-alive table entry outliving them."""
    run(sp, spb, _live_batches() + """
b = SPB(n_threads=1, gap_model="affine", mode="global", grad_mode="hard")
b.add_many([A, A], [B, B], params(2))
views = [b[0], b[1], b[0]]
b.score_and_grad()
del b
trash = churn()
expect(views[0], 2)
del views
assert live_batches() == 0, "views leaked their batch"
""")
