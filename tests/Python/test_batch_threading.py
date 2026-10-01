"""Batch threading hazards that no worker-exception guard can catch.

Issue 4: one SeqPair added to a batch twice.  The workers then aligned the same object
concurrently — two threads writing one pair's DP tables and cached score/path/grad:
wrong totals (some ABOVE the optimum), segfaults and hangs, with no error raised.  The
fix rejects the duplicate (ValueError) at the next dispatch; the tests would equally
accept rejection at add() or serialization — what they reject is a wrong answer.

Issue 5: run_workers_guarded() launches threads outside its guard.  If a launch fails
after earlier ones started, the joinable std::threads are destroyed during unwinding and
the process calls std::terminate.  Forced here with RLIMIT_NPROC (which counts threads).
The fix finishes on the threads that did start; the test also accepts a Python
exception, but never the abort, and never a short sum (the sorted scheduler assigns
chunks to workers statically, so a worker that never started must not lose its chunk).

Scenarios that can race or abort run in a child interpreter (conftest.run_isolated):
a regression would otherwise take pytest down with it.
"""

import os
import sys

import numpy as np
import pytest

import nwgrad
from conftest import describe, run_isolated, running_under_asan


PRECISIONS = [("SeqPair", "SeqPairBatch"), ("SeqPairDouble", "SeqPairBatchDouble")]
PRECISION_IDS = ["float32", "double"]


def dna_params():
    return nwgrad.AlignParams(2 * np.eye(4) - np.ones((4, 4)), alphabet="ACGT",
                              gap_open_a=2.0, gap_extend_a=1.0,
                              gap_open_b=2.0, gap_extend_b=1.0)


# --- Issue 4: duplicate pair pointers ---------------------------------------------------

@pytest.mark.parametrize("sp,spb", PRECISIONS, ids=PRECISION_IDS)
def test_duplicate_pair_rejected_or_correct(sp, spb):
    """One pair added twice among distinct ones, on several threads, both dispatch paths.
    Long enough that two workers genuinely overlap on it."""
    proc = run_isolated(f"""
        import numpy as np, nwgrad as n
        SP, SPB = n.{sp}, n.{spb}
        rng = np.random.default_rng(1)
        rand = lambda k: "".join(rng.choice(list("ACGT"), k))
        p = n.AlignParams(2 * np.eye(4) - np.ones((4, 4)), alphabet="ACGT",
                          gap_open_a=2.0, gap_extend_a=1.0, gap_open_b=2.0, gap_extend_b=1.0)
        A, B = rand(1500), rand(1450)
        others = [(rand(300), rand(290)) for _ in range(4)]
        want = 2 * SP(A, B, p, traceback="pointers").score_and_grad()[0] + sum(
            SP(a, b, p, traceback="pointers").score_and_grad()[0] for a, b in others)

        s = SP(A, B, p, traceback="pointers")
        keep = [SP(a, b, p, traceback="pointers") for a, b in others]
        b = SPB(n_threads=4)
        try:
            b.add(s)
            for o in keep: b.add(o)
            b.add(s)
            b.alloc_dp()
            for it in range(10):
                got = b.align_full(); b.compute_grad()
                assert got == want, f"align_full iter {{it}}: {{got}} != {{want}}"
                got = b.score_and_grad()
                assert got == want, f"score_and_grad iter {{it}}: {{got}} != {{want}}"
        except ValueError as e:
            print("REJECTED", e)
        print("OK")
    """, timeout=300)
    assert proc.returncode == 0 and "OK" in proc.stdout, describe(proc)


def test_duplicate_owned_pair_via_indexing():
    """The other way to duplicate: re-add a pair the batch already owns (from add_many).
    Single-threaded, so in-process is safe; the contract is rejection, since with one
    thread nothing races and a 'serialize' fix would have nothing to prove."""
    b = nwgrad.SeqPairBatch(n_threads=1)
    b.add_many(["ACGT"], ["AGT"], dna_params())
    with pytest.raises(ValueError):
        b.add(b[0])
        b.alloc_dp()
        b.align_full()


def test_duplicate_pair_single_thread_rejected():
    """With one thread a duplicate cannot race, but it is still a caller error (it double
    counts the pair in every sum).  Rejected at add() or at the first dispatch."""
    p = dna_params()
    s = nwgrad.SeqPair("ACGTACGT", "ACGTTCGT", p)
    b = nwgrad.SeqPairBatch(n_threads=1)
    with pytest.raises(ValueError):
        b.add(s)
        b.add(s)
        b.score_and_grad()


@pytest.mark.parametrize("sp,spb", PRECISIONS, ids=PRECISION_IDS)
def test_same_pair_in_two_batches_is_legal(sp, spb):
    """Not a duplicate: one pair in two batches, run one after the other.  A fix for
    issue 4 must not reject this (e.g. by a per-pair 'already in a batch' flag)."""
    SP, SPB = getattr(nwgrad, sp), getattr(nwgrad, spb)
    p = dna_params()
    s = SP("ACGTACGT", "ACGTTCGT", p)
    want = SP("ACGTACGT", "ACGTTCGT", p).score_and_grad()[0]
    b1, b2 = SPB(n_threads=2), SPB(n_threads=2)
    b1.add(s)
    b2.add(s)
    assert b1.score_and_grad() == want
    assert b2.score_and_grad() == want
    b1.add(SP("ACGT", "AGT", p))  # adding more after the fact is still fine
    assert b1.score_and_grad() == want + SP("ACGT", "AGT", p).score_and_grad()[0]


# --- Issue 5: thread-launch failure ------------------------------------------------------

@pytest.mark.skipif(not sys.platform.startswith("linux"),
                    reason="RLIMIT_NPROC counting threads is Linux behaviour")
@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0,
                    reason="RLIMIT_NPROC is not enforced for root")
@pytest.mark.skipif(running_under_asan(),
                    reason="ASan's own threads and allocator make the thread budget "
                           "unpredictable; the plain build is what matters here")
@pytest.mark.parametrize("entry", ["score_and_grad", "sorted", "align_full", "add_many"])
def test_thread_launch_failure_does_not_abort(entry):
    """Cap the user's thread count a few above what is running, then ask for 32
    workers: the first launches succeed, a later one fails.  The process must survive —
    raising to Python, or completing on the threads it got with the right answer."""
    proc = run_isolated(f"""
        import os, resource, numpy as np, nwgrad as n
        p = n.AlignParams(2 * np.eye(4) - np.ones((4, 4)), alphabet="ACGT",
                          gap_open_a=2.0, gap_extend_a=1.0, gap_open_b=2.0, gap_extend_b=1.0)
        A, B = ["ACGTACGTAC"] * 64, ["ACGTTCGTAC"] * 64
        want = 64 * n.SeqPair(A[0], B[0], p).score_and_grad()[0]
        b = n.SeqPairBatch(n_threads=32)
        if "{entry}" == "sorted":
            # Static chunk-per-worker scheduler: a worker that never started must
            # not take its chunk with it.
            b.schedule = "sorted"
        if "{entry}" != "add_many":
            b.add_many(A, B, p)
            if "{entry}" == "align_full":
                b.alloc_dp()
        # Count every thread this user owns (RLIMIT_NPROC's unit), leave room for 3.
        uid = os.getuid(); used = 0
        for pid in filter(str.isdigit, os.listdir("/proc")):
            try:
                if os.stat(f"/proc/{{pid}}").st_uid == uid:
                    used += len(os.listdir(f"/proc/{{pid}}/task"))
            except OSError:
                pass
        resource.setrlimit(resource.RLIMIT_NPROC, (used + 3, used + 3))
        try:
            if "{entry}" == "add_many":
                b.add_many(A, B, p)
                got = None
            elif "{entry}" == "align_full":
                got = b.align_full()
            else:
                got = b.score_and_grad()
        except Exception as e:
            print("RAISED", type(e).__name__, e)
        else:
            assert got is None or got == want, f"{{got}} != {{want}}"
            print("COMPLETED")
        print("OK")
    """)
    assert proc.returncode == 0 and "OK" in proc.stdout, describe(proc)
