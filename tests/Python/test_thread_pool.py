"""The persistent worker pool (parallel.hpp, run_workers_guarded).

Multi-threaded calls run on a process-wide pool of parked threads instead of spawning
n_threads - 1 threads per call.  The pool must change nothing but the time: these tests
compare it against the spawn path (``_tuning.set_thread_pool(False)``) bit for bit, hammer
it with many small calls from several Python threads, and check the process-level
hazards — fork (a child has no workers and must build its own pool) and interpreter exit
(parked workers must not hang it).  Process-level scenarios run in a child interpreter.
"""

import os
import random
import sys
import threading

import numpy as np
import pytest

import nwgrad
from nwgrad import _tuning
from conftest import describe, run_isolated

THREADS = [1, 2, 4, 8, 16]


def dna_params():
    return nwgrad.AlignParams(2 * np.eye(4) - np.ones((4, 4)), alphabet="ACGT",
                              gap_open_a=2.0, gap_extend_a=1.0,
                              gap_open_b=2.0, gap_extend_b=1.0)


def random_pairs(n, lo, hi, seed):
    rng = random.Random(seed)
    seq = lambda: "".join(rng.choice("ACGT") for _ in range(rng.randint(lo, hi)))
    return [seq() for _ in range(n)], [seq() for _ in range(n)]


def run_batch(A, B, n_threads, grad_mode, mode, gap_model, keep_paths):
    b = nwgrad.SeqPairBatchDouble(A, B, dna_params(), n_threads=n_threads,
                                  grad_mode=grad_mode, mode=mode, gap_model=gap_model)
    total = b.score_and_grad(keep_paths=keep_paths)
    out = {"total": total, "scores": np.array(b.scores())}
    if grad_mode != "none":
        out["grads"] = np.array(b.grads()[0])
        g = b.compute_grad()
        out["grad"] = np.concatenate([np.ravel(g.matrix.to_matrix()),
                                      [g.gap_open_a, g.gap_extend_a, g.gap_open_b, g.gap_extend_b]])
    if keep_paths and grad_mode == "hard":
        out["cigars"] = b.cigars()
    return out


def assert_same(x, y):
    assert x.keys() == y.keys()
    for k in x:
        if isinstance(x[k], np.ndarray):
            assert np.array_equal(x[k], y[k]), k   # bit-identical, not close
        else:
            assert x[k] == y[k], k


@pytest.mark.parametrize("grad_mode,mode,gap_model,keep_paths", [
    ("hard", "global", "affine", True),
    ("hard", "local", "linear", True),
    ("soft", "global", "affine", False),
    ("none", "local", "affine", False),
])
@pytest.mark.parametrize("n_pairs", [1, 16, 257])
def test_pool_matches_spawn_and_one_thread(grad_mode, mode, gap_model, keep_paths, n_pairs):
    A, B = random_pairs(n_pairs, 5, 60, seed=n_pairs)
    with _tuning.thread_pool_override(False):
        ref = run_batch(A, B, 1, grad_mode, mode, gap_model, keep_paths)
    for nt in THREADS:
        with _tuning.thread_pool_override(False):
            spawn = run_batch(A, B, nt, grad_mode, mode, gap_model, keep_paths)
        with _tuning.thread_pool_override(True):
            pool = run_batch(A, B, nt, grad_mode, mode, gap_model, keep_paths)
        assert_same(pool, spawn)
        if grad_mode != "soft":   # hard / score-only: thread count cannot matter at all
            assert_same(pool, ref)


def test_pool_grows_to_the_largest_request():
    with _tuning.thread_pool_override(True):
        A, B = random_pairs(64, 5, 20, seed=1)
        nwgrad.SeqPairBatchDouble(A, B, dna_params(), n_threads=6).score_and_grad()
        assert nwgrad.nwgrad_ext._thread_pool_size() >= 5
        before = nwgrad.nwgrad_ext._thread_pool_size()
        nwgrad.SeqPairBatchDouble(A, B, dna_params(), n_threads=3).score_and_grad()
        assert nwgrad.nwgrad_ext._thread_pool_size() == before   # never shrinks


def test_switch_round_trips():
    saved = _tuning.thread_pool()
    try:
        _tuning.set_thread_pool(False)
        assert _tuning.thread_pool() is False
        _tuning.set_thread_pool(True)
        assert _tuning.thread_pool() is True
    finally:
        _tuning.set_thread_pool(saved)


def test_many_small_calls_from_several_python_threads():
    """Each thread repeats small multi-threaded calls on its own batch.  With the GIL
    held around a call they interleave rather than overlap, but construction, scores and
    gradients still cross threads; the C++ stress test (test_thread_pool.cpp) overlaps
    calls for real."""
    jobs = []
    for t in range(6):
        A, B = random_pairs(8 + 13 * t, 3, 30, seed=100 + t)
        with _tuning.thread_pool_override(False):
            ref = run_batch(A, B, 1, "hard", "global", "affine", False)
        jobs.append((A, B, 2 + t % 5, ref))
    errors = []

    def body(A, B, nt, ref):
        try:
            b = nwgrad.SeqPairBatchDouble(A, B, dna_params(), n_threads=nt)
            for _ in range(150):
                assert b.score_and_grad(keep_paths=False) == ref["total"]
                assert np.array_equal(np.array(b.scores()), ref["scores"])
        except Exception as e:   # pragma: no cover - reported below
            errors.append(repr(e))

    ths = [threading.Thread(target=body, args=j) for j in jobs]
    for th in ths:
        th.start()
    for th in ths:
        th.join(timeout=300)
    assert not any(th.is_alive() for th in ths), "a thread hung"
    assert not errors, errors


def test_threads_one_never_creates_the_pool():
    proc = run_isolated("""
        import numpy as np, nwgrad as n
        p = n.AlignParams(2 * np.eye(4) - np.ones((4, 4)), alphabet="ACGT",
                          gap_open_a=2.0, gap_extend_a=1.0, gap_open_b=2.0, gap_extend_b=1.0)
        b = n.SeqPairBatchDouble(["ACGT"] * 50, ["ACGA"] * 50, p, n_threads=1)
        b.score_and_grad()
        assert n.nwgrad_ext._thread_pool_size() == 0
        print("OK")
    """)
    assert proc.returncode == 0 and "OK" in proc.stdout, describe(proc)


def test_env_switch_disables_the_pool():
    proc = run_isolated("""
        import os
        os.environ["NWGRAD_THREAD_POOL"] = "0"
        import numpy as np, nwgrad as n
        from nwgrad import _tuning
        p = n.AlignParams(2 * np.eye(4) - np.ones((4, 4)), alphabet="ACGT",
                          gap_open_a=2.0, gap_extend_a=1.0, gap_open_b=2.0, gap_extend_b=1.0)
        b = n.SeqPairBatchDouble(["ACGT"] * 50, ["ACGA"] * 50, p, n_threads=4)
        b.score_and_grad()
        assert _tuning.thread_pool() is False
        assert n.nwgrad_ext._thread_pool_size() == 0
        print("OK")
    """)
    assert proc.returncode == 0 and "OK" in proc.stdout, describe(proc)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="no fork on this platform")
def test_fork_child_rebuilds_the_pool():
    """The parent's workers do not exist in a forked child: its first call must build a
    fresh pool (not post to dead workers and wait forever) and get the same answer."""
    proc = run_isolated("""
        import os, sys, warnings, numpy as np, nwgrad as n
        warnings.simplefilter("ignore", DeprecationWarning)   # fork() of a threaded process
        p = n.AlignParams(2 * np.eye(4) - np.ones((4, 4)), alphabet="ACGT",
                          gap_open_a=2.0, gap_extend_a=1.0, gap_open_b=2.0, gap_extend_b=1.0)
        A, B = ["ACGTACGTAC"] * 200, ["ACGTTCGTAC"] * 200
        b = n.SeqPairBatchDouble(A, B, p, n_threads=8)
        want = b.score_and_grad()
        assert n.nwgrad_ext._thread_pool_size() >= 7
        for round in range(3):
            pid = os.fork()
            if pid == 0:
                ok = n.nwgrad_ext._thread_pool_size() == 0
                ok = ok and b.score_and_grad() == want
                ok = ok and n.nwgrad_ext._thread_pool_size() >= 7
                c = n.SeqPairBatchDouble(A, B, p, n_threads=4)
                ok = ok and c.score_and_grad() == want
                os._exit(0 if ok else 3)
            _, status = os.waitpid(pid, 0)
            assert os.waitstatus_to_exitcode(status) == 0, status
            assert b.score_and_grad() == want   # the parent's pool still works
        print("OK")
    """, timeout=120)
    assert proc.returncode == 0 and "OK" in proc.stdout, describe(proc)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="no fork on this platform")
def test_multiprocessing_fork_pool():
    proc = run_isolated("""
        import multiprocessing as mp, warnings, numpy as np, nwgrad as n
        warnings.simplefilter("ignore", DeprecationWarning)
        def job(k):
            p = n.AlignParams(2 * np.eye(4) - np.ones((4, 4)), alphabet="ACGT",
                              gap_open_a=2.0, gap_extend_a=1.0, gap_open_b=2.0, gap_extend_b=1.0)
            return n.SeqPairBatchDouble(["ACGTACGT"] * (50 + k), ["ACGAACGT"] * (50 + k), p,
                                        n_threads=4).score_and_grad()
        if __name__ == "__main__":
            here = [job(k) for k in range(6)]   # the parent's pool exists before the fork
            with mp.get_context("fork").Pool(3) as pool:
                assert pool.map(job, range(6)) == here
            print("OK")
    """, timeout=120)
    assert proc.returncode == 0 and "OK" in proc.stdout, describe(proc)


@pytest.mark.parametrize("scenario", ["plain", "daemon_thread", "sys_exit_in_thread"])
def test_interpreter_exit_does_not_hang(scenario):
    """Parked workers are detached and the pool is never destroyed: exit must neither
    wait for them nor crash in a static destructor, including while another Python
    thread keeps dispatching calls."""
    proc = run_isolated(f"""
        import threading, numpy as np, nwgrad as n
        p = n.AlignParams(2 * np.eye(4) - np.ones((4, 4)), alphabet="ACGT",
                          gap_open_a=2.0, gap_extend_a=1.0, gap_open_b=2.0, gap_extend_b=1.0)
        b = n.SeqPairBatchDouble(["ACGTACGTAC"] * 300, ["ACGTTCGTAC"] * 300, p, n_threads=16)
        b.score_and_grad()
        if "{scenario}" == "daemon_thread":
            def spin():
                c = n.SeqPairBatchDouble(["ACGTACGTAC"] * 300, ["ACGTTCGTAC"] * 300, p, n_threads=8)
                while True:
                    c.score_and_grad()
            threading.Thread(target=spin, daemon=True).start()
            for _ in range(20):
                b.score_and_grad()
        elif "{scenario}" == "sys_exit_in_thread":
            import os
            def leave():
                b.score_and_grad()
                os._exit(0)
            threading.Thread(target=leave).start()
            threading.Event().wait(60)
        print("OK", flush=True)
    """, timeout=60)
    assert proc.returncode == 0, describe(proc)
    if scenario != "sys_exit_in_thread":
        assert "OK" in proc.stdout, describe(proc)
