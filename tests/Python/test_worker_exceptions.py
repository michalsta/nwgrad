"""Exceptions raised inside batch worker threads must reach the caller.

An exception escaping a std::thread's callable calls std::terminate, which
aborts the interpreter with no traceback.  Every batch entry point that can
throw from a worker is exercised here with enough pairs and threads that the
throw lands on a spawned thread rather than the caller's.
"""

import numpy as np
import pytest
import nwgrad
from test_subst_matrix import BLOSUM62


N_PAIRS = 8
N_THREADS = 8


def make_params(gap_extend=1.0, gap_open=11.0):
    return nwgrad.AlignParams(BLOSUM62,
                              gap_open_a=gap_open, gap_extend_a=gap_extend,
                              gap_open_b=gap_open, gap_extend_b=gap_extend)


def make_batch(params, grad_mode="hard"):
    pairs = [nwgrad.SeqPair("ACDEFGHIK", "ACDEFGHIK", params,
                            gap_model="affine", mode="local", grad_mode=grad_mode)
             for _ in range(N_PAIRS)]
    batch = nwgrad.SeqPairBatch(n_threads=N_THREADS)
    for sp in pairs:
        batch.add(sp)
    # Keep the SeqPairs alive for the caller: the batch holds them by pointer.
    return batch, pairs


def test_compute_grad_before_align_raises_not_aborts():
    """The precondition throw fires inside a worker.  Must surface as a Python
    exception, not SIGABRT."""
    params = make_params()
    batch, _pairs = make_batch(params)
    batch.alloc_dp()
    with pytest.raises(RuntimeError, match="align_full"):
        batch.compute_grad()


def test_compute_grad_with_grad_mode_none_raises():
    """grad_mode=None makes SeqPair::compute_grad() throw, again from a worker."""
    params = make_params()
    batch, _pairs = make_batch(params, grad_mode="none")
    batch.alloc_dp()
    batch.align_full()
    with pytest.raises(RuntimeError, match="grad_mode"):
        batch.compute_grad()


def test_align_full_without_alloc_dp_raises():
    """The own-buffer precondition throw, from a worker."""
    params = make_params()
    batch, _pairs = make_batch(params)
    with pytest.raises(RuntimeError):
        batch.align_full()


def test_realign_banded_without_path_raises():
    """realign_banded() requires a prior align_full(); throws from a worker."""
    params = make_params()
    batch, _pairs = make_batch(params)
    batch.alloc_dp()
    with pytest.raises(RuntimeError, match="align_full"):
        batch.realign_banded(3)


def test_batch_add_keeps_seqpairs_alive():
    """The batch holds SeqPairs by raw pointer, so add() must own a reference.
    Dropping every Python reference must not leave the batch dangling."""
    params = make_params()
    batch = nwgrad.SeqPairBatch(n_threads=2)
    for _ in range(4):
        batch.add(nwgrad.SeqPair("ACDEFGHIK", "ACDEFGHIK", params,
                                 gap_model="affine", mode="local", grad_mode="hard"))
    import gc
    gc.collect()                       # nothing else references the SeqPairs
    total = batch.score_and_grad()     # must not use-after-free
    assert total == pytest.approx(4 * batch[0].score)


def test_batch_add_is_linear_not_quadratic():
    """add() used to be O(N^2): nanobind's keep_alive walks the nurse's patient
    list on every call to deduplicate.  The memory bug hid it -- you OOMed at
    513 KiB/pair long before N got big enough to notice."""
    import time
    params = make_params()

    def time_adds(n):
        pairs = [nwgrad.SeqPair("ACDEFGHIK", "ACDEFGHIK", params, gap_model="affine",
                                mode="local", grad_mode="hard") for _ in range(n)]
        batch = nwgrad.SeqPairBatch(n_threads=1)
        t = time.perf_counter()
        for sp in pairs:
            batch.add(sp)
        return time.perf_counter() - t

    time_adds(2_000)                   # warm up
    small = time_adds(10_000)
    large = time_adds(40_000)

    # 4x the pairs. Linear predicts ~4x the time; quadratic predicts ~16x.
    # Allow a lot of slack for a loaded machine and still catch quadratic.
    assert large < small * 8, (
        f"batch.add() looks super-linear: 10k took {small:.3f}s, "
        f"40k took {large:.3f}s ({large / max(small, 1e-9):.1f}x for 4x the pairs)")


def _fresh_params(scale):
    return nwgrad.AlignParams(nwgrad.SubstMatrix(BLOSUM62 * scale),
                              gap_open_a=11.0, gap_extend_a=1.0,
                              gap_open_b=11.0, gap_extend_b=1.0)


def test_set_params_keeps_the_current_params_alive():
    """C++ holds params by bare pointer, so Python must own a reference to
    whichever one is current -- even if the caller drops theirs."""
    sp = nwgrad.SeqPair("ACDEFGHIK", "ACDEFGHIK", make_params(),
                        gap_model="affine", mode="local", grad_mode="hard")
    sp.set_params(_fresh_params(2.0))     # no local reference kept
    import gc
    gc.collect()
    sp.alloc_dp()
    sp.align_full()                       # must not use-after-free
    _score, grad = sp.score_and_grad()

    ref = nwgrad.SeqPair("ACDEFGHIK", "ACDEFGHIK", _fresh_params(2.0),
                         gap_model="affine", mode="local", grad_mode="hard")
    _s2, want = ref.score_and_grad()
    np.testing.assert_allclose(grad.matrix.to_matrix(), want.matrix.to_matrix(),
                               atol=1e-12)


def test_set_params_releases_superseded_params():
    """A descent loop builds a fresh AlignParams each step.  nb::keep_alive
    pinned every one of them for the SeqPair's whole life -- 50k swaps retained
    ~16 MiB.  Only the current params should survive."""
    import gc, resource

    def rss():
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024

    sp = nwgrad.SeqPair("ACDEFGHIK", "ACDEFGHIK", make_params(),
                        gap_model="affine", mode="local", grad_mode="hard")

    for i in range(2_000):                       # settle the allocator first
        sp.set_params(_fresh_params(1.0 + i * 1e-9))
    gc.collect()
    before = rss()

    for i in range(50_000):
        sp.set_params(_fresh_params(2.0 + i * 1e-9))
    gc.collect()
    grew = rss() - before

    # ru_maxrss is a high-water mark, so this can only ever over-report. Each
    # retained AlignParams is a few hundred bytes; pinning 50k of them showed up
    # as ~16 MiB. 4 MiB is comfortably below that and above allocator noise.
    assert grew < 4 * 2**20, (
        f"set_params() retained superseded params: RSS grew {grew / 2**20:.1f} MiB "
        f"over 50k swaps")


def test_set_params_is_linear_not_quadratic():
    """Each set_params() used to append to a linked list that nanobind walks on
    every call, so K swaps cost O(K^2)."""
    import time
    sp = nwgrad.SeqPair("ACDEFGHIK", "ACDEFGHIK", make_params(),
                        gap_model="affine", mode="local", grad_mode="hard")

    def time_swaps(k):
        ps = [_fresh_params(1.0 + i * 1e-9) for i in range(k)]
        t = time.perf_counter()
        for p in ps:
            sp.set_params(p)
        return time.perf_counter() - t

    time_swaps(500)                       # warm up
    small = time_swaps(2_000)
    large = time_swaps(8_000)

    # 4x the swaps. Linear predicts ~4x; quadratic predicts ~16x.
    assert large < small * 8, (
        f"set_params() looks super-linear: 2k took {small:.3f}s, "
        f"8k took {large:.3f}s ({large / max(small, 1e-9):.1f}x for 4x the swaps)")


def test_batch_still_usable_after_a_caught_worker_exception():
    """A rethrown worker exception must leave the process healthy: the same
    batch, driven correctly, still produces the right answer."""
    params = make_params()
    batch, _pairs = make_batch(params)
    batch.alloc_dp()

    with pytest.raises(RuntimeError):
        batch.compute_grad()          # too early

    batch.align_full()                # now in the right order
    grad = batch.compute_grad()

    single = nwgrad.SeqPair("ACDEFGHIK", "ACDEFGHIK", params,
                            gap_model="affine", mode="local", grad_mode="hard")
    _score, one = single.score_and_grad()

    np.testing.assert_allclose(grad.matrix.to_matrix(),
                               one.matrix.to_matrix() * N_PAIRS, atol=1e-12)
