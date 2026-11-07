"""soft_guide = "eager" with its guide deferred to first use (the soft_guide_defer gate).

A soft score_and_grad() under "eager" leaves every pair's guide PENDING under a snapshot
of the params it scored; guide_j / banded_align / banded_grad compute it on first use.
The contract tested here:
- the guide is the eager one, bit for bit (same hard fill, same params), whenever it is
  asked for -- also after set_params() or an in-place change of the caller's params;
- every soft output (scores, gradients) is bit-identical to the gate-off arm, which runs
  the Viterbi inside the call as before;
- a call that never asks pays no Viterbi (the guides stay pending);
- the rule when params change: the snapshot is the params the LAST full soft call
  scored ("eager"); "lazy" keeps following the params current at first use.
"""

import threading

import numpy as np
import pytest

import nwgrad
from nwgrad import _tuning

MODES = [(g, m) for g in ("linear", "affine") for m in ("global", "local", "glocal")]
CLASSES = {"double": nwgrad.SeqPairBatchDouble, "float32": nwgrad.SeqPairBatch}


@pytest.fixture(autouse=True)
def _reset():
    _tuning.reset_gates()
    yield
    _tuning.reset_gates()


def rand_seq(rng, alpha, n):
    return "".join(rng.choice(list(alpha), n))


def dna_params(rng, gm):
    gaps = (1.0, 0.5, 1.5, 0.3) if gm == "affine" else (0.0, 0.9, 0.0, 1.2)
    return nwgrad.AlignParams(nwgrad.SubstMatrix(rng.normal(0.0, 1.0, (4, 4)), "ACGT"), *gaps)


def prot_params(rng, gm):
    gaps = (2.0, 0.5, 3.0, 1.0) if gm == "affine" else (0.0, 1.5, 0.0, 1.0)
    M = rng.normal(-0.5, 1.0, (20, 20))
    return nwgrad.AlignParams(nwgrad.SubstMatrix(M, nwgrad.PROTEIN.symbols), *gaps)


def pairs(rng, kind, n):
    if kind == "dna":   # miRNA-like: short A, B of a few shared lengths (interpair groups)
        A = [rand_seq(rng, "ACGT", int(rng.integers(5, 25))) for _ in range(n)]
        B = [rand_seq(rng, "ACGT", int(rng.choice([30, 40]))) for _ in range(n)]
    else:
        A = [rand_seq(rng, nwgrad.PROTEIN.symbols, int(rng.integers(20, 70))) for _ in range(n)]
        B = [rand_seq(rng, nwgrad.PROTEIN.symbols, int(rng.integers(20, 70))) for _ in range(n)]
    A += ["", "A", rand_seq(rng, A[0][:1] or "A", 3)]   # edge lengths
    B += ["AC" if kind == "dna" else "AR", "", "A"]
    if kind != "dna":
        A[-3:] = ["", "A", "AAA"]
        B[-3:] = ["AR", "", "A"]
    return A, B


def make(cls, A, B, params, gm, mode, fill, threads=3, **kw):
    b = cls(A, B, params, n_threads=threads, gap_model=gm, mode=mode, grad_mode="soft", **kw)
    b.fill = fill
    return b


def guides(b):
    return [list(b[i].guide_j) for i in range(len(b))]


def outputs(b):
    m, g = b.grads()
    return np.asarray(b.scores()), np.asarray(m), np.asarray(g)


def run(defer, build):
    _tuning.set_gate("soft_guide_defer", defer)
    try:
        b = build()
        b.score_and_grad()
        out = outputs(b)
        pend = b._guides_pending()
    finally:
        _tuning.reset_gates()
    return b, out, pend


@pytest.mark.parametrize("gm,mode", MODES)
@pytest.mark.parametrize("fill", ["striped", "rowwise", "interpair"])
@pytest.mark.parametrize("cname", list(CLASSES))
@pytest.mark.parametrize("kind", ["dna", "protein"])
def test_deferred_equals_eager(gm, mode, fill, cname, kind):
    """Every soft output and every guide: the deferred arm == the eager arm, bit for bit."""
    rng = np.random.default_rng(abs(hash((gm, mode, fill, cname, kind))) % 2**32)
    P = dna_params(rng, gm) if kind == "dna" else prot_params(rng, gm)
    A, B = pairs(rng, kind, 40)
    build = lambda: make(CLASSES[cname], A, B, P, gm, mode, fill)
    e, (es, em, eg), ep = run(0, build)
    d, (ds, dm, dg), dp = run(1, build)
    assert ep == 0 and dp == len(A)
    np.testing.assert_array_equal(es, ds)
    np.testing.assert_array_equal(em, dm)
    np.testing.assert_array_equal(eg, dg)
    assert guides(d) == guides(e)
    assert d._guides_pending() == 0
    # Banding around a deferred guide == banding around the eager one.
    d2, _, _ = run(1, build)
    assert d2.banded_grad(4) == e.banded_grad(4)
    np.testing.assert_array_equal(outputs(d2)[1], outputs(e)[1])


@pytest.mark.parametrize("gm", ["linear", "affine"])
def test_pending_only_when_unasked(gm):
    """Which calls leave the guides pending, and that first use resolves one pair only."""
    rng = np.random.default_rng(5)
    P = dna_params(rng, gm)
    A, B = pairs(rng, "dna", 20)
    N = len(A)
    b = make(nwgrad.SeqPairBatchDouble, A, B, P, gm, "local", "interpair")
    assert b.soft_guide == "eager"
    b.score_and_grad()
    assert b._guides_pending() == N
    b[3].guide_j
    assert b._guides_pending() == N - 1
    b.score_and_grad(keep_paths=True)      # stored paths need the Viterbi now
    assert b._guides_pending() == 0
    b.score_and_grad()
    assert b._guides_pending() == N
    b.banded_grad(3)
    assert b._guides_pending() == 0
    b.soft_guide = "posterior"
    b.score_and_grad()
    assert b._guides_pending() == 0
    b.soft_guide = "lazy"
    b.score_and_grad()
    assert b._guides_pending() == N
    b.soft_guide = "eager"
    _tuning.set_gate("soft_guide_defer", 0)
    b.score_and_grad()
    assert b._guides_pending() == 0


@pytest.mark.parametrize("gm,mode", MODES)
@pytest.mark.parametrize("fill", ["striped", "interpair"])
def test_snapshot_rule(gm, mode, fill):
    """The rule: an "eager" guide is the Viterbi path under the params the last full soft
    call scored, whenever it is computed.  set_params() after the call, or an in-place
    change of the caller's AlignParams, cannot move it; the next soft call re-snapshots.
    "lazy" follows the params current at first use."""
    rng = np.random.default_rng(abs(hash((gm, mode, fill, "snap"))) % 2**32)
    P0, P1 = dna_params(rng, gm), dna_params(rng, gm)
    A, B = pairs(rng, "dna", 30)
    cls = nwgrad.SeqPairBatchDouble

    def ref(P):   # the eager guides and banded scores under P, gate off
        _tuning.set_gate("soft_guide_defer", 0)
        r = make(cls, A, B, P, gm, mode, fill)
        r.score_and_grad()
        _tuning.reset_gates()
        return guides(r)

    g0, g1 = ref(P0), ref(P1)
    assert g0 != g1   # (the two params must disagree somewhere for the test to bite)

    # set_params between the soft call and the guide request: the guide is P0's.
    b = make(cls, A, B, P0, gm, mode, fill)
    b.score_and_grad()
    b.set_params(P1)
    assert guides(b) == g0

    # The training step score_and_grad(); set_params(); banded_grad(): bands around P0's
    # path, exactly as the eager arm does.
    _tuning.set_gate("soft_guide_defer", 0)
    e = make(cls, A, B, P0, gm, mode, fill)
    e.score_and_grad()
    e.set_params(P1)
    _tuning.reset_gates()
    d = make(cls, A, B, P0, gm, mode, fill)
    d.score_and_grad()
    d.set_params(P1)
    assert d._guides_pending() == len(A)
    assert d.banded_grad(3) == e.banded_grad(3)
    np.testing.assert_array_equal(outputs(d)[1], outputs(e)[1])
    assert guides(d) == guides(e)

    # An in-place change of the caller's params object: the batch scored its own copy.
    Pm = P0 * 1.0   # (a copy the test owns)
    b = make(cls, A, B, Pm, gm, mode, fill)
    b.score_and_grad()
    Pm += P1
    assert guides(b) == g0

    # A second soft call re-snapshots: the guide is then the new params'.
    b = make(cls, A, B, P0, gm, mode, fill)
    b.score_and_grad()
    b.set_params(P1)
    b.score_and_grad()
    b.set_params(P0)
    assert guides(b) == g1

    # Mixed: half the guides used under P0, the rest after two set_params -- still P0's.
    b = make(cls, A, B, P0, gm, mode, fill)
    b.score_and_grad()
    first = [list(b[i].guide_j) for i in range(0, len(A), 2)]
    b.set_params(P1)
    b.set_params(P1 * 2.0)
    assert first == g0[0::2]
    assert guides(b) == g0

    # lazy: the params current at first use.
    b = make(cls, A, B, P0, gm, mode, fill)
    b.soft_guide = "lazy"
    b.score_and_grad()
    b.set_params(P1)
    assert guides(b) == g1


def test_pair_views_and_standalone():
    """One pair's own score_and_grad (a view of the batch) next to other pairs pending
    under an older snapshot: theirs stay that snapshot's, its own guide follows the
    params it scored.  A standalone pair defers too (keep_paths=False)."""
    rng = np.random.default_rng(17)
    P0, P1 = dna_params(rng, "affine"), dna_params(rng, "affine")
    A, B = pairs(rng, "dna", 12)
    cls = nwgrad.SeqPairBatchDouble

    def ref(P):
        _tuning.set_gate("soft_guide_defer", 0)
        r = make(cls, A, B, P, "affine", "local", "striped")
        r.score_and_grad()
        _tuning.reset_gates()
        return guides(r)

    g0, g1 = ref(P0), ref(P1)
    b = make(cls, A, B, P0, "affine", "local", "striped")
    b.score_and_grad()
    b.set_params(P1)
    b[2].score_and_grad(keep_paths=False)
    b[5].score_and_grad(keep_paths=False)
    got = guides(b)
    for i in range(len(A)):
        assert got[i] == (g1[i] if i in (2, 5) else g0[i]), i
    # Under unchanged params a view's call defers too.
    b.score_and_grad()
    b[4].score_and_grad(keep_paths=False)
    assert b._guides_pending() == len(A)
    assert guides(b) == g1

    sp = nwgrad.SeqPairDouble(A[0], B[0], P0, gap_model="affine", mode="local", grad_mode="soft")
    sp.score_and_grad(keep_paths=False)
    sp.set_params(P1)
    assert list(sp.guide_j) == g0[0]
    sp.score_and_grad(keep_paths=False)
    assert list(sp.guide_j) == g1[0]


def test_tracked_batch():
    """Tracked soft batches defer as well; guides and outputs as the eager arm's."""
    rng = np.random.default_rng(23)
    K = 3
    mats = rng.normal(0, 1.0, size=(K, 4, 4))
    gaps = rng.uniform(0.3, 1.5, size=(K, 4))
    P0 = nwgrad.AlignParams.from_arrays(mats, gaps, "ACGT")
    P1 = nwgrad.AlignParams.from_arrays(rng.normal(0, 1.0, size=(K, 4, 4)), gaps, "ACGT")
    A, B = pairs(rng, "dna", 24)
    tracks = [list(rng.integers(0, K, len(a))) for a in A]
    gtracks = [list(rng.integers(0, K, len(a) + 1)) for a in A]
    for fill in ("striped", "interpair"):
        build = lambda: make(nwgrad.SeqPairBatchDouble, A, B, P0, "affine", "global", fill,
                             tracks=tracks, gap_tracks=gtracks)
        e, eo, ep = run(0, build)
        d, do, dp = run(1, build)
        assert ep == 0 and dp == len(A)
        for x, y in zip(eo, do):
            np.testing.assert_array_equal(x, y)
        e.set_params(P1)
        d.set_params(P1)
        assert d.banded_grad(3) == e.banded_grad(3)
        assert guides(d) == guides(e)


@pytest.mark.parametrize("fill", ["striped", "interpair"])
def test_threads(fill):
    """Deferred guides resolved inside a parallel banded_grad (each pair by the one
    worker that owns it) and from several Python threads at once: the eager results,
    every time, at every thread count."""
    rng = np.random.default_rng(31)
    P0, P1 = prot_params(rng, "affine"), prot_params(rng, "affine")
    A, B = pairs(rng, "protein", 300)
    cls = nwgrad.SeqPairBatchDouble
    _tuning.set_gate("soft_guide_defer", 0)
    e = make(cls, A, B, P0, "affine", "local", fill, threads=1)
    e.score_and_grad()
    e.set_params(P1)
    want_sum = e.banded_grad(5)
    want = outputs(e)
    want_g = guides(e)
    _tuning.reset_gates()
    for threads in (2, 8, 16):
        for _ in range(2):
            d = make(cls, A, B, P0, "affine", "local", fill, threads=threads)
            d.score_and_grad()
            d.set_params(P1)
            assert d.banded_grad(5) == want_sum
            for x, y in zip(outputs(d), want):
                np.testing.assert_array_equal(x, y)
            assert guides(d) == want_g
    # guide_j from many Python threads on one batch, each pair asked several times.
    _tuning.set_gate("soft_guide_defer", 0)
    e = make(cls, A, B, P0, "affine", "local", fill, threads=4)
    e.score_and_grad()
    g0 = guides(e)
    _tuning.reset_gates()
    d = make(cls, A, B, P0, "affine", "local", fill, threads=4)
    d.score_and_grad()
    d.set_params(P1)
    got = [None] * len(A)
    errs = []

    def ask(k):
        try:
            for i in list(range(k, len(A), 3)) + list(range(len(A))):
                g = list(d[i].guide_j)
                if got[i] is None:
                    got[i] = g
                elif got[i] != g:
                    errs.append(i)
        except Exception as x:   # pragma: no cover
            errs.append(x)

    ts = [threading.Thread(target=ask, args=(k,)) for k in range(6)]
    for t in ts: t.start()
    for t in ts: t.join()
    assert not errs
    assert got == g0


def test_exception_keeps_unfinished_guides():
    """A soft call that raises part-way (a pair out of the scaled path's range) after
    set_params(): the pairs it did not finish keep the guide of the snapshot they were
    pending under, the pairs it finished take the new params'."""
    rng = np.random.default_rng(41)
    P0 = prot_params(rng, "affine")
    A, B = pairs(rng, "protein", 20)
    # A long self-pair last: at a tiny temperature it leaves the scaled range.
    long = rand_seq(rng, nwgrad.PROTEIN.symbols, 400)
    A.append(long); B.append(long)
    cls = nwgrad.SeqPairBatchDouble

    def eager(P, T):
        _tuning.set_gate("soft_guide_defer", 0)
        r = make(cls, A, B, P, "affine", "local", "striped", threads=1)
        r.soft_impl = "scaled_or_log"
        r.soft_temperature = T
        r.score_and_grad()
        _tuning.reset_gates()
        return guides(r)

    P1 = P0 * 3.0
    g0 = eager(P0, 1.0)
    b = make(cls, A, B, P0, "affine", "local", "striped", threads=1)
    b.score_and_grad()
    b.set_params(P1)
    b.soft_temperature = 1e-3
    try:
        b.score_and_grad()
    except ValueError:
        pass
    else:
        pytest.skip("no pair left the scaled range at this temperature")
    g1 = eager(P1, 1.0)   # (the guide is the hard Viterbi: no temperature in it)
    got = guides(b)
    done = [b[i].score_valid for i in range(len(A))]
    assert not all(done)
    for i in range(len(A)):
        assert got[i] == (g1[i] if done[i] else g0[i]), i
