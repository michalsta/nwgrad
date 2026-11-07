"""nwgrad._tuning: runtime overrides of the fill gates.

Gates choose between fills that are bit-identical, so overriding them must change no
result (scores, paths, gradients) — only which fill runs.  The routing itself is checked
through the batch's inter-pair plan, including that a batch regroups after a change.
"""

import os

import numpy as np
import pytest

import conftest
import nwgrad
from nwgrad import _tuning
from nwgrad.matrices import BLOSUM62

AA = "ARNDCQEGHILKMFPSTWYV"

# Every pair to its own fill, and the scalar linear fill at every length.
OWN = dict(inter_group_bytes=0, inter_soft_bytes=0, inter_score_gather_cells=0,
           inter_linear_int_gather_cells=0, inter_linear_f32_gather_cells=0,
           inter_linear_f64_gather_cells=0, inter_global_linear_int_cells=0,
           inter_global_linear_f32_cells=0, linear_striped_min_n=10**6,
           soft_f32_min_cells=0,   # (float32 at any size, as with the gate off: one precision)
           inter_min_fill=100,
           soft_ckpt_rows=0, soft_ckpt_min_bytes=0,   # (soft only: the hard results below ignore them)
           soft_double_ftz=0, soft_guide_defer=0)
# Every gate off: the shared fill wherever it can run, the striped linear kernel always.
OFF = {g: None for g in OWN}

CLASSES = {"float32": nwgrad.SeqPairBatch, "float64": nwgrad.SeqPairBatchDouble,
           "int16": nwgrad.SeqPairBatchInt16, "int32": nwgrad.SeqPairBatchInt32,
           "int64": nwgrad.SeqPairBatchInt64}


@pytest.fixture(autouse=True)
def _reset():
    _tuning.reset_gates()
    yield
    _tuning.reset_gates()


def _pairs(n, lo, hi, seed):
    rng = np.random.default_rng(seed)
    A, B = [], []
    for la in rng.integers(lo, hi, n):
        a = rng.choice(list(AA), int(la))
        b = a.copy()
        mut = rng.random(len(b)) < 0.3
        b[mut] = rng.choice(list(AA), int(mut.sum()))
        b = np.delete(b, rng.integers(0, len(b), 2))
        A.append("".join(a)); B.append("".join(b))
    return A, B


def _params():
    return nwgrad.AlignParams(BLOSUM62, 11.0, 1.0, 11.0, 1.0)


def test_api_roundtrip():
    assert set(_tuning.GATES) == set(OWN)
    d = _tuning.defaults()
    assert d["inter_score_gather_cells"] > 0 and d["linear_striped_min_n"] == "per type"
    assert all(v == "default" for v in _tuning.gates().values())
    _tuning.set_gate("inter_score_gather_cells", 100)
    assert _tuning.get_gate("inter_score_gather_cells") == 100
    _tuning.set_gate("inter_score_gather_cells", None)
    assert _tuning.get_gate("inter_score_gather_cells") is None
    _tuning.set_gate("linear_striped_min_n", None)       # off = striped from any length
    assert _tuning.get_gate("linear_striped_min_n") == 0
    _tuning.set_gate("inter_score_gather_cells", "default")
    assert _tuning.get_gate("inter_score_gather_cells") == "default"
    for bad in (-1, 1.5, "off"):
        with pytest.raises(ValueError):
            _tuning.set_gate("inter_group_bytes", bad)
    with pytest.raises(ValueError):
        _tuning.set_gate("no_such_gate", 1)


def test_override_restores():
    _tuning.set_gate("inter_group_bytes", 5)
    with _tuning.override(inter_group_bytes=None, inter_soft_bytes=7):
        assert _tuning.get_gate("inter_group_bytes") is None
        assert _tuning.get_gate("inter_soft_bytes") == 7
    assert _tuning.get_gate("inter_group_bytes") == 5
    assert _tuning.get_gate("inter_soft_bytes") == "default"


def _needs_shared_fill():
    # NWGRAD_FORCE_KERNEL=scalar_fallback injects that kernel, which has no shared fill:
    # every plan is [0, N], so a test of how the plan follows the gates has nothing to see.
    if conftest._current_backend == "scalar_fallback":
        pytest.skip("the scalar backend has no shared fill")


def test_plan_follows_the_gates():
    # 30-500 aa: the default byte cap sends the long pairs to their own fill (with the
    # other threads busy -- n_threads 2; one thread alone has a wider budget; at W = 4
    # lanes the 3 MiB budget reaches L250).  Enough pairs to fill 16-lane groups.
    _needs_shared_fill()
    A, B = _pairs(300, 30, 500, seed=1)
    b = nwgrad.SeqPairBatch(A, B, _params(), n_threads=2)
    elig, other, _ = b._inter_plan()
    assert elig > 0 and other > 0 and elig + other == len(A)
    _tuning.set_gate("inter_group_bytes", None)           # the same batch regroups
    with _tuning.override(inter_min_fill=None):           # (mostly empty groups too)
        assert b._inter_plan()[:2] == [len(A), 0]
    _tuning.set_gate("inter_group_bytes", 0)
    assert b._inter_plan()[:2] == [0, len(A)]
    _tuning.reset_gates()
    assert b._inter_plan()[:2] == [elig, other]


def test_kept_paths_keep_the_byte_budget():
    _needs_shared_fill()
    # grad_mode "none" with keep_paths runs the hard shared fill, tables and all: the byte
    # budget applies (it did not, 0b52842..: L1000 path calls ran 2x slower than their own
    # fill).  Only the score-only call, whose kernel keeps rows, goes without it.
    A, B = _pairs(64, 400, 401, seed=3)
    b = nwgrad.SeqPairBatch(A, B, _params(), n_threads=2, grad_mode="none")
    assert b._inter_plan()[:2] == [0, len(A)]
    elig, other, _ = b._inter_plan(score_only=True)
    assert elig > 0 and elig + other == len(A)
    ref = nwgrad.SeqPairBatch(A, B, _params(), n_threads=2, grad_mode="none")
    ref.fill = "striped"
    b.score_and_grad(keep_paths=True); ref.score_and_grad(keep_paths=True)
    assert list(b.scores()) == list(ref.scores())
    assert [b[i].aligned() for i in range(len(A))] == [ref[i].aligned() for i in range(len(A))]


def test_hyperthreads_shrink_the_byte_budget():
    _needs_shared_fill()
    # More threads than physical cores: two threads share a core's caches and the hard
    # budget's default is 9/16 (inter_pair_fits).  The longest square pair that shares at
    # 2 threads runs its own fill at 4x the logical count; one 0.7 as long (cells 0.49x,
    # under 9/16) still shares; an override is taken as given.
    rng = np.random.default_rng(7)

    def plan(L, n_threads):
        A = ["".join(rng.choice(list(AA), L)) for _ in range(64)]
        return nwgrad.SeqPairBatch(A, A, _params(), n_threads=n_threads,
                                   traceback="pointers")._inter_plan()[:2]

    lo, hi = 10, 4000
    assert plan(lo, 2) == [64, 0] and plan(hi, 2) == [0, 64]
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if plan(mid, 2) == [64, 0] else (lo, mid)
    many = 4 * (os.cpu_count() or 1)
    assert plan(lo, many) == [0, 64]
    assert plan(int(0.7 * lo), many) == [64, 0]
    _tuning.set_gate("inter_group_bytes", 10**9)
    assert plan(lo, many) == [64, 0]


def test_dna_alone_keeps_a_wider_byte_budget():
    # One thread, alphabet <= 8: the mask-blend shared fill keeps winning to ~16 MiB of
    # tables (protein int16's budget is 450 KiB-1 MiB).  DNA int16 affine L150: ~2-4 MB.
    level = nwgrad.simd_isa()
    if level not in ("avx2", "avx512") or os.environ.get("NWGRAD_FORCE_KERNEL"):
        pytest.skip("the small-alphabet budget is set for AVX2 / AVX-512")
    rng = np.random.default_rng(6)
    A = ["".join(rng.choice(list("ACGT"), 150)) for _ in range(64)]
    m = np.where(np.eye(4, dtype=bool), 2.0, -1.0)
    p = nwgrad.AlignParams(nwgrad.SubstMatrix(m, alphabet="ACGT"), 3.0, 1.0, 3.0, 1.0)
    one = nwgrad.SeqPairBatchInt16(A, A, p, n_threads=1)
    two = nwgrad.SeqPairBatchInt16(A, A, p, n_threads=2)
    assert one._inter_plan()[:2] == [64, 0]
    assert two._inter_plan()[:2] == [0, 64]


def test_mostly_empty_groups_run_their_own_fill():
    _needs_shared_fill()
    # A group's empty lanes cost as much as its real ones: one short pair alone runs its
    # own fill; with the gate off it shares.  A batch of whole groups shares regardless.
    A, B = _pairs(1, 60, 61, seed=4)
    for gm in ("hard", "none"):
        b = nwgrad.SeqPairBatch(A, B, _params(), n_threads=1, grad_mode=gm)
        assert b._inter_plan()[:2] == [0, 1]
        assert b._inter_plan(score_only=True)[:2] == [0, 1]
        with _tuning.override(inter_min_fill=None):
            assert b._inter_plan()[:2] == [1, 0]
        assert b._inter_plan()[:2] == [0, 1]
    A, B = _pairs(64, 60, 61, seed=5)
    b = nwgrad.SeqPairBatch(A, B, _params(), n_threads=1)
    assert b._inter_plan()[:2] == [64, 0]


@pytest.mark.parametrize("dtype", list(CLASSES))
@pytest.mark.parametrize("mode", ["global", "local"])
@pytest.mark.parametrize("gap_model", ["affine", "linear"])
def test_results_do_not_depend_on_gates(dtype, mode, gap_model):
    A, B = _pairs(40, 5, 180, seed=2)
    p = _params()
    out = {}
    for name, gates in (("default", {}), ("own", OWN), ("off", OFF)):
        with _tuning.override(**gates):
            # score only (the score kernels and their gather gate)
            s = CLASSES[dtype](A, B, p, gap_model=gap_model, mode=mode, grad_mode="none", n_threads=2)
            s.score_and_grad()
            # hard, with paths (the shared Viterbi fill, the linear striped / scalar fills)
            h = CLASSES[dtype](A, B, p, gap_model=gap_model, mode=mode, grad_mode="hard",
                               n_threads=2, traceback="pointers")
            h.score_and_grad(keep_paths=True)
            mats, gaps = h.grads()
            out[name] = (s.scores(), h.scores(), [h[i].aligned() for i in range(len(A))], mats, gaps)
    ref = out["default"]
    for name in ("own", "off"):
        got = out[name]
        assert np.array_equal(got[0], ref[0]), name
        assert np.array_equal(got[1], ref[1]), name
        assert got[2] == ref[2], name
        assert np.array_equal(got[3], ref[3]) and np.array_equal(got[4], ref[4]), name
