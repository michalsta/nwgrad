"""nwgrad._tuning: runtime overrides of the fill gates.

Gates choose between fills that are bit-identical, so overriding them must change no
result (scores, paths, gradients) — only which fill runs.  The routing itself is checked
through the batch's inter-pair plan, including that a batch regroups after a change.
"""

import numpy as np
import pytest

import nwgrad
from nwgrad import _tuning
from nwgrad.matrices import BLOSUM62

AA = "ARNDCQEGHILKMFPSTWYV"

# Every pair to its own fill, and the scalar linear fill at every length.
OWN = dict(inter_group_bytes=0, inter_soft_bytes=0, inter_score_gather_cells=0,
           inter_linear_int_gather_cells=0, inter_linear_f32_gather_cells=0,
           inter_linear_f64_gather_cells=0, inter_global_linear_int_cells=0,
           inter_global_linear_f32_cells=0, linear_striped_min_n=10**6)
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


def test_plan_follows_the_gates():
    # 30-500 aa: the default byte cap sends the long pairs to their own fill (with the
    # other threads busy -- n_threads 2; one thread alone has a wider budget; at W = 4
    # lanes the 3 MiB budget reaches L250)
    A, B = _pairs(60, 30, 500, seed=1)
    b = nwgrad.SeqPairBatch(A, B, _params(), n_threads=2)
    elig, other, _ = b._inter_plan()
    assert elig > 0 and other > 0 and elig + other == len(A)
    _tuning.set_gate("inter_group_bytes", None)           # the same batch regroups
    assert b._inter_plan()[:2] == [len(A), 0]
    _tuning.set_gate("inter_group_bytes", 0)
    assert b._inter_plan()[:2] == [0, len(A)]
    _tuning.reset_gates()
    assert b._inter_plan()[:2] == [elig, other]


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
