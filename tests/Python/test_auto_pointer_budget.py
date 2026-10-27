"""`auto` with a pointer budget (2026-10-25).

Under `auto`, affine Global Full at float64 (exact Hirschberg) and int64
(hirschberg_pmax), and affine Local Full at float32 and the integer types
(hirschberg_pmax) run a pair of up to 2**25 cells (Aligner::kAutoPointerCells; float64 Global 2**26) as
Pointers and split only above it; before, they split past hb_cutoff (512) rows.  With
one-byte direction codes Pointers measured faster than both recursions there (AGENTS.md).

Within the budget the pair runs the Pointers fill itself, so it must equal an explicit
traceback="pointers" BIT FOR BIT — score, gradient, path — past hb_cutoff, in a batch
whose planner mixes short and long pairs too.  The types without a budget (Global
int32) must still split exactly as their explicit mode does.

Since 2026-10-26 (`traceback="grid"`), affine Global Full at float32 and float64 resolves
to grid, which returns Pointers' result by construction; the budget applies where grid
does not (the run falls back to the old default).  So float64 Global still equals explicit
Pointers bit for bit, now reported as "grid", and float32 Global equals Pointers too.
"""
import numpy as np
import pytest

import nwgrad

ALPHA = "ACDEFGHIKLMNPQRSTVWY"


def _seqs(n, lo, hi, seed):
    rng = np.random.default_rng(seed)
    return ["".join(rng.choice(list(ALPHA), int(rng.integers(lo, hi)))) for _ in range(n)]


def _homologue(a, seed):
    rng = np.random.default_rng(seed)
    y = list(a)
    for k in rng.choice(len(y), len(y) // 3, replace=False):
        y[k] = ALPHA[rng.integers(20)]
    k = int(rng.integers(len(y)))
    y[k:k] = list(rng.choice(list(ALPHA), 7))
    del y[k // 2:k // 2 + 5]
    return "".join(y)


@pytest.fixture
def params():
    rng = np.random.default_rng(7)
    m = rng.integers(-3, 6, (20, 20)).astype(float)
    return nwgrad.AlignParams(nwgrad.SubstMatrix(m, alphabet=ALPHA), 5.0, 1.0, 6.0, 1.0)


CLS = {"double": nwgrad.SeqPairBatchDouble, "float": nwgrad.SeqPairBatch,
       "int32": nwgrad.SeqPairBatchInt32, "int64": nwgrad.SeqPairBatchInt64}


def _run(prec, mode, tb, A, B, p, fill="striped"):
    b = CLS[prec](A, B, p, n_threads=3, traceback=tb, gap_model="affine", mode=mode)
    b.fill = fill
    total = b.score_and_grad(keep_paths=True)
    out = [total, np.asarray(b.scores())]
    for i in range(len(b)):
        g = b[i].grad.to_dict()
        out.append([np.asarray(g[k]) for k in sorted(g) if not isinstance(g[k], str)])
        out.append(b[i].aligned())
    return b, out


def _same(x, y):
    if isinstance(x, (list, tuple)):
        return len(x) == len(y) and all(_same(u, v) for u, v in zip(x, y))
    if isinstance(x, np.ndarray):
        return np.array_equal(x, y)
    return x == y


def _problem(seed):
    # past hb_cutoff (512 rows) and far inside the budget; related and unrelated pairs
    A = _seqs(4, 600, 900, seed)
    B = [_homologue(a, seed + k) for k, a in enumerate(A[:2])] + _seqs(2, 600, 900, seed + 50)
    # short pairs too: the planner groups them, the long ones run their own path
    A += _seqs(10, 20, 60, seed + 1); B += _seqs(10, 20, 60, seed + 2)
    return A, B


@pytest.mark.parametrize("prec,mode", [("double", "global"), ("int64", "global"),
                                       ("float", "local"), ("int32", "local")])
@pytest.mark.parametrize("fill", ["striped", "interpair"])
def test_budgeted_auto_is_pointers_past_the_cutoff(params, prec, mode, fill):
    A, B = _problem(11)
    ba, ra = _run(prec, mode, "auto", A, B, params, fill)
    _, rp = _run(prec, mode, "pointers", A, B, params, fill)
    # resolved: grid for float64 Global (Pointers' result), the budgeted mode otherwise
    assert ba[0].traceback == ("grid" if prec == "double" else "hirschberg_pmax")
    assert _same(ra, rp)


@pytest.mark.parametrize("prec,mode,explicit", [("float", "global", "pointers"),   # grid
                                                ("int32", "global", "hirschberg_pmax")])
def test_unbudgeted_auto_still_splits_as_its_mode(params, prec, mode, explicit):
    A, B = _problem(23)
    _, ra = _run(prec, mode, "auto", A, B, params)
    _, rx = _run(prec, mode, explicit, A, B, params)
    assert _same(ra, rx)
