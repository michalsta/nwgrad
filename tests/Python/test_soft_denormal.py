"""The double soft pass's denormal handling: FTZ/DAZ around the double soft kernels
(``soft_double_ftz``, the default since 2026-11-06; 0 restores gradual underflow) and a
tighter lazy-rescale window (a probe that gained nothing; removed 2026-11-06).

Both lose only mass the lost-mass bound already charges: a table or carry term below
DBL_MIN in its row's stored units is counted as lost whether it is kept as a subnormal or
flushed, and the gradient pass's products are charged their worst case (every product of
the row flushed).  So each form must agree with the log path (the oracle) to the scaled
path's own tolerance on pairs that do go subnormal (long related protein pairs, low
temperature), and with the default form likewise.
"""

import numpy as np
import pytest

import nwgrad
from nwgrad import _tuning

REL = ABS = 1e-11
# dict(): the default (FTZ on since 2026-11-06); soft_double_ftz=0: gradual underflow.
FORMS = [dict(), dict(soft_double_ftz=0), dict(soft_double_ftz=1)]
MODES = [(g, m) for g in ("linear", "affine") for m in ("global", "local", "glocal")]


@pytest.fixture(autouse=True)
def _reset():
    _tuning.reset_gates()
    yield
    _tuning.reset_gates()


def protein_case(rng, L, n):
    alpha = nwgrad.PROTEIN.symbols[:20]
    M = rng.normal(-0.6, 0.6, (len(nwgrad.PROTEIN.symbols),) * 2) + 2.0 * np.eye(len(nwgrad.PROTEIN.symbols))
    P = nwgrad.AlignParams(nwgrad.SubstMatrix(M, nwgrad.PROTEIN.symbols), 3.3, 0.3, 3.3, 0.3)
    A, B = [], []
    for _ in range(n):
        a = rng.choice(list(alpha), L)
        b = a.copy()
        mut = rng.random(L) < 0.2
        b[mut] = rng.choice(list(alpha), int(mut.sum()))
        A.append("".join(a)); B.append("".join(b[: L - int(rng.integers(0, 20))]))
    return A, B, P


def run(A, B, P, gm, mode, impl, T=1.0, band=None, **gates):
    with _tuning.override(**gates):
        b = nwgrad.SeqPairBatchDouble(A, B, P, n_threads=2, gap_model=gm, mode=mode, grad_mode="soft")
        b.fill = "striped"
        b.soft_impl = impl
        b.soft_temperature = T
        b.score_and_grad()
        if band is not None:
            b.banded_grad(band)
        m, g = b.grads()
        return np.asarray(b.scores()), np.asarray(m), np.asarray(g)


def close(x, y):
    for a, b in zip(x, y):
        np.testing.assert_allclose(a, b, rtol=REL, atol=ABS)


@pytest.mark.parametrize("gm,mode", MODES)
def test_forms_match_log_on_long_pairs(gm, mode):
    """L ~250 related protein pairs: the cells far from the diagonal sit below DBL_MIN
    of their row's max (subnormal in the default form)."""
    rng = np.random.default_rng(7)
    A, B, P = protein_case(rng, 250, 3)
    ref = run(A, B, P, gm, mode, "log")
    for f in FORMS:
        close(run(A, B, P, gm, mode, "scaled", **f), ref)


@pytest.mark.parametrize("gm,mode", MODES)
@pytest.mark.parametrize("T", [0.3, 0.1])
def test_forms_match_log_low_temperature(gm, mode, T):
    rng = np.random.default_rng(11)
    A, B, P = protein_case(rng, 60, 4)
    ref = run(A, B, P, gm, mode, "log", T=T)
    for f in FORMS:
        got = run(A, B, P, gm, mode, "scaled_or_log", T=T, **f)
        close(got, ref)


@pytest.mark.parametrize("gm,mode", MODES)
def test_forms_banded(gm, mode):
    rng = np.random.default_rng(13)
    A, B, P = protein_case(rng, 150, 3)
    ref = run(A, B, P, gm, mode, "log", band=6)
    for f in FORMS:
        close(run(A, B, P, gm, mode, "scaled", band=6, **f), ref)


def test_out_of_range_still_raises():
    """A step score of 800 nats overflows exp() in every form: the scaled path still refuses."""
    M = np.full((20, 20), -1.0)
    np.fill_diagonal(M, 800.0)
    P = nwgrad.AlignParams(nwgrad.SubstMatrix(M, nwgrad.PROTEIN.symbols[:20]), 2.0, 0.5, 2.0, 0.5)
    s = "ACDEFGHIKLMNPQRSTVWY" * 3
    for f in FORMS:
        with _tuning.override(**f):
            b = nwgrad.SeqPairBatchDouble([s], [s], P, grad_mode="soft")
            b.fill = "striped"
            b.soft_impl = "scaled"
            with pytest.raises(ValueError):
                b.score_and_grad()


def test_ftz_is_the_default():
    assert _tuning.defaults()["soft_double_ftz"] == 1
