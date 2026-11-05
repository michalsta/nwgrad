"""The inter-pair kernels' substitution lookup forms, each against the pair's own fill.

inter_kernel_impl.inl picks the lookup per job (inter_lv_dispatch): a byte table indexed
by 4a + b for integer lanes on <= 4 letters whose scores fit int8 (at levels with a byte
shuffle), a blend tree otherwise (2 levels up to 4 letters, 3 up to 8), whose row profile
is built by byte shuffle where every score is exactly an int8 (floats: -0 is not) and
gathered lane by lane where not.  Every form must leave each lane's scores, paths and
hard gradients bit-identical to the scalar own fill, at every level; these fixtures
reach each one: int8-fitting and wider integer matrices, integral, non-integral and
-0-holding float matrices, 4 and 6 letters, ragged B lengths and lanes of every A length
(Global's capture visits the lanes in order of m).
"""
import numpy as np
import pytest

import nwgrad
from nwgrad import _tuning

LEVELS = [lv for lv in nwgrad.available_isa_levels() if lv != "scalar_fallback"]
CLS = {"int16": nwgrad.SeqPairBatchInt16, "int32": nwgrad.SeqPairBatchInt32,
       "int64": nwgrad.SeqPairBatchInt64, "float32": nwgrad.SeqPairBatch,
       "double": nwgrad.SeqPairBatchDouble}
ALPHA = {4: "ACGT", 6: "ACGTNX"}


def _matrix(kind, na, rng):
    if kind == "int8":       # every score an int8: the byte table / the shuffled profile
        m = rng.integers(-6, 7, (na, na)).astype(float)
        np.fill_diagonal(m, rng.integers(3, 9, na))
    elif kind == "wide":     # past int8: the blend tree with the gathered profile
        m = rng.integers(-40, 41, (na, na)).astype(float)
        m[0, 1], m[1, 0], m[2, 2] = -200.0, 150.0, 300.0
    elif kind == "negzero":  # integral but -0: not an int8 bit pattern
        m = rng.integers(-3, 4, (na, na)).astype(float)
        np.fill_diagonal(m, 4.0)
        m[1, 2] = -0.0
    else:                    # non-integral
        m = rng.normal(size=(na, na))
        np.fill_diagonal(m, 2.0)
    return m


def _cases():
    for t in CLS:
        kinds = ["int8", "wide"] if t in ("int32", "int64") else ["int8"] if t == "int16" \
            else ["int8", "negzero", "real"]
        for kind in kinds:
            for na in (4, 6):
                yield t, kind, na


def _run(t, kind, na, gm, mode, grad, level):
    rng = np.random.default_rng(na * 31 + len(kind) + len(t) + len(mode) + len(gm))
    alpha = ALPHA[na]
    m = _matrix(kind, na, rng)
    go = 0.0 if gm == "linear" else 3.0
    p = nwgrad.AlignParams(nwgrad.SubstMatrix(m, alphabet=alpha), go, 1.0, go + 1.0, 2.0)
    n = 90
    A = ["".join(rng.choice(list(alpha), int(k))) for k in rng.integers(0, 25, n)]
    B = ["".join(rng.choice(list(alpha), int(k))) for k in rng.integers(28, 36, n)]
    out = []
    for fill, kern in (("striped", "scalar_fallback"), ("interpair", level)):
        b = CLS[t](A, B, p, gap_model=gm, mode=mode, grad_mode=grad, n_threads=2,
                   kernel=kern, traceback="pointers")
        b.fill = fill
        b.score_and_grad(keep_paths=grad == "hard")
        r = [np.asarray(b.scores()).tobytes()]
        if grad == "hard":
            r += [b[i].aligned() for i in range(n)]
            w = np.linspace(0.5, 1.5, n)
            g = b.weighted_grad(w)
            r += [np.asarray(g.matrices).tobytes(), np.asarray(g.gaps).tobytes()]
        out.append(r)
    return out


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("gm", ["affine", "linear"])
@pytest.mark.parametrize("mode", ["global", "local", "glocal"])
@pytest.mark.parametrize("grad", ["none", "hard"])
@pytest.mark.parametrize("t,kind,na", list(_cases()))
def test_lookup_forms_match_own_fill(t, kind, na, gm, mode, grad, level):
    gates = {g: None for g in _tuning.GATES if g not in ("soft_f32_min_cells", "linear_striped_min_n")}
    with _tuning.override(**gates):
        own, shared = _run(t, kind, na, gm, mode, grad, level)
    assert own == shared
