"""The inter-pair kernels' per-lane query profile (alphabets over 8 letters), against the
pair's own fill and against the lane-cell gather it replaces.

inter_kernel_impl.inl (InterCommon::build_profile / row_profile) builds, once per group,
each lane's profile rows over the letters its A uses, and reads a DP row's scores as W
contiguous rows transposed into the column vectors; past the `inter_profile_bytes` cap
(0: always) the rows are gathered lane-cell by lane-cell as before.  Both forms load the
same matrix entries, so every lane's scores, paths and hard gradients must be
bit-identical to the scalar own fill, at every level and lane type: these fixtures give
lanes of every A length (rows past a lane's A read letter 0), ragged B lengths, B lengths
that are not a multiple of W (the profile's last block reads past n), -0 and non-integral
float scores, and wide integer scores.
"""
import numpy as np
import pytest

import nwgrad
from nwgrad import _tuning

LEVELS = [lv for lv in nwgrad.available_isa_levels() if lv != "scalar_fallback"]
CLS = {"int16": nwgrad.SeqPairBatchInt16, "int32": nwgrad.SeqPairBatchInt32,
       "int64": nwgrad.SeqPairBatchInt64, "float32": nwgrad.SeqPairBatch,
       "double": nwgrad.SeqPairBatchDouble}
AA = nwgrad.PROTEIN.symbols


def _matrix(t, rng):
    na = len(AA)
    if t in ("int16", "int32", "int64"):
        m = rng.integers(-40, 41, (na, na)).astype(float)
        np.fill_diagonal(m, rng.integers(10, 30, na))
        if t != "int16":
            m[0, 1], m[1, 0], m[2, 2] = -300.0, 250.0, 400.0
    else:
        m = rng.normal(size=(na, na))
        np.fill_diagonal(m, 2.0)
        m[1, 2] = -0.0
        m[3, 0] = 0.0
    return m


def _run(t, gm, mode, grad, level, forms, nb=(28, 36), seed=0):
    rng = np.random.default_rng(seed + len(t) * 7 + len(mode) + len(gm))
    m = _matrix(t, rng)
    go = 0.0 if gm == "linear" else 3.0
    p = nwgrad.AlignParams(nwgrad.SubstMatrix(m, alphabet=AA), go, 1.0, go + 1.0, 2.0)
    n = 90
    A = ["".join(rng.choice(list(AA), int(k))) for k in rng.integers(0, 25, n)]
    B = ["".join(rng.choice(list(AA), int(k))) for k in rng.integers(nb[0], nb[1], n)]
    out = []
    gates = {g: None for g in _tuning.GATES if g not in ("soft_f32_min_cells", "linear_striped_min_n")}
    for fill, kern, prof in forms(level):
        with _tuning.override(**gates, inter_profile_bytes=prof):
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


def _three(level):   # own fill; shared with the profile (no cap); shared with the gather
    return [("striped", "scalar_fallback", None), ("interpair", level, None),
            ("interpair", level, 0)]


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("gm", ["affine", "linear"])
@pytest.mark.parametrize("mode", ["global", "local", "glocal"])
@pytest.mark.parametrize("grad", ["none", "hard"])
@pytest.mark.parametrize("t", list(CLS))
def test_profile_matches_own_fill_and_gather(t, gm, mode, grad, level):
    own, prof, gather = _run(t, gm, mode, grad, level, _three)
    assert prof == own
    assert gather == own


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("t", ["int16", "float32"])
@pytest.mark.parametrize("nb", [(1, 3), (33, 34), (40, 41)])
def test_profile_b_lengths(t, nb, level):
    """B of one or two letters (a single profile block), and one fixed length on either
    side of a multiple of every W (no ragged pad: only the block tail past n)."""
    own, prof, gather = _run(t, "affine", "local", "hard", level, _three, nb=nb, seed=3)
    assert prof == own and gather == own


@pytest.mark.parametrize("level", LEVELS)
def test_profile_cap(level):
    """A cap below the group's profile falls back to the gather: the same results."""
    forms = lambda lv: [("striped", "scalar_fallback", None), ("interpair", lv, 64),
                        ("interpair", lv, 1 << 30)]
    own, small, big = _run("int32", "affine", "global", "hard", level, forms, seed=5)
    assert small == own and big == own


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("t", ["int16", "float32"])
@pytest.mark.parametrize("band", [0, 3])
def test_profile_banded(t, band, level):
    """GuideBanded groups (band 3; band 0 with guides from the aligned strings) read the
    profile over the whole row and compute only the lanes' spans."""
    rng = np.random.default_rng(11)
    m = _matrix(t, rng)
    p = nwgrad.AlignParams(nwgrad.SubstMatrix(m, alphabet=AA), 3.0, 1.0, 4.0, 2.0)
    n = 60
    A = ["".join(rng.choice(list(AA), int(k))) for k in rng.integers(10, 30, n)]
    B = ["".join(rng.choice(list(AA), int(k))) for k in rng.integers(25, 35, n)]
    if band == 0:   # guides: each pair's own optimal alignment
        g = CLS[t](A, B, p, gap_model="affine", mode="global", grad_mode="hard", n_threads=1)
        g.score_and_grad(keep_paths=True)
        al = [g[i].aligned() for i in range(n)]
        guides = ([x[0] for x in al], [x[1] for x in al])
    else:
        guides = ((), ())
    gates = {g: None for g in _tuning.GATES if g not in ("soft_f32_min_cells", "linear_striped_min_n")}
    out = []
    for fill, kern, prof in _three(level):
        with _tuning.override(**gates, inter_profile_bytes=prof):
            b = CLS[t](2, gap_model="affine", mode="global", grad_mode="hard")
            b.fill = fill
            r = b.align(A, B, p, band, list(guides[0]), list(guides[1]), kern)
            gr = r.grad
            out.append((np.asarray(r.scores).tobytes(), np.asarray(gr.matrix.to_matrix()).tobytes(),
                        (gr.gap_open_a, gr.gap_extend_a, gr.gap_open_b, gr.gap_extend_b)))
    assert out[1] == out[0] and out[2] == out[0]
