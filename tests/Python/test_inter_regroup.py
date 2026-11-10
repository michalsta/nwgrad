"""The inter-pair planner's tiled grouping (inter_group_tiles, simd_levels.hpp).

The sorted pairs are cut into tiles of k groups, each tile re-sorted by (len A, len B)
before its groups are cut; k = 1 is the old (len B, len A) grouping.  A lane's result
does not depend on the group it shares, so every k must give bit-identical scores, paths
and gradients, in the original pair order; only the groups (and so the speed) move.
"""
import math

import numpy as np
import pytest

import nwgrad
from nwgrad import _tuning

LEVELS = [lv for lv in nwgrad.available_isa_levels() if lv != "scalar_fallback"]
CLS = {"int16": nwgrad.SeqPairBatchInt16, "int32": nwgrad.SeqPairBatchInt32,
       "int64": nwgrad.SeqPairBatchInt64, "float32": nwgrad.SeqPairBatch,
       "double": nwgrad.SeqPairBatchDouble}
DNA = "ACGT"


@pytest.fixture(autouse=True)
def _reset():
    _tuning.reset_gates()
    yield
    _tuning.reset_gates()


def _ragged_ok(shortest, lb):
    return lb * 4 <= shortest * 5 + 16


def _reference(la, lb, W, k):
    """The planner in Python: groups of pair indices (fill gate 75 %)."""
    keys = sorted(range(len(la)), key=lambda i: (lb[i], la[i], i))
    filled = lambda r: r * 100 >= W * 75
    groups, s = [], 0
    while s < len(keys):
        t = s + 1
        while t < len(keys) and (k is None or t - s < k * W) and _ragged_ok(lb[keys[s]], lb[keys[t]]):
            t += 1
        if k != 1 and t < len(keys) and t - s > W and not filled((t - s) % W):
            t = s + (t - s) // W * W
        tile = keys[s:t] if k == 1 else sorted(keys[s:t], key=lambda i: (la[i], lb[i], i))
        groups += [tile[g:g + W] for g in range(0, len(tile), W) if filled(len(tile[g:g + W]))]
        s = t
    return groups


def _lengths(kind, n, rng):
    if kind == "independent":
        return rng.integers(5, 80, n), rng.integers(5, 80, n)
    if kind == "correlated":
        a = rng.integers(5, 80, n)
        return a, np.maximum(1, (a * rng.uniform(0.9, 1.1, n)).astype(int))
    return rng.integers(18, 26, n), np.full(n, 50)   # miRNA-like: one B length


def _batch(la, lb, rng, t="float32", gm="affine", mode="global", grad="none", level=None,
           threads=2):
    A = ["".join(rng.choice(list(DNA), int(k))) for k in la]
    B = ["".join(rng.choice(list(DNA), int(k))) for k in lb]
    m = rng.integers(-3, 4, (4, 4)).astype(float)
    np.fill_diagonal(m, 5.0)
    go = 0.0 if gm == "linear" else 3.0
    p = nwgrad.AlignParams(nwgrad.SubstMatrix(m, alphabet=DNA), go, 1.0, go + 1.0, 2.0)
    kw = {} if level is None else {"kernel": level}
    return CLS[t](A, B, p, gap_model=gm, mode=mode, grad_mode=grad, n_threads=threads, **kw)


def _util(groups, la, lb, W):
    real = sum(int(la[i]) * int(lb[i]) for g in groups for i in g)
    paid = sum(W * max(int(la[i]) for i in g) * max(int(lb[i]) for i in g) for g in groups)
    return real / paid


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("kind", ["independent", "correlated", "mirna"])
@pytest.mark.parametrize("n", [37, 600])
def test_grouping_invariants(kind, n, level):
    rng = np.random.default_rng(n + len(kind))
    la, lb = _lengths(kind, n, rng)
    b = _batch(la, lb, rng, t="int16", level=level)
    with _tuning.override(inter_group_tiles=1, inter_group_bytes=None):
        old = b._inter_groups()
    W = max(len(g) for g in old) if old else 0
    if W < 2:
        pytest.skip("no shared groups at this level")
    W = 1 << (W - 1).bit_length()     # a full group may not exist in a small batch
    auto = int(math.sqrt(n / W) + 0.5) if math.sqrt(n / W) >= 1.5 else 1
    for k, gate in ((1, 1), (1, 0), (auto, "default"), (3, 3), (None, None)):
        # The byte budget (inter_pair_fits) depends on W and the threads: lifted, so every
        # pair is eligible and the reference sees the planner's input (at 32 lanes it
        # excluded the longest pairs).
        with _tuning.override(inter_group_tiles=gate, inter_group_bytes=None):
            groups = b._inter_groups()
            elig, other, ng = b._inter_plan()
        assert ng == len(groups) and elig == sum(map(len, groups)) and elig + other == n
        flat = [i for g in groups for i in g]
        assert len(set(flat)) == len(flat)
        for g in groups:
            assert W * 3 // 4 <= len(g) <= W
            assert _ragged_ok(min(lb[i] for i in g), max(lb[i] for i in g))
        assert groups == _reference(list(la), list(lb), W, k), (k, gate)
    with _tuning.override(inter_group_tiles=1, inter_group_bytes=None):
        u_old = _util(b._inter_groups(), la, lb, W)
    with _tuning.override(inter_group_bytes=None):
        u_new = _util(b._inter_groups(), la, lb, W)
    if kind == "independent" and n == 600:
        assert u_new > u_old + 0.05, (u_old, u_new)   # the point of the tiles
    else:
        assert u_new >= u_old - 0.05, (u_old, u_new)


def _cases():
    for t in CLS:
        for grad in ("none", "hard", "soft"):
            if grad == "soft" and t not in ("float32", "double"):
                continue
            yield t, grad


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("gm", ["affine", "linear"])
@pytest.mark.parametrize("mode", ["global", "local", "glocal"])
@pytest.mark.parametrize("t,grad", list(_cases()))
def test_old_and_new_grouping_identical(t, grad, mode, gm, level):
    n = 160
    seed = len(t) * 7 + len(grad) + len(mode) * 3 + len(gm) * 5 + len(level)
    out, plans = [], []
    for gate in (1, "default", None):
        rng = np.random.default_rng(seed)
        la, lb = _lengths("independent", n, rng)
        b = _batch(la, lb, rng, t=t, gm=gm, mode=mode, grad=grad, level=level)
        with _tuning.override(inter_group_tiles=gate):
            plans.append((b._inter_groups(), b._inter_groups(score_only=True)))
            r = []
            if grad == "none":   # score only: the rows plan (plan_rows_)
                b.score_and_grad(keep_paths=False)
                r.append(np.asarray(b.scores()).tobytes())
            b.score_and_grad(keep_paths=True)
            r += [np.asarray(b.scores()).copy()] + [b[i].aligned() for i in range(n)]
            if grad != "none":
                g = b.weighted_grad(np.linspace(0.5, 1.5, n))
                r += [np.asarray(g.matrices).copy(), np.asarray(g.gaps).copy()]
        out.append(r)
    if plans[0][0] or plans[0][1]:
        assert any(p != plans[0] for p in plans[1:]), "the gate did not regroup"
    # Soft scores (log Z) and gradients follow the group in their last bits (float32 under
    # flushed denormals; double too at 4+ lanes, CI avx2 / avx512 / neon), as shared vs own
    # fills do (test_batch_interpair).  The paths, and every hard score and gradient, are
    # bit-identical.
    exact = grad != "soft"
    tol = 2e-4 if t == "float32" else 1e-7
    for o in out[1:]:
        assert len(o) == len(out[0])
        for x, y in zip(out[0], o):
            if isinstance(x, np.ndarray):
                if exact:
                    assert x.tobytes() == y.tobytes()
                else:
                    np.testing.assert_allclose(y, x, rtol=tol, atol=tol * np.abs(x).max())
            else:
                assert x == y
