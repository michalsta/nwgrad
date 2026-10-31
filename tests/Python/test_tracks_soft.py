"""Tracks on the soft path (forward-backward): log Z and its gradient, per set.

The soft path is tolerance-tested (it is not bit-exact by design): against the exhaustive
oracle for Global (every path), against finite differences of log Z for every mode, the
scaled path against the log path, equal sets against the untracked library, and every
batch fill against the single-pair functions.
"""

import numpy as np
import pytest

import nwgrad

import track_oracle as orc

ALPHA = "ACGT"
N = len(ALPHA)
SOFT = {
    ("linear", "global"): "nw_soft_grad",
    ("linear", "local"): "sw_soft_grad",
    ("affine", "global"): "nw_affine_soft_grad",
    ("affine", "local"): "sw_affine_soft_grad",
}
CASES = list(SOFT)
IMPLS = ["scaled", "log"]


def to_params(mats, gaps):
    return nwgrad.AlignParams.from_arrays(np.ascontiguousarray(mats), np.ascontiguousarray(gaps), ALPHA)


def enc(s):
    return [ALPHA.index(c) for c in s]


def rnd(rng, m, n, K):
    a = "".join(rng.choice(list(ALPHA), size=m))
    b = "".join(rng.choice(list(ALPHA), size=n))
    mats = rng.normal(0, 1.0, size=(K, N, N))
    gaps = rng.uniform(0.3, 1.5, size=(K, 4))
    track = list(rng.integers(0, K, size=m))
    gtrack = list(rng.integers(0, K, size=m + 1))
    return a, b, mats, gaps, track, gtrack


def soft(case, a, b, P, track, gtrack, impl="scaled", T=1.0):
    return getattr(nwgrad, SOFT[case] + "_double")(a, b, P, soft_impl=impl, temperature=T,
                                                    track=track, gap_track=gtrack)


@pytest.mark.parametrize("gm", ["linear", "affine"])
@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("T", [1.0, 0.7])
def test_global_soft_matches_oracle(gm, impl, T):
    rng = np.random.default_rng(41)
    for _ in range(25):
        m, n, K = int(rng.integers(0, 5)), int(rng.integers(0, 5)), int(rng.integers(1, 4))
        a, b, mats, gaps, track, gtrack = rnd(rng, m, n, K)
        logz, gm_, gg = orc.soft(enc(a), enc(b), mats, gaps, track, gtrack, gm == "affine", False, T)
        s, g = soft((gm, "global"), a, b, to_params(mats, gaps), track, gtrack, impl, T)
        assert s == pytest.approx(logz, abs=1e-10)
        np.testing.assert_allclose(g.matrices, gm_, atol=1e-10)
        np.testing.assert_allclose(g.gaps, gg, atol=1e-10)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("impl", IMPLS)
def test_soft_gradient_is_the_derivative_of_log_z(case, impl):
    rng = np.random.default_rng(42)
    h = 1e-6
    for _ in range(6):
        m, n, K = int(rng.integers(2, 9)), int(rng.integers(2, 9)), int(rng.integers(2, 4))
        a, b, mats, gaps, track, gtrack = rnd(rng, m, n, K)
        _, g = soft(case, a, b, to_params(mats, gaps), track, gtrack, impl, 0.8)
        for _ in range(8):
            dm, dg = np.zeros_like(mats), np.zeros_like(gaps)
            if rng.integers(0, 2):
                dm[tuple(rng.integers(0, [K, N, N]))] = h
            else:
                dg[tuple(rng.integers(0, [K, 4]))] = h
            sp, _ = soft(case, a, b, to_params(mats + dm, gaps + dg), track, gtrack, impl, 0.8)
            sm, _ = soft(case, a, b, to_params(mats - dm, gaps - dg), track, gtrack, impl, 0.8)
            fd = (sp - sm) / (2 * h)
            an = float((g.matrices * dm).sum() + (g.gaps * dg).sum()) / h
            assert fd == pytest.approx(an, abs=1e-6, rel=1e-6)


@pytest.mark.parametrize("case", CASES)
def test_scaled_and_log_agree(case):
    rng = np.random.default_rng(43)
    for _ in range(20):
        m, n, K = int(rng.integers(0, 30)), int(rng.integers(0, 30)), int(rng.integers(1, 5))
        a, b, mats, gaps, track, gtrack = rnd(rng, m, n, K)
        P = to_params(mats, gaps)
        for T in (1.0, 0.6):
            s1, g1 = soft(case, a, b, P, track, gtrack, "scaled", T)
            s2, g2 = soft(case, a, b, P, track, gtrack, "log", T)
            assert s1 == pytest.approx(s2, rel=1e-11, abs=1e-11)
            np.testing.assert_allclose(g1.matrices, g2.matrices, rtol=1e-9, atol=1e-11)
            np.testing.assert_allclose(g1.gaps, g2.gaps, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("impl", IMPLS)
def test_equal_sets_soft_is_untracked(case, impl):
    rng = np.random.default_rng(44)
    for _ in range(20):
        m, n, K = int(rng.integers(0, 25)), int(rng.integers(0, 25)), int(rng.integers(1, 4))
        a, b, mats, gaps, track, gtrack = rnd(rng, m, n, 1)
        P1 = to_params(mats, gaps)
        PK = to_params(np.repeat(mats, K, axis=0), np.repeat(gaps, K, axis=0))
        track = list(rng.integers(0, K, size=m))
        gtrack = list(rng.integers(0, K, size=m + 1))
        s1, g1 = getattr(nwgrad, SOFT[case] + "_double")(a, b, P1, soft_impl=impl)
        sk, gk = soft(case, a, b, PK, track, gtrack, impl)
        assert sk == pytest.approx(s1, rel=1e-12, abs=1e-12)
        np.testing.assert_allclose(gk.matrices.sum(axis=0), g1.matrices[0], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(gk.gaps.sum(axis=0), g1.gaps[0], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("fill", ["interpair", "striped"])
@pytest.mark.parametrize("prec", ["_double", ""])
@pytest.mark.parametrize("guide", ["eager", "lazy", "posterior"])
def test_soft_batch_matches_per_pair(case, fill, prec, guide):
    rng = np.random.default_rng(45)
    K = 4
    A, B, tracks, gtracks = [], [], [], []
    for _ in range(90):
        m, n = int(rng.integers(0, 25)), int(rng.integers(0, 25))
        A.append("".join(rng.choice(list(ALPHA), size=m)))
        B.append("".join(rng.choice(list(ALPHA), size=n)))
        tracks.append(list(rng.integers(0, K, size=m)))
        gtracks.append(list(rng.integers(0, K, size=m + 1)))
    mats = rng.normal(0, 1.0, size=(K, N, N))
    gaps = rng.uniform(0.3, 1.5, size=(K, 4))
    P = to_params(mats, gaps)
    cls = nwgrad.SeqPairBatchDouble if prec == "_double" else nwgrad.SeqPairBatch
    # A float32 batch runs its soft pass in float32: ~1e-6 relative of the double path.
    rel, atol = (1e-11, 1e-11) if prec == "_double" else (2e-4, 1e-5)
    batch = cls(A, B, P,
                n_threads=4, gap_model=case[0], mode=case[1], grad_mode="soft", tracks=tracks, gap_tracks=gtracks)
    batch.fill = fill
    batch.soft_guide = guide
    batch.soft_temperature = 0.9
    batch.score_and_grad()
    sc = batch.scores()
    gm, gg = batch.grads()
    w = rng.normal(size=len(A))
    G = batch.weighted_grad(w)
    accm, accg = np.zeros((K, N, N)), np.zeros((K, 4))
    for k, (a, b) in enumerate(zip(A, B)):
        s, g = soft(case, a, b, P, tracks[k], gtracks[k], "scaled", 0.9)
        assert sc[k] == pytest.approx(s, rel=rel, abs=atol)
        np.testing.assert_allclose(gm[k], g.matrices, rtol=rel, atol=atol)
        np.testing.assert_allclose(gg[k], g.gaps, rtol=rel, atol=atol)
        accm += w[k] * gm[k]
        accg += w[k] * gg[k]
    np.testing.assert_allclose(G.matrices, accm, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(G.gaps, accg, rtol=1e-12, atol=1e-12)
    if guide != "lazy":
        batch.set_params(P * 1.05)
        batch.banded_grad(5)   # bands around the tracked guides
        assert np.isfinite(batch.scores()).all()
