"""The float32 soft path: SeqPair / SeqPairBatch run the scaled forward-backward in
float32 (soft_precision="float32", the default), the *Double classes in double.

float32 results are tolerance-equal to the double path (REL32 below; observed ~1e-6 on
pairs of up to 60 residues), not bit-equal; the shared (inter-pair) and own fills agree
the same way.  A float32 pair whose weights or lost-mass bound do not fit float32 reruns
in double by itself, before soft_impl decides anything, so soft_impl="scaled" does not
raise for it and the result is the double path's.
"""
import numpy as np
import pytest

import nwgrad
from conftest import StreamAligner, StreamAlignerDouble

REL32 = 2e-4    # float32 against double, log Z and every gradient field
ABS32 = 1e-5
RELD = 1e-9     # a float32 pair that fell back to double, against the double path
MODES = [(g, m) for g in ("linear", "affine") for m in ("global", "local")]
ALPHA = "ACGT"
PROT = nwgrad.PROTEIN.symbols


def dna_params(rng, scale=1.0, gaps=(1.0, 0.5, 1.5, 0.3)):
    M = rng.normal(0.0, scale, (4, 4))
    return nwgrad.AlignParams(nwgrad.SubstMatrix(M, ALPHA), *gaps)


def protein_params(rng, gaps=(2.0, 0.5, 3.0, 1.0)):
    M = rng.normal(-0.5, 1.0, (20, 20))
    return nwgrad.AlignParams(nwgrad.SubstMatrix(M, PROT), *gaps)


def rand_seq(rng, alpha, n):
    return "".join(rng.choice(list(alpha), n))


def pairs(rng, sym, n=30, lo=1, hi=60):
    lens = [(1, 1), (1, 7), (7, 1), (22, 50), (50, 22), (33, 33)]
    lens += [(int(rng.integers(lo, hi)), int(rng.integers(lo, hi))) for _ in range(n)]
    return [rand_seq(rng, sym, a) for a, _ in lens], [rand_seq(rng, sym, b) for _, b in lens]


def run(cls, A, B, params, gm, mode, fill="interpair", precision=None, impl="scaled",
        threads=2, guide=None, T=1.0):
    b = cls(A, B, params, n_threads=threads, gap_model=gm, mode=mode, grad_mode="soft")
    b.fill = fill
    b.soft_impl = impl
    b.soft_temperature = T
    if precision is not None:
        b.soft_precision = precision
    if guide is not None:
        b.soft_guide = guide
    b.score_and_grad()
    mats, gaps = b.grads()
    return np.asarray(b.scores()), np.asarray(mats), np.asarray(gaps), b


def close(x, y, rel=REL32, abs_=ABS32):
    np.testing.assert_allclose(x, y, rtol=rel, atol=abs_)


def test_default_precision_per_class():
    rng = np.random.default_rng(0)
    A, B = pairs(rng, ALPHA, 2)
    p = dna_params(rng)
    f = nwgrad.SeqPairBatch(A, B, p, grad_mode="soft")
    d = nwgrad.SeqPairBatchDouble(A, B, p, grad_mode="soft")
    assert f.soft_precision == "float32"
    assert d.soft_precision == "double"
    f.soft_precision = "double"
    assert f.soft_precision == "double"
    f.soft_precision = "float32"
    assert f.soft_precision == "float32"
    with pytest.raises(ValueError):
        d.soft_precision = "float32"
    with pytest.raises(ValueError):
        f.soft_precision = "single"
    assert nwgrad.SeqPair(A[0], B[0], p).soft_precision == "float32"
    assert nwgrad.SeqPairDouble(A[0], B[0], p).soft_precision == "double"


@pytest.mark.parametrize("gm,mode", MODES)
@pytest.mark.parametrize("alpha", ["ACGT", "protein"])
@pytest.mark.parametrize("fill", ["interpair", "striped"])
def test_float32_matches_double(gm, mode, alpha, fill):
    rng = np.random.default_rng(hash((gm, mode, alpha, fill)) % 2**32)
    params, sym = (dna_params(rng), ALPHA) if alpha == "ACGT" else (protein_params(rng), PROT)
    A, B = pairs(rng, sym)
    s32, m32, g32, b32 = run(nwgrad.SeqPairBatch, A, B, params, gm, mode, fill=fill)
    s64, m64, g64, _ = run(nwgrad.SeqPairBatchDouble, A, B, params, gm, mode, fill=fill)
    close(s32, s64); close(m32, m64); close(g32, g64)
    # The same batch switched to double matches the double class far more closely.
    sd, md, gd, _ = run(nwgrad.SeqPairBatch, A, B, params, gm, mode, fill=fill, precision="double")
    close(sd, s64, RELD, 1e-9); close(md, m64, RELD, 1e-9); close(gd, g64, RELD, 1e-9)
    # ... and differs from the float32 run (it really ran in float32).
    assert not np.array_equal(s32, sd)


@pytest.mark.parametrize("gm,mode", MODES)
def test_float32_inter_matches_own(gm, mode):
    rng = np.random.default_rng(hash(("inter", gm, mode)) % 2**32)
    params = dna_params(rng)
    A, B = pairs(rng, ALPHA, 40)
    si, mi, gi, bi = run(nwgrad.SeqPairBatch, A, B, params, gm, mode, fill="interpair")
    so, mo, go, _ = run(nwgrad.SeqPairBatch, A, B, params, gm, mode, fill="striped")
    if bi._inter_plan()[0] == 0:
        pytest.skip("no inter-pair kernel at this level")   # scalar_fallback
    close(si, so); close(mi, mo); close(gi, go)


@pytest.mark.parametrize("gm,mode", MODES)
def test_out_of_range_falls_back_to_double(gm, mode):
    # Step scores of ~100 nats: exp overflows float32 (max ~88 nats) but not double, so
    # the float32 pass is refused and the pair reruns in double: soft_impl="scaled" does
    # not raise, and the result is the double path's.
    rng = np.random.default_rng(hash(("range", gm, mode)) % 2**32)
    M = rng.normal(0.0, 1.0, (4, 4)) + 100.0 * np.eye(4)
    params = nwgrad.AlignParams(nwgrad.SubstMatrix(M, ALPHA), 1.0, 0.5, 1.5, 0.3)
    # Short pairs only: with 100-nat steps the double bound itself refuses a 50 x 22 pair.
    lens = [(int(rng.integers(1, 20)), int(rng.integers(1, 20))) for _ in range(12)]
    A = [rand_seq(rng, ALPHA, a) for a, _ in lens]; B = [rand_seq(rng, ALPHA, b) for _, b in lens]
    s32, m32, g32, _ = run(nwgrad.SeqPairBatch, A, B, params, gm, mode, impl="scaled")
    s64, m64, g64, _ = run(nwgrad.SeqPairBatchDouble, A, B, params, gm, mode, impl="scaled")
    close(s32, s64, RELD, 1e-9); close(m32, m64, RELD, 1e-9); close(g32, g64, RELD, 1e-9)


@pytest.mark.parametrize("gm,mode", MODES)
def test_low_temperature_in_range(gm, mode):
    rng = np.random.default_rng(hash(("temp", gm, mode)) % 2**32)
    params = dna_params(rng, scale=0.5)
    A, B = pairs(rng, ALPHA, 20)
    for T in (0.5, 0.2, 0.05):
        s32, m32, g32, _ = run(nwgrad.SeqPairBatch, A, B, params, gm, mode, impl="scaled_or_log", T=T)
        s64, m64, g64, _ = run(nwgrad.SeqPairBatchDouble, A, B, params, gm, mode, impl="scaled_or_log", T=T)
        close(s32, s64, 5e-4, 1e-5); close(m32, m64, 5e-4, 1e-5); close(g32, g64, 5e-4, 1e-5)


@pytest.mark.parametrize("gm,mode", MODES)
def test_posterior_guide_agrees(gm, mode):
    rng = np.random.default_rng(hash(("post", gm, mode)) % 2**32)
    params = dna_params(rng)
    A, B = pairs(rng, ALPHA, 20, 5, 40)
    *_, b32 = run(nwgrad.SeqPairBatch, A, B, params, gm, mode, guide="posterior")
    *_, b64 = run(nwgrad.SeqPairBatchDouble, A, B, params, gm, mode, guide="posterior")
    agree = total = 0
    for i in range(len(A)):
        g32 = np.asarray(b32[i].guide_j); g64 = np.asarray(b64[i].guide_j)
        assert g32.shape == g64.shape
        agree += int((g32 == g64).sum()); total += g32.size
    assert agree >= 0.95 * total


@pytest.mark.parametrize("gm,mode", MODES)
def test_stream_float32_matches_double(gm, mode):
    rng = np.random.default_rng(hash(("stream", gm, mode)) % 2**32)
    params = protein_params(rng)
    A, B = pairs(rng, PROT, 20)
    for fill in ("interpair", "striped"):
        s32 = StreamAligner(params, gap_model=gm, mode=mode, grad_mode="soft", n_threads=2)
        s64 = StreamAlignerDouble(params, gap_model=gm, mode=mode, grad_mode="soft", n_threads=2)
        s32._b.fill = fill; s64._b.fill = fill
        r32 = s32.align(A, B)
        r64 = s64.align(A, B)
        gaps = lambda g: np.array([g.gap_open_a, g.gap_extend_a, g.gap_open_b, g.gap_extend_b])
        close(np.asarray(r32.scores), np.asarray(r64.scores))
        close(r32.grad.matrix.to_matrix(), r64.grad.matrix.to_matrix())
        close(gaps(r32.grad), gaps(r64.grad))


def test_tracked_float32_matches_double():
    rng = np.random.default_rng(7)
    K, N = 3, 4
    for gm, mode in MODES:
        mats = rng.normal(0, 1.0, size=(K, N, N)); gaps = rng.uniform(0.3, 1.5, size=(K, 4))
        P = nwgrad.AlignParams.from_arrays(np.ascontiguousarray(mats), np.ascontiguousarray(gaps), ALPHA)
        A, B = pairs(rng, ALPHA, 12, 3, 40)
        tr = [list(rng.integers(0, K, size=len(a))) for a in A]
        gt = [list(rng.integers(0, K, size=len(a) + 1)) for a in A]
        out = {}
        for cls in (nwgrad.SeqPairBatch, nwgrad.SeqPairBatchDouble):
            for fill in ("interpair", "striped"):
                b = cls(A, B, P, n_threads=2, gap_model=gm, mode=mode, grad_mode="soft",
                        tracks=tr, gap_tracks=gt)
                b.fill = fill
                b.score_and_grad()
                m, g = b.grads()
                out[(cls, fill)] = (np.asarray(b.scores()), np.asarray(m), np.asarray(g))
        ref = out[(nwgrad.SeqPairBatchDouble, "striped")]
        for k, v in out.items():
            for x, y in zip(v, ref):
                close(x, y)
