"""The scaled-probability soft path (soft_impl="scaled", the default) against the
log-space one (soft_impl="log", the original recurrences) it replaced.

The soft path is not bit-exact across implementations: log Z and every gradient
field must agree to REL (relative, with an absolute floor ABS for fields near zero).
Both are double precision end to end; the observed agreement is ~1e-14, and REL
leaves two orders of headroom for longer pairs and other compilers/ISAs.
"""

import numpy as np
import pytest
import nwgrad

REL = 1e-11
ABS = 1e-11

MODES = [(g, m) for g in ("linear", "affine") for m in ("global", "local")]


def dna_params(rng, scale=1.0, gaps=(1.0, 0.5, 1.5, 0.3)):
    M = rng.normal(0.0, scale, (4, 4))
    return nwgrad.AlignParams(nwgrad.SubstMatrix(M, "ACGT"), *gaps)


def protein_params(rng, gaps=(2.0, 0.5, 3.0, 1.0)):
    M = rng.normal(-0.5, 1.0, (20, 20))
    return nwgrad.AlignParams(nwgrad.SubstMatrix(M, nwgrad.PROTEIN.symbols), *gaps)


def rand_seq(rng, alpha, n):
    return "".join(rng.choice(list(alpha), n))


def batch_results(A, B, params, gm, mode, impl, threads=2):
    b = nwgrad.SeqPairBatchDouble(n_threads=threads)
    b.soft_impl = impl
    b.add_many(A, B, params, gap_model=gm, mode=mode, grad_mode="soft")
    b.score_and_grad()
    mats, gaps = b.grads()
    return b.scores(), mats, gaps


def close(x, y):
    np.testing.assert_allclose(x, y, rtol=REL, atol=ABS)


@pytest.mark.parametrize("gm,mode", MODES)
@pytest.mark.parametrize("alpha", ["ACGT", "protein"])
def test_scaled_matches_log(gm, mode, alpha):
    rng = np.random.default_rng(hash((gm, mode, alpha)) % 2**32)
    if alpha == "ACGT":
        params, sym = dna_params(rng), "ACGT"
    else:
        params, sym = protein_params(rng), nwgrad.PROTEIN.symbols
    # Lengths include the empty and length-1 edge cases on either side.
    lens = [(0, 0), (0, 3), (3, 0), (1, 1), (1, 7), (7, 1), (22, 50), (50, 22), (33, 33)]
    lens += [(int(rng.integers(1, 60)), int(rng.integers(1, 60))) for _ in range(30)]
    A = [rand_seq(rng, sym, a) for a, _ in lens]
    B = [rand_seq(rng, sym, b) for _, b in lens]
    s1, m1, g1 = batch_results(A, B, params, gm, mode, "scaled")
    s2, m2, g2 = batch_results(A, B, params, gm, mode, "log")
    close(s1, s2)
    close(m1, m2)
    close(g1, g2)


@pytest.mark.parametrize("gm,mode", MODES)
@pytest.mark.parametrize("bw", [0, 1, 3, 8])
def test_scaled_matches_log_banded(gm, mode, bw):
    """realign_banded runs the GuideBanded aligner — band edges, guard cells, and
    (local) borders that lie outside the band."""
    rng = np.random.default_rng(7 + bw)
    params = dna_params(rng)
    for _ in range(15):
        a = rand_seq(rng, "ACGT", int(rng.integers(1, 40)))
        b = rand_seq(rng, "ACGT", int(rng.integers(1, 40)))
        out = []
        for impl in ("scaled", "log"):
            sp = nwgrad.SeqPairDouble(a, b, params, gap_model=gm, mode=mode, grad_mode="soft")
            sp.soft_impl = impl
            sp.alloc_dp()
            sp.align_full()
            sp.realign_banded(bw)
            sp.compute_grad()
            g = sp.grad
            out.append((sp.score, g.matrix.to_matrix(), [g.gap_open_a, g.gap_extend_a,
                                                       g.gap_open_b, g.gap_extend_b]))
        close(out[0][0], out[1][0])
        close(out[0][1], out[1][1])
        close(out[0][2], out[1][2])


def test_batch_aligner_soft_impl():
    rng = np.random.default_rng(3)
    params = dna_params(rng)
    A = [rand_seq(rng, "ACGT", 20) for _ in range(20)]
    B = [rand_seq(rng, "ACGT", 45) for _ in range(20)]
    res = []
    for impl in ("scaled", "log"):
        ba = nwgrad.BatchAlignerDouble(params, gap_model="affine", mode="local",
                                       grad_mode="soft", n_threads=2)
        ba.soft_impl = impl
        assert ba.soft_impl == impl
        r = ba.align(A, B)
        res.append((np.array(r.scores), r.grad.matrix.to_matrix(), r.grad.gap_open_a))
    close(res[0][0], res[1][0])
    close(res[0][1], res[1][1])
    close(res[0][2], res[1][2])


def out_of_range_case():
    """A step score of 800 nats: exp(800) does not fit a double, so the scaled path
    cannot even form its weights (the low-temperature extreme)."""
    rng = np.random.default_rng(11)
    M = np.full((20, 20), -1.0)
    np.fill_diagonal(M, 800.0)
    params = nwgrad.AlignParams(nwgrad.SubstMatrix(M, nwgrad.PROTEIN.symbols), 2.0, 0.5, 2.0, 0.5)
    s = rand_seq(rng, nwgrad.PROTEIN.symbols, 50)
    return s, params


def test_scaled_long_local_protein_in_range():
    """A local self-pair with log Z ~ 1600 nats: the free starts of late rows sit ~2300
    binary orders below their row max and are lost, but their posteriors are
    negligible, and the loss bound must see that rather than refuse."""
    rng = np.random.default_rng(12)
    M = rng.normal(-1.0, 0.5, (20, 20))
    np.fill_diagonal(M, 8.0)
    params = nwgrad.AlignParams(nwgrad.SubstMatrix(M, nwgrad.PROTEIN.symbols), 2.0, 0.5, 2.0, 0.5)
    s = rand_seq(rng, nwgrad.PROTEIN.symbols, 200)
    t = s[:120] + rand_seq(rng, nwgrad.PROTEIN.symbols, 10) + s[130:]
    s1, m1, g1 = batch_results([s, t], [s, s], params, "affine", "local", "scaled")
    s2, m2, g2 = batch_results([s, t], [s, s], params, "affine", "local", "log")
    assert s1[0] > 1000
    close(s1, s2)
    close(m1, m2)
    close(g1, g2)


def test_scaled_raises_out_of_range():
    s, params = out_of_range_case()
    b = nwgrad.SeqPairBatchDouble(n_threads=1)
    b.add_many([s], [s], params, gap_model="affine", mode="local", grad_mode="soft")
    assert b.soft_impl == "scaled"
    with pytest.raises(ValueError, match="scaled_or_log"):
        b.score_and_grad()


def test_scaled_or_log_falls_back():
    s, params = out_of_range_case()
    rng = np.random.default_rng(5)
    t = rand_seq(rng, nwgrad.PROTEIN.symbols, 30)
    A, B = [s, t], [s, t]          # one pair out of range, one in range
    s1, m1, g1 = batch_results(A, B, params, "affine", "local", "scaled_or_log")
    s2, m2, g2 = batch_results(A, B, params, "affine", "local", "log")
    close(s1, s2)
    close(m1, m2)
    close(g1, g2)
    assert s1[0] > 1000   # the pair that needed the fallback really is out of range


def test_soft_impl_rejects_unknown():
    b = nwgrad.SeqPairBatchDouble()
    with pytest.raises(ValueError):
        b.soft_impl = "fast"


@pytest.mark.parametrize("mode", ["global", "local"])
def test_soft_interpair_identical(mode):
    """fill="interpair" shares only the guide Viterbi across lanes; forward-backward
    is the pair's own, so scores, gradients and guides match the striped fill bit for
    bit — including short groups and empty sequences, which take their own fill."""
    rng = np.random.default_rng(21)
    params = dna_params(rng)
    lens = [(int(rng.integers(15, 25)), 50) for _ in range(37)] + [(0, 50), (20, 0), (22, 31)]
    A = [rand_seq(rng, "ACGT", a) for a, _ in lens]
    B = [rand_seq(rng, "ACGT", b) for _, b in lens]
    out = []
    for fill in ("striped", "interpair"):
        b = nwgrad.SeqPairBatchDouble(n_threads=3, traceback="pointers")
        b.fill = fill
        b.add_many(A, B, params, gap_model="affine", mode=mode, grad_mode="soft")
        b.score_and_grad()
        mats, gaps = b.grads()
        guides = [list(b[i].guide_j) for i in range(len(b))]
        banded = b.banded_grad(3)
        out.append((b.scores(), mats, gaps, guides, banded))
    np.testing.assert_array_equal(out[0][0], out[1][0])
    np.testing.assert_array_equal(out[0][1], out[1][1])
    np.testing.assert_array_equal(out[0][2], out[1][2])
    assert out[0][3] == out[1][3]
    assert out[0][4] == out[1][4]
