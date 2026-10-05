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
    """fill="interpair" runs the guide Viterbi AND the scaled forward-backward W pairs
    per vector (InterSoftJob): guides and banded results match the striped fill bit for
    bit (Viterbi is bit-exact), scores and gradients to REL — including short groups,
    lanes of different len(A), and empty sequences, which take their own path."""
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
    close(out[0][0], out[1][0])
    close(out[0][1], out[1][1])
    close(out[0][2], out[1][2])
    assert out[0][3] == out[1][3]
    assert out[0][4] == out[1][4]


# ── soft temperature ──────────────────────────────────────────────────────────

def scaled_params(params, f):
    return params * f


@pytest.mark.parametrize("gm,mode", MODES)
@pytest.mark.parametrize("impl", ["scaled", "log"])
def test_temperature_is_scaled_params(gm, mode, impl):
    """soft_temperature=T returns T*log Z(params/T) and the expected counts under
    params/T: the same as running at T=1 on params/T and multiplying the score by T."""
    rng = np.random.default_rng(31)
    params = dna_params(rng, gaps=(1.0, 0.5, 1.5, 0.3))
    A = [rand_seq(rng, "ACGT", int(rng.integers(1, 30))) for _ in range(12)]
    B = [rand_seq(rng, "ACGT", int(rng.integers(1, 40))) for _ in range(12)]
    for T in (0.3, 2.5):
        b = nwgrad.SeqPairBatchDouble(n_threads=2)
        b.soft_impl = impl
        b.soft_temperature = T
        assert b.soft_temperature == T
        b.add_many(A, B, params, gap_model=gm, mode=mode, grad_mode="soft")
        b.score_and_grad()
        m1, g1 = b.grads()
        s2, m2, g2 = batch_results(A, B, scaled_params(params, 1.0 / T), gm, mode, impl)
        close(b.scores(), T * s2)
        close(m1, m2)
        close(g1, g2)


@pytest.mark.parametrize("fn,hard", [("nw_affine_soft_grad", "nw_affine_grad"),
                                     ("sw_affine_soft_grad", "sw_affine_grad"),
                                     ("nw_soft_grad", "nw_grad"),
                                     ("sw_soft_grad", "sw_grad")])
def test_temperature_low_limit(fn, hard):
    """T -> 0: T*log Z -> the Viterbi score, expected counts -> the hard counts on a
    tie-free pair.  At T = 0.02 the steps are ~150 nats, inside the scaled range."""
    rng = np.random.default_rng(41)
    M = rng.normal(0.0, 1.0, (4, 4)) + 3.0 * np.eye(4)
    params = nwgrad.AlignParams(nwgrad.SubstMatrix(M, "ACGT"), 1.3, 0.7, 1.1, 0.9)
    a, b = "ACGTTGCAAGT", "ACGTGCAAGGT"
    vs, vg = getattr(nwgrad, hard)(a, b, params)
    s, g = getattr(nwgrad, fn)(a, b, params, temperature=0.02)
    assert s == pytest.approx(vs, abs=0.02 * 3)
    np.testing.assert_allclose(g.matrix.to_matrix(), vg.matrix.to_matrix(), atol=1e-6)
    s_log, g_log = getattr(nwgrad, fn)(a, b, params, temperature=0.02, soft_impl="log")
    close(s, s_log)
    close(g.matrix.to_matrix(), g_log.matrix.to_matrix())


def test_temperature_invalidates_and_validates():
    rng = np.random.default_rng(51)
    params = dna_params(rng)
    sp = nwgrad.SeqPairDouble("ACGTAC", "ACTTAC", params, grad_mode="soft")
    sp.alloc_dp()
    sp.align_full()
    s1 = sp.score
    sp.soft_temperature = 0.5
    assert not sp.score_valid
    sp.align_full()
    assert sp.score != s1
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            sp.soft_temperature = bad


@pytest.mark.parametrize("mode", ["global", "local"])
def test_soft_interpair_mixed_groups_and_temperature(mode):
    """Groups mixing soft and hard pairs (soft lanes then take their own path), and a
    non-unit temperature through the inter-pair soft pass, against the striped fill."""
    rng = np.random.default_rng(61)
    params = dna_params(rng)
    A = [rand_seq(rng, "ACGT", int(rng.integers(10, 30))) for _ in range(40)]
    B = [rand_seq(rng, "ACGT", 40) for _ in range(40)]
    res = []
    for fill in ("striped", "interpair"):
        b = nwgrad.SeqPairBatchDouble(n_threads=2, traceback="pointers")
        b.fill = fill
        b.soft_temperature = 0.4
        b.add_many(A[:25], B[:25], params, gap_model="affine", mode=mode, grad_mode="soft")
        b.add_many(A[25:], B[25:], params, gap_model="affine", mode=mode, grad_mode="hard")
        b.score_and_grad()
        mats, gaps = b.grads()
        res.append((b.scores(), mats, gaps))
    close(res[0][0], res[1][0])
    close(res[0][1], res[1][1])
    close(res[0][2], res[1][2])


@pytest.mark.parametrize("mode", ["global", "local"])
@pytest.mark.parametrize("T", [1.0, 0.6])
def test_soft_interpair_linear(mode, T):
    """Linear soft pairs under fill="interpair" share the forward-backward (InterSoftJob,
    linear) but run their own guide Viterbi: scores/gradients to REL against the striped
    fill, guides exact; uneven len(A), short groups and empty sequences included."""
    rng = np.random.default_rng(71)
    params = dna_params(rng, gaps=(0.0, 0.8, 0.0, 1.1))
    lens = [(int(rng.integers(5, 28)), 45) for _ in range(29)] + [(0, 45), (12, 0), (20, 33)]
    A = [rand_seq(rng, "ACGT", a) for a, _ in lens]
    B = [rand_seq(rng, "ACGT", b) for _, b in lens]
    out = []
    for fill in ("striped", "interpair"):
        b = nwgrad.SeqPairBatchDouble(n_threads=3)
        b.fill = fill
        b.soft_temperature = T
        b.add_many(A, B, params, gap_model="linear", mode=mode, grad_mode="soft")
        b.score_and_grad()
        mats, gaps = b.grads()
        out.append((b.scores(), mats, gaps, [list(b[i].guide_j) for i in range(len(b))]))
    close(out[0][0], out[1][0])
    close(out[0][1], out[1][1])
    close(out[0][2], out[1][2])
    assert out[0][3] == out[1][3]


# ── soft_guide = "lazy" ───────────────────────────────────────────────────────

@pytest.mark.parametrize("gm", ["affine", "linear"])
@pytest.mark.parametrize("fill", ["striped", "interpair"])
def test_soft_guide_lazy(gm, fill):
    """Lazy skips the guide Viterbi at score time: scores and gradients are those of
    eager (to REL — interpair may route them differently); the guide, computed on first
    use, equals eager's while params are unchanged and follows set_params() after."""
    rng = np.random.default_rng(81)
    params = dna_params(rng, gaps=(1.0, 0.5, 1.5, 0.3) if gm == "affine" else (0, 0.9, 0, 1.2))
    p2 = dna_params(rng)
    A = [rand_seq(rng, "ACGT", int(rng.integers(5, 25))) for _ in range(30)]
    B = [rand_seq(rng, "ACGT", 40) for _ in range(30)]

    def batch(policy):
        b = nwgrad.SeqPairBatchDouble(n_threads=2, traceback="pointers")
        b.fill = fill
        b.soft_guide = policy
        b.add_many(A, B, params, gap_model=gm, mode="local", grad_mode="soft")
        b.score_and_grad()
        return b

    e, l = batch("eager"), batch("lazy")
    assert l.soft_guide == "lazy"
    close(e.scores(), l.scores())
    close(e.grads()[0], l.grads()[0])
    close(e.grads()[1], l.grads()[1])
    assert [list(e[i].guide_j) for i in range(30)] == [list(l[i].guide_j) for i in range(30)]
    # banded_grad on a still-pending guide resolves it first.
    l2 = batch("lazy")
    assert l2.banded_grad(3) == pytest.approx(e.banded_grad(3), rel=1e-12)
    # After set_params, a pending lazy guide follows the NEW params.
    l3 = batch("lazy")
    l3.set_params(p2)
    e2 = nwgrad.SeqPairBatchDouble(n_threads=2, traceback="pointers")
    e2.add_many(A, B, p2, gap_model=gm, mode="local", grad_mode="soft")
    e2.score_and_grad()
    assert [list(l3[i].guide_j) for i in range(30)] == [list(e2[i].guide_j) for i in range(30)]


def test_soft_guide_rejects_unknown():
    b = nwgrad.SeqPairBatchDouble()
    with pytest.raises(ValueError):
        b.soft_guide = "never"
