"""BatchAligner.fill = "interpair" must be indistinguishable from fill = "striped".

Scores are bit-identical (each lane's table is the problem's own, bit for bit), and so
are hard gradients: they are sums of integer path counts, exact in any order.  Soft
results come from the inter-pair soft pass, tolerance-equal like every soft path.
"""

import numpy as np
import pytest

import nwgrad

DNA = "ACGT"
REL = 1e-11
CLS = {"double": nwgrad.BatchAlignerDouble, "float32": nwgrad.BatchAligner}


def _params(kind, gm):
    rng = np.random.default_rng(5)
    if kind == "ties":
        m = np.full((4, 4), -1.0)
        np.fill_diagonal(m, 2.0)
        go, ge = 3.0, 1.0
    else:
        m = rng.normal(size=(4, 4))
        go, ge = 2.5, 0.7
    if gm == "linear":
        go = 0.0
    return nwgrad.AlignParams(nwgrad.SubstMatrix(m, alphabet=DNA), gap_open_a=go,
                              gap_extend_a=ge, gap_open_b=go * 1.3, gap_extend_b=ge * 0.8)


def _seqs(n, lo, hi, seed):
    rng = np.random.default_rng(seed)
    return ["".join(rng.choice(list(DNA), int(l))) for l in rng.integers(lo, hi, n)]


def _align(prec, p, gm, mode, grad, A, B, fill, band=0, kernel="auto", threads=3,
           aligned=None, T=1.0):
    ba = CLS[prec](p, band=band, gap_model=gm, mode=mode, grad_mode=grad,
                   n_threads=threads, kernel=kernel)
    ba.fill = fill
    ba.soft_temperature = T
    r = ba.align(A, B, *(aligned or ()))
    g = r.grad
    return (np.asarray(r.scores), np.asarray(g.matrix.to_matrix()),
            np.array([g.gap_open_a, g.gap_extend_a, g.gap_open_b, g.gap_extend_b]))


def test_fill_property():
    ba = nwgrad.BatchAligner(_params("ties", "affine"))
    assert ba.fill == "striped"
    ba.fill = "interpair"
    assert ba.fill == "interpair"
    with pytest.raises(ValueError, match="unknown fill"):
        ba.fill = "rowwise"


@pytest.mark.parametrize("prec", ["double", "float32"])
@pytest.mark.parametrize("gm", ["affine", "linear"])
@pytest.mark.parametrize("mode", ["local", "global"])
@pytest.mark.parametrize("grad", ["hard", "none"])
@pytest.mark.parametrize("kind", ["ties", "random"])
@pytest.mark.parametrize("lengths", [(0, 30, 0, 60), (15, 30, 50, 51), (100, 700, 100, 700)])
def test_viterbi_matches(prec, gm, mode, grad, kind, lengths):
    # (100, 700): affine Global pairs past the Hirschberg cutoff run their own path.
    n = 211 if lengths[1] < 100 else 23
    A, B = _seqs(n, *lengths[:2], 1), _seqs(n, *lengths[2:], 2)
    p = _params(kind, gm)
    r0 = _align(prec, p, gm, mode, grad, A, B, "striped")
    r1 = _align(prec, p, gm, mode, grad, A, B, "interpair")
    for x, y in zip(r0, r1):
        assert np.array_equal(x, y)


@pytest.mark.parametrize("prec", ["double", "float32"])
@pytest.mark.parametrize("mode", ["local", "global"])
@pytest.mark.parametrize("band", [1, 4])
def test_banded_matches(prec, mode, band):
    A, B = _seqs(150, 10, 40, 3), _seqs(150, 30, 45, 4)
    p = _params("ties", "affine")
    r0 = _align(prec, p, "affine", mode, "hard", A, B, "striped", band=band)
    r1 = _align(prec, p, "affine", mode, "hard", A, B, "interpair", band=band)
    for x, y in zip(r0, r1):
        assert np.array_equal(x, y)


@pytest.mark.parametrize("prec", ["double", "float32"])
def test_guided_matches(prec):
    """Guides from aligned strings, band 0 (GuideBanded) and band 3."""
    A, B = _seqs(80, 10, 30, 5), _seqs(80, 33, 34, 6)
    p = _params("random", "affine")
    sp = [nwgrad.SeqPairDouble(a, b, p, gap_model="affine", mode="global") for a, b in zip(A, B)]
    al = []
    for s in sp:
        s.alloc_dp(); s.align_full(); al.append(s.aligned())
    aligned = ([x[0] for x in al], [x[1] for x in al])
    for band in (0, 3):
        r0 = _align(prec, p, "affine", "global", "hard", A, B, "striped", band=band, aligned=aligned)
        r1 = _align(prec, p, "affine", "global", "hard", A, B, "interpair", band=band, aligned=aligned)
        for x, y in zip(r0, r1):
            assert np.array_equal(x, y)


@pytest.mark.parametrize("prec", ["double", "float32"])
@pytest.mark.parametrize("gm", ["affine", "linear"])
@pytest.mark.parametrize("mode", ["local", "global"])
@pytest.mark.parametrize("T", [1.0, 0.5])
def test_soft_matches(prec, gm, mode, T):
    A, B = _seqs(157, 0, 30, 7), _seqs(157, 40, 42, 8)
    p = _params("random", gm)
    r0 = _align(prec, p, gm, mode, "soft", A, B, "striped", T=T)
    r1 = _align(prec, p, gm, mode, "soft", A, B, "interpair", T=T)
    for x, y in zip(r0, r1):
        np.testing.assert_allclose(x, y, rtol=REL, atol=REL)


@pytest.mark.parametrize("prec", ["double", "float32"])
@pytest.mark.parametrize("gm", ["affine", "linear"])
@pytest.mark.parametrize("mode", ["local", "global"])
@pytest.mark.parametrize("level", nwgrad.available_isa_levels())
def test_every_isa_level(prec, gm, mode, level):
    A, B = _seqs(97, 5, 30, 9), _seqs(97, 30, 32, 10)
    p = _params("ties", gm)
    r0 = _align(prec, p, gm, mode, "hard", A, B, "striped", kernel="scalar_fallback")
    r1 = _align(prec, p, gm, mode, "hard", A, B, "interpair", kernel=level)
    for x, y in zip(r0, r1):
        assert np.array_equal(x, y)


def test_falls_back_and_raises_like_striped():
    # protein alphabet: never shared; a bad character raises naming the pair
    AA = nwgrad.PROTEIN.symbols
    rng = np.random.default_rng(11)
    m = rng.normal(size=(20, 20))
    p = nwgrad.AlignParams(nwgrad.SubstMatrix(m, alphabet=AA), 2.0, 0.5, 2.0, 0.5)
    A = ["".join(rng.choice(list(AA), 20)) for _ in range(30)]
    B = ["".join(rng.choice(list(AA), 25)) for _ in range(30)]
    r0 = _align("double", p, "affine", "local", "hard", A, B, "striped")
    r1 = _align("double", p, "affine", "local", "hard", A, B, "interpair")
    for x, y in zip(r0, r1):
        assert np.array_equal(x, y)
    ba = nwgrad.BatchAligner(_params("ties", "affine"))
    ba.fill = "interpair"
    with pytest.raises(ValueError, match="pair 3, sequence b"):
        ba.align(["ACGT"] * 5, ["ACGT", "ACGT", "ACGT", "ACXT", "ACGT"])


@pytest.mark.parametrize("prec", ["double", "float32"])
@pytest.mark.parametrize("fill", ["striped", "interpair"])
@pytest.mark.parametrize("band", [0, 1, 3])
def test_global_banded_guide_ending_short_of_n(prec, fill, band):
    """A guide from aligned strings with trailing gaps in A stops short of column n
    (guide_j_from_aligned appends no entry for them).  A Global banded DP must still
    reach (m, n): banded around the optimal path, it returns the optimum.  It used to
    read a cell no fill had written (0.0 here, or a stale value of an earlier pair)."""
    m = np.full((4, 4), -1.0)
    np.fill_diagonal(m, 2.0)
    p = nwgrad.AlignParams(nwgrad.SubstMatrix(m, alphabet=DNA), 3.0, 1.0, 3.0, 1.0)
    a, b = "ACGTACGTAC", "ACGTACGTACGTTTGCA"
    assert nwgrad.guide_j_from_aligned("ACGTACGTAC-------", b)[-1] == 10
    A = ["TTTTTTTTTTTTTTTTT"] * 3 + [a] * 5
    sp = [nwgrad.SeqPairDouble(x, b, p, gap_model="affine", mode="global") for x in A]
    al = []
    for s in sp:
        s.alloc_dp(); s.align_full(); al.append(s.aligned())
    al[-1] = ("ACGTACGTAC-------", b)
    ba = CLS[prec](p, band=band, gap_model="affine", mode="global", n_threads=1)
    ba.fill = fill
    r = ba.align(A, [b] * len(A), [x[0] for x in al], [x[1] for x in al])
    assert r.scores[-1] == 10.0
    assert list(r.scores) == [s.score for s in sp]


@pytest.mark.parametrize("prec", ["double", "float32"])
@pytest.mark.parametrize("gm", ["affine", "linear"])
@pytest.mark.parametrize("mode", ["local", "global"])
@pytest.mark.parametrize("grad", ["hard", "soft"])
def test_ragged_b(prec, gm, mode, grad):
    """B lengths 40..50: full groups mixing B lengths (padded to the longest)."""
    A, B = _seqs(171, 8, 30, 53), _seqs(171, 40, 51, 54)
    p = _params("ties" if grad == "hard" else "random", gm)
    r0 = _align(prec, p, gm, mode, grad, A, B, "striped")
    r1 = _align(prec, p, gm, mode, grad, A, B, "interpair")
    for x, y in zip(r0, r1):
        if grad == "hard":
            assert np.array_equal(x, y)
        else:
            np.testing.assert_allclose(x, y, rtol=REL, atol=REL)
    if gm == "affine" and grad == "hard":
        r0 = _align(prec, p, gm, mode, grad, A, B, "striped", band=3)
        r1 = _align(prec, p, gm, mode, grad, A, B, "interpair", band=3)
        for x, y in zip(r0, r1):
            assert np.array_equal(x, y)


@pytest.mark.parametrize("prec", ["double", "float32"])
@pytest.mark.parametrize("gm", ["affine", "linear"])
@pytest.mark.parametrize("mode", ["local", "global"])
@pytest.mark.parametrize("grad", ["hard", "soft"])
def test_protein(prec, gm, mode, grad):
    """20 letters: the gathered per-row profile (hard) and gathered weights with per-lane
    count scatter (soft); B lengths ragged within the cap."""
    AA = nwgrad.PROTEIN.symbols
    rng = np.random.default_rng(55)
    m = rng.normal(size=(20, 20))
    go = 0.0 if gm == "linear" else 3.0
    p = nwgrad.AlignParams(nwgrad.SubstMatrix(m, alphabet=AA), go, 1.0, go * 1.3, 0.8)
    A = ["".join(rng.choice(list(AA), int(k))) for k in rng.integers(0, 40, 139)]
    B = ["".join(rng.choice(list(AA), int(k))) for k in rng.integers(30, 38, 139)]
    r0 = _align(prec, p, gm, mode, grad, A, B, "striped")
    r1 = _align(prec, p, gm, mode, grad, A, B, "interpair")
    for x, y in zip(r0, r1):
        if grad == "hard":
            assert np.array_equal(x, y)
        else:
            np.testing.assert_allclose(x, y, rtol=REL, atol=REL)


def test_long_local_float32_pairs_keep_their_hirschberg_default():
    """Local float32 affine defaults to hirschberg_pmax: pairs past the cutoff run their
    own path (and so equal fill='striped'), shorter ones join groups."""
    A, B = _seqs(12, 500, 700, 81), _seqs(12, 600, 640, 82)
    A += _seqs(40, 10, 30, 83); B += _seqs(40, 50, 51, 84)
    p = _params("random", "affine")
    r0 = _align("float32", p, "affine", "local", "hard", A, B, "striped")
    r1 = _align("float32", p, "affine", "local", "hard", A, B, "interpair")
    for x, y in zip(r0, r1):
        assert np.array_equal(x, y)
