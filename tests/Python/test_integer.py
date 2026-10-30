"""The integer scalar types (SeqPairInt16/32/64, SeqPairBatchInt16/32/64, *_int16/32/64).

On integral parameters double arithmetic is exact, so the double classes are a bit-exact
ORACLE for the integer ones: every hard-path result — scores, alignments, guides,
gradients, weighted sums, banded steps, streaming align, score-only — must be identical.
Then the contracts the integer types add: integral parameters only (ValueError), the
a-priori overflow bound (OverflowError, int16 before int32), and no soft path.
"""
import numpy as np
import pytest

import nwgrad

INT_BATCHES = {16: nwgrad.SeqPairBatchInt16, 32: nwgrad.SeqPairBatchInt32, 64: nwgrad.SeqPairBatchInt64}
INT_PAIRS = {16: nwgrad.SeqPairInt16, 32: nwgrad.SeqPairInt32, 64: nwgrad.SeqPairInt64}
WIDTHS = [16, 32, 64]


def int_params(rng, alpha, affine, sets=1, lo=-4, hi=5):
    n = len(alpha)
    m = rng.integers(lo, hi + 1, (sets, n, n)).astype(float)
    for k in range(sets):
        np.fill_diagonal(m[k], rng.integers(1, 6))
    if sets == 1:
        p = nwgrad.AlignParams(nwgrad.SubstMatrix(m[0], alphabet=alpha),
                               float(rng.integers(0, 7)) if affine else 0.0, float(rng.integers(0, 4)),
                               float(rng.integers(0, 7)) if affine else 0.0, float(rng.integers(0, 4)))
        return p
    gaps = rng.integers(0, 7, (sets, 4)).astype(float)
    if not affine:
        gaps[:, 0] = 0.0
        gaps[:, 2] = 0.0
    return nwgrad.AlignParams.from_arrays(m, gaps, alphabet=alpha)


def rseq(rng, alpha, n):
    return "".join(rng.choice(list(alpha), n))


def same_params(a, b):
    return np.array_equal(a.matrices, b.matrices) and np.array_equal(a.gaps, b.gaps)


def _grad_arrays(p):
    return p.matrices.ravel(), p.gaps.ravel()


@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("gap_model", ["affine", "linear"])
@pytest.mark.parametrize("mode", ["global", "local"])
@pytest.mark.parametrize("fill", ["interpair", "striped", "rowwise"])
def test_batch_matches_double(width, gap_model, mode, fill):
    rng = np.random.default_rng(width * 7 + len(gap_model) + len(mode) + len(fill))
    alpha = rng.choice(["ACGT", nwgrad.PROTEIN.symbols])
    p = int_params(rng, alpha, gap_model == "affine")
    n = 60
    seqs_a = [rseq(rng, alpha, int(rng.integers(0, 40))) for _ in range(n)]
    seqs_b = [rseq(rng, alpha, int(rng.integers(0, 40))) for _ in range(n)]
    for tb in ["auto", "pointers", "scores", "hirschberg", "hirschberg_pmax"]:
        kw = dict(gap_model=gap_model, mode=mode, n_threads=3, traceback=tb)
        bd = nwgrad.SeqPairBatchDouble(seqs_a, seqs_b, p, **kw)
        bi = INT_BATCHES[width](seqs_a, seqs_b, p, **kw)
        for b in (bd, bi):
            b.fill = fill
            b.hb_cutoff = 4
        sd = bd.score_and_grad(keep_paths=True)
        si = bi.score_and_grad(keep_paths=True)
        assert sd == si
        assert np.array_equal(bd.scores(), bi.scores())
        # Hirschberg (and "auto" on affine: pointers at double, hirschberg_pmax on the
        # integer types) may take another optimal path at ties above its cutoff, so the
        # path-derived sums are compared on the exact modes only.
        hb = tb in ("hirschberg", "hirschberg_pmax") or (tb == "auto" and gap_model == "affine")
        w = rng.random(n)
        if not hb:
            assert same_params(bd.weighted_grad(w), bi.weighted_grad(w))
        for i in range(n):
            # exact modes: identical alignments; Hirschberg above its cutoff may pick
            # another optimal path, so compare its score and the gradient's cost
            if not hb or len(seqs_a[i]) <= 4:
                assert bd[i].aligned() == bi[i].aligned()
                assert bd[i].guide_j == bi[i].guide_j
                assert same_params(bd[i].grad, bi[i].grad)
            mi, gi4 = _grad_arrays(bi[i].grad)
            pm, pg = _grad_arrays(p)
            assert np.dot(mi, pm) + np.dot(gi4, pg) == bi[i].score
        # a banded step under new integral params (around the same guides: exact modes)
        if not hb:
            q = int_params(rng, alpha, gap_model == "affine")
            bd.set_params(q); bi.set_params(q)
            assert bd.banded_grad(5) == bi.banded_grad(5)
            assert np.array_equal(bd.scores(), bi.scores())
            assert same_params(bd.compute_grad(), bi.compute_grad())


@pytest.mark.parametrize("width", WIDTHS)
def test_tracked_batch_matches_double(width):
    rng = np.random.default_rng(width)
    alpha = "ACGT"
    for gap_model in ["affine", "linear"]:
        p = int_params(rng, alpha, gap_model == "affine", sets=3)
        n = 40
        seqs_a = [rseq(rng, alpha, int(rng.integers(1, 30))) for _ in range(n)]
        seqs_b = [rseq(rng, alpha, int(rng.integers(1, 30))) for _ in range(n)]
        tracks = [rng.integers(0, 3, len(a)).astype(np.uint16) for a in seqs_a]
        for storage in ["paths", "counts"]:
            kw = dict(gap_model=gap_model, mode="local", n_threads=2, tracks=tracks)
            bd = nwgrad.SeqPairBatchDouble(seqs_a, seqs_b, p, **kw)
            bi = INT_BATCHES[width](seqs_a, seqs_b, p, **kw)
            bd.track_storage = storage; bi.track_storage = storage
            assert bd.score_and_grad() == bi.score_and_grad()
            assert np.array_equal(bd.scores(), bi.scores())
            w = rng.random(n)
            assert same_params(bd.weighted_grad(w), bi.weighted_grad(w))


@pytest.mark.parametrize("width", WIDTHS)
def test_streaming_and_score_only_match_double(width):
    rng = np.random.default_rng(width + 100)
    alpha = nwgrad.PROTEIN.symbols
    p = int_params(rng, alpha, True)
    seqs_a = [rseq(rng, alpha, int(rng.integers(0, 50))) for _ in range(50)]
    seqs_b = [rseq(rng, alpha, int(rng.integers(0, 50))) for _ in range(50)]
    for grad_mode in ["hard", "none"]:
        for mode in ["global", "local"]:
            bd = nwgrad.SeqPairBatchDouble(n_threads=2, gap_model="affine", mode=mode, grad_mode=grad_mode)
            bi = INT_BATCHES[width](n_threads=2, gap_model="affine", mode=mode, grad_mode=grad_mode)
            rd, ri = bd.align(seqs_a, seqs_b, p), bi.align(seqs_a, seqs_b, p)
            assert np.array_equal(rd.scores, ri.scores)
            if grad_mode == "hard":
                assert same_params(rd.grad, ri.grad)
            # score-only batches
            bd2 = nwgrad.SeqPairBatchDouble(seqs_a, seqs_b, p, gap_model="affine", mode=mode, grad_mode="none")
            bi2 = INT_BATCHES[width](seqs_a, seqs_b, p, gap_model="affine", mode=mode, grad_mode="none")
            assert bd2.score_and_grad() == bi2.score_and_grad()
            assert np.array_equal(bd2.scores(), bi2.scores())


@pytest.mark.parametrize("width", WIDTHS)
def test_functions_and_pairs_match_double(width):
    rng = np.random.default_rng(width + 7)
    sfx = f"_int{width}"
    alpha = "ACGT"
    for affine in (True, False):
        p = int_params(rng, alpha, affine)
        a, b = rseq(rng, alpha, 25), rseq(rng, alpha, 31)
        names = (["nw_score_affine", "sw_score_affine", "nw_affine_grad", "sw_affine_grad"] if affine
                 else ["nw_score", "sw_score", "nw_grad", "sw_grad"])
        for name in names:
            rd = getattr(nwgrad, name + "_double")(a, b, p)
            ri = getattr(nwgrad, name + sfx)(a, b, p)
            if isinstance(rd, tuple):
                assert rd[0] == ri[0] and same_params(rd[1], ri[1])
            else:
                assert rd == ri
        gm = "affine" if affine else "linear"
        for mode in ["global", "local"]:
            sd = nwgrad.SeqPairDouble(a, b, p, gap_model=gm, mode=mode, traceback="pointers")
            si = INT_PAIRS[width](a, b, p, gap_model=gm, mode=mode, traceback="pointers")
            assert sd.align() == si.align()
            assert sd.aligned() == si.aligned()
            assert same_params(sd.grad, si.grad)
            assert sd.banded_align(3) == si.banded_align(3)
    assert not hasattr(nwgrad, "nw_soft_grad" + sfx)


def test_integral_parameters_only():
    alpha = "ACGT"
    m = np.where(np.eye(4, dtype=bool), 2.0, -1.0)
    p = nwgrad.AlignParams(nwgrad.SubstMatrix(m, alphabet=alpha), 2.0, 1.0, 2.0, 1.0)
    ok = nwgrad.SeqPairBatchInt32(["ACGT"], ["AGGT"], p)
    assert ok.score_and_grad() == nwgrad.SeqPairBatchDouble(["ACGT"], ["AGGT"], p).score_and_grad()
    q = nwgrad.AlignParams(nwgrad.SubstMatrix(m, alphabet=alpha), 2.0, 0.5, 2.0, 1.0)
    with pytest.raises(ValueError, match="integer"):
        nwgrad.SeqPairBatchInt32(["ACGT"], ["AGGT"], q)
    with pytest.raises(ValueError, match="integer"):
        ok.set_params(q)
    m2 = m.copy(); m2[1, 2] = -1.25
    q2 = nwgrad.AlignParams(nwgrad.SubstMatrix(m2, alphabet=alpha), 2.0, 1.0, 2.0, 1.0)
    with pytest.raises(ValueError, match="integer"):
        nwgrad.nw_score_affine_int16("ACGT", "AGGT", q2)
    with pytest.raises(ValueError, match="integer"):
        nwgrad.SeqPairInt64("ACGT", "AGGT", q2)
    # streaming align checks the params it is given too
    with pytest.raises(ValueError, match="integer"):
        nwgrad.SeqPairBatchInt32(n_threads=1).align(["ACGT"], ["AGGT"], q)


def test_overflow_bound_per_width():
    alpha = "ACGT"
    m = np.where(np.eye(4, dtype=bool), 100.0, -100.0)
    p = nwgrad.AlignParams(nwgrad.SubstMatrix(m, alphabet=alpha), 50.0, 10.0, 50.0, 10.0)
    a, b = "A" * 400, "C" * 400   # 400 * 100 > 32767 (int16's value range); int32 takes it
    with pytest.raises(OverflowError, match="int16"):
        nwgrad.nw_score_affine_int16(a, b, p)
    with pytest.raises(OverflowError, match="int16"):
        nwgrad.SeqPairBatchInt16([a], [b], p).score_and_grad()
    # within the value range, the old move-count rule no longer applies
    a60, b60 = "A" * 60, "C" * 60   # (60 + 60 + 66) * 100 > 8191 once refused it
    assert nwgrad.nw_score_affine_int16(a60, b60, p) == nwgrad.nw_score_affine_double(a60, b60, p)
    assert nwgrad.nw_score_affine_int32(a, b, p) == nwgrad.nw_score_affine_double(a, b, p)
    assert nwgrad.nw_score_affine_int64(a, b, p) == nwgrad.nw_score_affine_double(a, b, p)
    # a short problem fits int16
    assert nwgrad.nw_score_affine_int16("A" * 7, "C" * 7, p) == nwgrad.nw_score_affine_double("A" * 7, "C" * 7, p)
    # a parameter beyond int16's cap is refused as a parameter
    big = nwgrad.AlignParams(nwgrad.SubstMatrix(np.where(np.eye(4, dtype=bool), 9000.0, -1.0), alphabet=alpha),
                             1.0, 1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="integer"):
        nwgrad.sw_score_int16("ACGT", "ACGT", big)


def test_no_soft_path_on_integers():
    alpha = "ACGT"
    m = np.where(np.eye(4, dtype=bool), 2.0, -1.0)
    p = nwgrad.AlignParams(nwgrad.SubstMatrix(m, alphabet=alpha), 2.0, 1.0, 2.0, 1.0)
    with pytest.raises(ValueError, match="soft"):
        nwgrad.SeqPairBatchInt32(["ACGT"], ["AGGT"], p, grad_mode="soft")
    with pytest.raises(ValueError, match="soft"):
        nwgrad.SeqPairInt16("ACGT", "AGGT", p, grad_mode="soft")
    with pytest.raises(ValueError, match="soft"):
        nwgrad.SeqPairBatchInt64(n_threads=1, grad_mode="soft")
    for name in ["nw_soft_grad", "sw_soft_grad", "nw_affine_soft_grad", "sw_affine_soft_grad"]:
        for sfx in ["_int16", "_int32", "_int64"]:
            assert not hasattr(nwgrad.nwgrad_ext, name + sfx)


@pytest.mark.parametrize("width", WIDTHS)
def test_every_isa_level_is_bit_identical(width):
    rng = np.random.default_rng(width + 999)
    alpha = "ACGT"
    p = int_params(rng, alpha, True)
    seqs_a = [rseq(rng, alpha, int(rng.integers(1, 40))) for _ in range(30)]
    seqs_b = [rseq(rng, alpha, int(rng.integers(1, 40))) for _ in range(30)]
    ref = None
    for level in nwgrad.available_isa_levels():
        b = INT_BATCHES[width](seqs_a, seqs_b, p, mode="local", kernel=level, traceback="hirschberg_pmax")
        b.hb_cutoff = 3
        b.score_and_grad(keep_paths=True)
        got = (list(b.scores()), [b[i].aligned() for i in range(30)], b.compute_grad().matrices.tolist())
        if ref is None:
            ref = got
        else:
            assert got == ref, level
