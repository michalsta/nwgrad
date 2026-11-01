"""mode="glocal": global in a, local in b -- "a within b".

b's leading and trailing residues are free (left unaligned, not gap moves); every residue
of a is aligned or pays a gap.  Checked against `bruteforce`'s exhaustive enumeration
(scores, log-partition functions, hard and soft counts), against the Global and Local
scores it sits between, and across the API surfaces (convenience functions, SeqPair,
SeqPairBatch, the streaming align(), score-only, both traceback modes, a guide band).
"""

import random

import numpy as np
import pytest

import nwgrad
from conftest import StreamAligner, StreamAlignerDouble

import bruteforce as bf

ALPHABET = "ACGT"


@pytest.fixture(scope="module")
def matrix():
    return nwgrad.SubstMatrix(np.eye(4) * 2 - 1, alphabet=ALPHABET)


@pytest.fixture(scope="module")
def generic_matrix():
    rng = np.random.default_rng(20261101)
    return nwgrad.SubstMatrix(rng.normal(size=(4, 4)) * 2.0, alphabet=ALPHABET)


def params(matrix, oa, ea, ob, eb):
    return nwgrad.AlignParams(matrix, gap_open_a=oa, gap_extend_a=ea,
                              gap_open_b=ob, gap_extend_b=eb)


def _corpus(n=24, seed=5):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        a = "".join(rng.choice(ALPHABET) for _ in range(rng.randint(1, 4)))
        b = "".join(rng.choice(ALPHABET) for _ in range(rng.randint(1, 6)))
        out.append((a, b,
                    rng.choice([0.5, 2.0, 4.0]), rng.choice([0.5, 1.0, 2.0]),
                    rng.choice([0.5, 2.0, 4.0]), rng.choice([0.5, 1.0, 2.0])))
    out.append(("A", "", 1.0, 1.0, 1.0, 1.0))        # empty b: a all gapped
    out.append(("", "ACG", 1.0, 1.0, 1.0, 1.0))      # empty a: 0 (b free)
    out.append(("ACGT", "ACGT", 1.0, 1.0, 1.0, 1.0))
    return out


CORPUS = _corpus()


def _grad_arrays(grad):
    d = grad.to_dict()
    return (d["matrix"], d["gap_open_a"], d["gap_extend_a"], d["gap_open_b"], d["gap_extend_b"])


def _pairs_to_matrix(pairs):
    m = np.zeros((4, 4))
    for (x, y), v in pairs.items():
        m[ALPHABET.index(x), ALPHABET.index(y)] += v
    return m


def _optimum_is_unique(a, b, p, affine):
    scores = [s for _, s, _ in bf._glocal_scored_paths(a, b, p, affine)]
    best = max(scores)
    return sum(1 for s in scores if s > best - 1e-9) == 1


# ── Scores against the oracle ───────────────────────────────────────────────

@pytest.mark.parametrize("a,b,oa,ea,ob,eb", CORPUS)
def test_affine_score_matches_bruteforce(matrix, a, b, oa, ea, ob, eb):
    p = params(matrix, oa, ea, ob, eb)
    ref = bf.glocal_score(a, b, p, affine=True)
    assert nwgrad.glocal_score_affine(a, b, p) == pytest.approx(ref)
    assert nwgrad.glocal_score_affine_double(a, b, p) == pytest.approx(ref)


@pytest.mark.parametrize("a,b,oa,ea,ob,eb", CORPUS)
def test_linear_score_matches_bruteforce(matrix, a, b, oa, ea, ob, eb):
    p = params(matrix, 0.0, ea, 0.0, eb)
    ref = bf.glocal_score(a, b, p, affine=False)
    assert nwgrad.glocal_score(a, b, p) == pytest.approx(ref)
    assert nwgrad.glocal_score_double(a, b, p) == pytest.approx(ref)


@pytest.mark.parametrize("a,b,oa,ea,ob,eb", CORPUS)
def test_integer_scores_match(matrix, a, b, oa, ea, ob, eb):
    p = params(matrix, round(oa), round(ea), round(ob), round(eb))
    ref_aff = nwgrad.glocal_score_affine_double(a, b, p)
    ref_lin = nwgrad.glocal_score_double(a, b, p)
    for sfx in ("int16", "int32", "int64"):
        assert getattr(nwgrad, "glocal_score_affine_" + sfx)(a, b, p) == ref_aff
        assert getattr(nwgrad, "glocal_score_" + sfx)(a, b, p) == ref_lin


@pytest.mark.parametrize("a,b,oa,ea,ob,eb", CORPUS)
def test_between_global_and_local(matrix, a, b, oa, ea, ob, eb):
    # Non-negative gap costs: a global alignment is a glocal one (nothing free), a glocal
    # one is a local one (b's free ends, a fully aligned).
    p = params(matrix, oa, ea, ob, eb)
    g = nwgrad.glocal_score_affine_double(a, b, p)
    assert g >= nwgrad.nw_score_affine_double(a, b, p) - 1e-12
    assert g <= nwgrad.sw_score_affine_double(a, b, p) + 1e-12


def test_a_within_b(matrix):
    # a occurs inside b: the glocal score is a's self-score, b's flanks are free.
    p = params(matrix, 2.0, 1.0, 2.0, 1.0)
    assert nwgrad.glocal_score_affine("ACGT", "TTTACGTGGG", p) == pytest.approx(4.0)   # match = +1
    assert nwgrad.nw_score_affine("ACGT", "TTTACGTGGG", p) < 4.0
    # the other way round, a's flanks are charged
    assert nwgrad.glocal_score_affine("TTTACGTGGG", "ACGT", p) < 4.0


# ── The soft path against the oracle ────────────────────────────────────────

@pytest.mark.parametrize("a,b,oa,ea,ob,eb", CORPUS)
def test_affine_log_z_matches_bruteforce(generic_matrix, a, b, oa, ea, ob, eb):
    p = params(generic_matrix, oa, ea, ob, eb)
    ref = bf.glocal_log_z(a, b, p, affine=True)
    for impl in ("log", "scaled_or_log"):
        log_z, _ = nwgrad.glocal_affine_soft_grad_double(a, b, p, soft_impl=impl)
        assert log_z == pytest.approx(ref, rel=1e-9, abs=1e-9)
    log_z, _ = nwgrad.glocal_affine_soft_grad(a, b, p)
    assert log_z == pytest.approx(ref, rel=1e-5, abs=1e-5)


@pytest.mark.parametrize("a,b,oa,ea,ob,eb", CORPUS)
def test_linear_log_z_matches_bruteforce(generic_matrix, a, b, oa, ea, ob, eb):
    p = params(generic_matrix, 0.0, ea, 0.0, eb)
    ref = bf.glocal_log_z(a, b, p, affine=False)
    log_z, _ = nwgrad.glocal_soft_grad_double(a, b, p, soft_impl="log")
    assert log_z == pytest.approx(ref, rel=1e-9, abs=1e-9)


@pytest.mark.parametrize("a,b,oa,ea,ob,eb", CORPUS)
def test_soft_counts_match_bruteforce(generic_matrix, a, b, oa, ea, ob, eb):
    p = params(generic_matrix, oa, ea, ob, eb)
    _, grad = nwgrad.glocal_affine_soft_grad_double(a, b, p)
    got_matrix, got_oa, got_ea, got_ob, got_eb = _grad_arrays(grad)
    pairs, exp_oa, exp_ea, exp_ob, exp_eb = bf.glocal_soft_counts(a, b, p, affine=True)
    np.testing.assert_allclose(got_matrix, _pairs_to_matrix(pairs), atol=1e-9)
    assert (got_oa, got_ea, got_ob, got_eb) == pytest.approx((-exp_oa, -exp_ea, -exp_ob, -exp_eb), abs=1e-9)


@pytest.mark.parametrize("a,b,oa,ea,ob,eb", CORPUS)
def test_linear_soft_counts_match_bruteforce(generic_matrix, a, b, oa, ea, ob, eb):
    p = params(generic_matrix, 0.0, ea, 0.0, eb)
    _, grad = nwgrad.glocal_soft_grad_double(a, b, p)
    got_matrix, _, got_ea, _, got_eb = _grad_arrays(grad)
    pairs, _, exp_ea, _, exp_eb = bf.glocal_soft_counts(a, b, p, affine=False)
    np.testing.assert_allclose(got_matrix, _pairs_to_matrix(pairs), atol=1e-9)
    assert (got_ea, got_eb) == pytest.approx((-exp_ea, -exp_eb), abs=1e-9)


# ── Hard gradients ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("a,b,oa,ea,ob,eb", CORPUS)
def test_hard_counts_match_bruteforce(generic_matrix, a, b, oa, ea, ob, eb):
    p = params(generic_matrix, oa, ea, ob, eb)
    if not _optimum_is_unique(a, b, p, affine=True):
        pytest.skip("optimal alignment is not unique")
    score, grad = nwgrad.glocal_affine_grad_double(a, b, p)
    got_matrix, got_oa, got_ea, got_ob, got_eb = _grad_arrays(grad)
    pairs, exp_oa, exp_ea, exp_ob, exp_eb = bf.glocal_hard_counts(a, b, p, affine=True)
    np.testing.assert_allclose(got_matrix, _pairs_to_matrix(pairs), atol=1e-9)
    assert (got_oa, got_ea, got_ob, got_eb) == pytest.approx((-exp_oa, -exp_ea, -exp_ob, -exp_eb))
    assert score == pytest.approx(bf.glocal_score(a, b, p, affine=True))


@pytest.mark.parametrize("a,b,oa,ea,ob,eb", CORPUS)
def test_hard_gradient_dot_params_is_the_score(generic_matrix, a, b, oa, ea, ob, eb):
    # the score is linear in the parameters: grad . params == score (any tie)
    p = params(generic_matrix, oa, ea, ob, eb)
    for fn, lin in ((nwgrad.glocal_affine_grad_double, False), (nwgrad.glocal_grad_double, True)):
        pp = params(generic_matrix, 0.0, ea, 0.0, eb) if lin else p
        score, grad = fn(a, b, pp)
        d, g = pp.to_dict(), grad.to_dict()
        dot = float(np.sum(np.asarray(g["matrix"]) * np.asarray(d["matrix"])))
        for k in ("gap_open_a", "gap_extend_a", "gap_open_b", "gap_extend_b"):
            dot += g[k] * d[k]   # the gap gradients are the negated counts
        assert dot == pytest.approx(score, abs=1e-9)


# ── The classes: pairs, batches, streaming, traceback modes, a band ─────────

def _aligned_score(sa, sb, p, affine):
    """Score a gapped pair of strings as glocal scores them: b's flanks free."""
    assert len(sa) == len(sb)
    lo, hi = 0, len(sa)
    while lo < hi and sa[lo] == "-": lo += 1
    while hi > lo and sa[hi - 1] == "-": hi -= 1
    # a trailing gap run in a (b residues after a ends) is charged; the oracle's path
    # scorer charges every move, so hand it the trimmed strings
    sa, sb = sa[lo:hi], sb[lo:hi]
    d = p.to_dict()
    path = "".join("D" if x != "-" and y != "-" else ("X" if y == "-" else "Y") for x, y in zip(sa, sb))
    a = sa.replace("-", ""); b = sb.replace("-", "")
    return bf.path_score(path, a, b, p.matrix, d["gap_open_a"], d["gap_extend_a"],
                         d["gap_open_b"], d["gap_extend_b"], affine)


@pytest.mark.parametrize("gap_model", ["affine", "linear"])
@pytest.mark.parametrize("traceback", ["pointers", "scores"])
def test_pair_alignment_scores_its_score(generic_matrix, gap_model, traceback):
    rng = random.Random(3)
    p = params(generic_matrix, 2.0, 1.0, 1.5, 0.5)
    for _ in range(40):
        a = "".join(rng.choice(ALPHABET) for _ in range(rng.randint(1, 9)))
        b = "".join(rng.choice(ALPHABET) for _ in range(rng.randint(1, 14)))
        sp = nwgrad.SeqPairDouble(a, b, p, gap_model=gap_model, mode="glocal", traceback=traceback)
        score = sp.align()
        sa, sb = sp.aligned()
        assert sa.replace("-", "") == a                      # every residue of a
        assert sb.replace("-", "") in b                      # a window of b
        assert _aligned_score(sa, sb, p, gap_model == "affine") == pytest.approx(score)
        ref = (nwgrad.glocal_score_affine_double if gap_model == "affine" else nwgrad.glocal_score_double)(a, b, p)
        assert score == pytest.approx(ref)
        assert sp.mode == "glocal"


def test_pointers_and_scores_agree(generic_matrix):
    rng = random.Random(9)
    p = params(generic_matrix, 2.0, 1.0, 1.5, 0.5)
    for _ in range(30):
        a = "".join(rng.choice(ALPHABET) for _ in range(rng.randint(1, 12)))
        b = "".join(rng.choice(ALPHABET) for _ in range(rng.randint(1, 20)))
        r = []
        for tb in ("pointers", "scores"):
            sp = nwgrad.SeqPair(a, b, p, mode="glocal", traceback=tb)
            s, g = sp.score_and_grad()
            r.append((s, sp.aligned(), g.to_dict()))
        assert r[0][0] == r[1][0] and r[0][1] == r[1][1]
        assert np.array_equal(np.asarray(r[0][2]["matrix"]), np.asarray(r[1][2]["matrix"]))


def test_hirschberg_is_refused(matrix):
    p = params(matrix, 1.0, 1.0, 1.0, 1.0)
    # long enough to split above any default hb_cutoff (a shorter pair runs AS pointers)
    sp = nwgrad.SeqPair("ACGTACGT" * 90, "TTACGTACGTTT" * 90, p, mode="glocal", traceback="hirschberg")
    with pytest.raises(Exception, match="glocal"):
        sp.align()


@pytest.mark.parametrize("gap_model", ["affine", "linear"])
@pytest.mark.parametrize("grad_mode", ["hard", "soft", "none"])
def test_batch_matches_pairs(generic_matrix, gap_model, grad_mode):
    rng = random.Random(21)
    p = params(generic_matrix, 2.0, 1.0, 1.5, 0.5)
    A = ["".join(rng.choice(ALPHABET) for _ in range(rng.randint(1, 10))) for _ in range(40)]
    B = ["".join(rng.choice(ALPHABET) for _ in range(rng.randint(1, 16))) for _ in range(40)]
    batch = nwgrad.SeqPairBatchDouble(A, B, p, gap_model=gap_model, mode="glocal", grad_mode=grad_mode, n_threads=3)
    assert batch.mode == "glocal"
    total = batch.score_and_grad()
    scores = batch.scores()
    assert total == pytest.approx(float(np.sum(scores)))
    grad = None
    if grad_mode != "none":
        for i in range(len(A)):
            grad = batch[i].grad if grad is None else grad + batch[i].grad
    single = []
    gsum = None
    for a, b in zip(A, B):
        if grad_mode == "soft":
            fn = nwgrad.glocal_affine_soft_grad_double if gap_model == "affine" else nwgrad.glocal_soft_grad_double
        elif grad_mode == "hard":
            fn = nwgrad.glocal_affine_grad_double if gap_model == "affine" else nwgrad.glocal_grad_double
        else:
            fn = None
        if fn is None:
            s = (nwgrad.glocal_score_affine_double if gap_model == "affine" else nwgrad.glocal_score_double)(a, b, p)
        else:
            s, g = fn(a, b, p)
            gsum = g if gsum is None else gsum + g
        single.append(s)
    np.testing.assert_allclose(np.asarray(scores), np.asarray(single), rtol=1e-9, atol=1e-9)
    if grad is not None:
        got, exp = grad.to_dict(), gsum.to_dict()
        np.testing.assert_allclose(np.asarray(got["matrix"]), np.asarray(exp["matrix"]), rtol=1e-8, atol=1e-8)
        for k in ("gap_open_a", "gap_extend_a", "gap_open_b", "gap_extend_b"):
            assert got[k] == pytest.approx(exp[k], rel=1e-8, abs=1e-8)


@pytest.mark.parametrize("cls", [StreamAligner, StreamAlignerDouble])
@pytest.mark.parametrize("grad_mode", ["hard", "soft"])
def test_streaming_matches_pairs(generic_matrix, cls, grad_mode):
    rng = random.Random(33)
    p = params(generic_matrix, 2.0, 1.0, 1.5, 0.5)
    A = ["".join(rng.choice(ALPHABET) for _ in range(rng.randint(1, 10))) for _ in range(30)]
    B = ["".join(rng.choice(ALPHABET) for _ in range(rng.randint(1, 16))) for _ in range(30)]
    res = cls(p, mode="glocal", grad_mode=grad_mode, n_threads=2).align(A, B)
    fn = nwgrad.glocal_affine_soft_grad_double if grad_mode == "soft" else nwgrad.glocal_affine_grad_double
    exp = [fn(a, b, p)[0] for a, b in zip(A, B)]
    tol = 1e-4 if cls is StreamAligner else 1e-9
    np.testing.assert_allclose(np.asarray(res.scores), np.asarray(exp), rtol=tol, atol=tol)


def test_band_around_the_alignment(generic_matrix):
    # A guide band wide enough to hold the optimum gives the full score.
    p = params(generic_matrix, 2.0, 1.0, 1.5, 0.5)
    a, b = "ACGTTGCA", "GGACGTTGCATT"
    sp = nwgrad.SeqPairDouble(a, b, p, mode="glocal")
    full = sp.align()
    sa, sb = sp.aligned()
    banded = nwgrad.glocal_score_affine_double(a, b, p, band=3, aligned_a=sa, aligned_b=sb)
    assert banded == pytest.approx(full)
    s_hard, _ = nwgrad.glocal_affine_grad_double(a, b, p, band=3, aligned_a=sa, aligned_b=sb)
    assert s_hard == pytest.approx(full)


def test_mode_name_rejected_elsewhere():
    with pytest.raises(ValueError, match="glocal"):
        nwgrad.SeqPairBatch(n_threads=1, mode="glocla")


# ── The score-only kernels against the scalar fill, bit for bit ─────────────

@pytest.mark.parametrize("cls,fn_aff,fn_lin", [
    (nwgrad.SeqPairDouble, nwgrad.glocal_score_affine_double, nwgrad.glocal_score_double),
    (nwgrad.SeqPair, nwgrad.glocal_score_affine, nwgrad.glocal_score),
])
def test_score_only_equals_the_pointer_fill(generic_matrix, cls, fn_aff, fn_lin):
    # the score-only kernel of the running backend vs the traceback fill: the exact
    # optimum, bit-identical (as test_score_only pins it for the other modes)
    rng = random.Random(77)
    for oa, ea, ob, eb in ((2.0, 1.0, 1.5, 0.5), (0.0, 0.5, 0.0, 1.0), (3.0, 0.0, 1.0, 0.0)):
        p = params(generic_matrix, oa, ea, ob, eb)
        for _ in range(25):
            a = "".join(rng.choice(ALPHABET) for _ in range(rng.randint(0, 40)))
            b = "".join(rng.choice(ALPHABET) for _ in range(rng.randint(0, 70)))
            assert fn_aff(a, b, p) == cls(a, b, p, mode="glocal", traceback="pointers").align()
            assert fn_lin(a, b, p) == cls(a, b, p, gap_model="linear", mode="glocal").align()
