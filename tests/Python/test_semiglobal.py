"""The semi-global modes: mode="semiglobal" with free_ends=(a_begin, a_end, b_begin, b_end)
(or the names of the free ends), and mode="overlap" (all four free).

A free flank's residues are left unaligned at no cost: a free A begin makes column 0 a free
start, a free B begin row 0 (glocal's), a free A end makes every cell of column n an end, a
free B end every cell of row m (glocal's).  Among equal optima the end is the first
candidate cell in row-major order (A's index, then B's), the state by M > X > Y.

Checked against a reference DP written here (scores, end cells under the tie rule, exact
with integer-valued parameters), against Biopython's PairwiseAligner with zero end-gap
scores (an independent statement of the same semantics), and bit for bit: every level
against the scalar backend, the score-only kernels against the traceback fill, the empty
flag set against mode="global", glocal's flags against mode="glocal".
"""

import itertools
import random

import numpy as np
import pytest

import nwgrad

ALPHABET = "ACGT"
FLAGS = list(itertools.product([False, True], repeat=4))   # (a_begin, a_end, b_begin, b_end)
# named sets: the modes that fix their ends, and free_ends given by name
NAMED = {
    "overlap": (True, True, True, True),
    "glocal": (False, False, True, True),
    "global": (False, False, False, False),
}
NINF = float("-inf")


def _params(matrix, oa, ea, ob, eb):
    return nwgrad.AlignParams(matrix, gap_open_a=oa, gap_extend_a=ea, gap_open_b=ob, gap_extend_b=eb)


@pytest.fixture(scope="module")
def imatrix():
    # integer-valued: every score is exact in double, so ties are real ties
    rng = np.random.default_rng(20261107)
    return nwgrad.SubstMatrix(rng.integers(-4, 5, size=(4, 4)).astype(float), alphabet=ALPHABET)


@pytest.fixture(scope="module")
def gmatrix():
    rng = np.random.default_rng(1107)
    return nwgrad.SubstMatrix(rng.normal(size=(4, 4)) * 2.0, alphabet=ALPHABET)


def _rand(rng, lo, hi):
    return "".join(rng.choice(ALPHABET) for _ in range(rng.randint(lo, hi)))


def _level():
    import conftest
    lv = conftest._current_backend
    return lv if lv not in (None, "auto", "simd") else nwgrad.get_isa_level()


# ── The reference: the 4-flag DP, three states, in Python ───────────────────

def sg_reference(a, b, M, oa, ea, ob, eb, flags, affine=True):
    """(score, (i, j), state) of the best semi-global alignment, the end by the documented
    rule: the first candidate in row-major order holding the maximum, M > X > Y.  Linear
    gaps: the affine recurrence with zero opens (its cell maxima are the linear H)."""
    ab, ae, bb, be = flags
    if not affine:
        oa = ob = 0.0
    m, n = len(a), len(b)
    ia = [ALPHABET.index(c) for c in a]
    ib = [ALPHABET.index(c) for c in b]
    VM = [[NINF] * (n + 1) for _ in range(m + 1)]
    VX = [[NINF] * (n + 1) for _ in range(m + 1)]
    VY = [[NINF] * (n + 1) for _ in range(m + 1)]
    VM[0][0] = 0.0
    for j in range(1, n + 1):
        if bb: VM[0][j] = 0.0
        else:  VY[0][j] = -(oa + j * ea)
    for i in range(1, m + 1):
        if ab: VM[i][0] = 0.0
        else:  VX[i][0] = -(ob + i * eb)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            VM[i][j] = max(VM[i-1][j-1], VX[i-1][j-1], VY[i-1][j-1]) + M[ia[i-1]][ib[j-1]]
            VX[i][j] = max((VM[i-1][j] - ob) - eb, VX[i-1][j] - eb, (VY[i-1][j] - ob) - eb)
            VY[i][j] = max((VM[i][j-1] - oa) - ea, (VX[i][j-1] - oa) - ea, VY[i][j-1] - ea)
    best = None
    for i in range(m + 1):
        for j in range(n + 1):
            cand = (i == m and j == n) or (i == m and be) or (j == n and ae)
            if not cand:
                continue
            vm, vx, vy = VM[i][j], VX[i][j], VY[i][j]
            v = max(vm, vx, vy)
            if best is None or v > best[0]:
                st = "M" if (vm >= vx and vm >= vy) else ("X" if vx >= vy else "Y")
                best = (v, (i, j), st)
    return best


def _path_score(sa, sb, M, oa, ea, ob, eb, affine):
    """The score of a gapped pair of strings (no flank trimming: aligned() holds the path)."""
    s, run = 0.0, None
    for x, y in zip(sa, sb):
        if x != "-" and y != "-":
            s += M[ALPHABET.index(x)][ALPHABET.index(y)]; run = None
        elif y == "-":      # gap in B, consumes A
            s -= eb + (ob if affine and run != "X" else 0.0); run = "X"
        else:               # gap in A, consumes B
            s -= ea + (oa if affine and run != "Y" else 0.0); run = "Y"
    return s


def _corpus(seed, n, la=(0, 7), lb=(0, 9)):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        out.append((_rand(rng, *la), _rand(rng, *lb),
                    rng.choice([0.0, 1.0, 3.0]), rng.choice([0.0, 1.0, 2.0]),
                    rng.choice([0.0, 1.0, 3.0]), rng.choice([0.0, 1.0, 2.0])))
    out += [("", "", 1.0, 1.0, 1.0, 1.0), ("ACG", "", 1.0, 1.0, 1.0, 1.0),
            ("", "ACG", 1.0, 1.0, 1.0, 1.0), ("ACGT", "ACGT", 1.0, 1.0, 1.0, 1.0)]
    return out


# ── The reference itself, against an independent statement of the semantics ──

@pytest.mark.filterwarnings("ignore:The attribute")
def test_reference_matches_biopython(gmatrix):
    Align = pytest.importorskip("Bio.Align")
    M = np.asarray(gmatrix.to_matrix())
    rng = random.Random(11)
    for _ in range(60):
        a, b = _rand(rng, 1, 8), _rand(rng, 1, 10)
        oa, ea, ob, eb = (rng.choice([0.5, 2.0, 3.0]), rng.choice([0.5, 1.0]),
                          rng.choice([0.5, 2.0, 3.0]), rng.choice([0.5, 1.0]))
        for flags in FLAGS:
            ab, ae, bb, be = flags
            al = Align.PairwiseAligner()
            al.mode = "global"
            al.substitution_matrix = Align.substitution_matrices.Array(alphabet=ALPHABET, dims=2, data=M)
            # target = A, query = B.  A gap in A (consumes B) is a target gap: gap_a costs.
            al.target_internal_open_gap_score = -(oa + ea); al.target_internal_extend_gap_score = -ea
            al.query_internal_open_gap_score = -(ob + eb); al.query_internal_extend_gap_score = -eb
            # B's free prefix / suffix: end gaps in the target (A); A's: in the query (B)
            for side, free in (("target_left", bb), ("target_right", be)):
                setattr(al, side + "_open_gap_score", 0.0 if free else -(oa + ea))
                setattr(al, side + "_extend_gap_score", 0.0 if free else -ea)
            for side, free in (("query_left", ab), ("query_right", ae)):
                setattr(al, side + "_open_gap_score", 0.0 if free else -(ob + eb))
                setattr(al, side + "_extend_gap_score", 0.0 if free else -eb)
            ref = sg_reference(a, b, M, oa, ea, ob, eb, flags)[0]
            assert al.score(a, b) == pytest.approx(ref), (a, b, flags)


# ── Scores, end cells and paths against the reference ───────────────────────

@pytest.mark.parametrize("gap_model", ["affine", "linear"])
@pytest.mark.parametrize("traceback", ["pointers", "scores"])
def test_matches_reference(imatrix, gap_model, traceback):
    M = np.asarray(imatrix.to_matrix())
    aff = gap_model == "affine"
    for a, b, oa, ea, ob, eb in _corpus(7, 70):
        p = _params(imatrix, oa if aff else 0.0, ea, ob if aff else 0.0, eb)
        for flags in FLAGS:
            v, (ie, je), _ = sg_reference(a, b, M, oa, ea, ob, eb, flags, aff)
            sp = nwgrad.SeqPairDouble(a, b, p, gap_model=gap_model, mode="semiglobal",
                                      free_ends=flags, traceback=traceback)
            s, g = sp.score_and_grad()
            assert s == v, (a, b, flags)
            ci, cj = sp.coordinates()
            assert (int(ci[-1]), int(cj[-1])) == (ie, je), (a, b, flags, "end cell")
            i0, j0 = int(ci[0]), int(cj[0])
            assert (i0, j0) == (0, 0) or (i0 == 0 and flags[2]) or (j0 == 0 and flags[0]), (a, b, flags)
            sa, sb = sp.aligned()
            assert sa.replace("-", "") == a[i0:ie] and sb.replace("-", "") == b[j0:je]
            assert _path_score(sa, sb, M, oa, ea, ob, eb, aff) == s
            # the hard gradient is the path's counts: grad . params == score
            d, gd = p.to_dict(), g.to_dict()
            dot = float(np.sum(np.asarray(gd["matrix"]) * np.asarray(d["matrix"])))
            dot += sum(gd[k] * d[k] for k in ("gap_open_a", "gap_extend_a", "gap_open_b", "gap_extend_b"))
            assert dot == s


@pytest.mark.parametrize("gap_model", ["affine", "linear"])
def test_negative_gaps_and_float_params(gmatrix, gap_model):
    # non-representable and negative gap costs: the end rule and the borders still hold
    M = np.asarray(gmatrix.to_matrix())
    aff = gap_model == "affine"
    rng = random.Random(5)
    for _ in range(40):
        a, b = _rand(rng, 0, 8), _rand(rng, 0, 10)
        oa, ea, ob, eb = (rng.choice([0.1, -0.3, 1.7]), rng.choice([0.1, -0.2, 0.7]),
                          rng.choice([0.1, 0.0, 2.3]), rng.choice([0.3, -0.1, 0.9]))
        p = _params(gmatrix, oa if aff else 0.0, ea, ob if aff else 0.0, eb)
        for flags in FLAGS:
            v, end, _ = sg_reference(a, b, M, oa, ea, ob, eb, flags, aff)
            sp = nwgrad.SeqPairDouble(a, b, p, gap_model=gap_model, mode="semiglobal",
                                      free_ends=flags, traceback="pointers", kernel="scalar_fallback")
            assert sp.align() == pytest.approx(v, abs=1e-9), (a, b, flags)


# ── The soft path (log space) against a log-space reference ─────────────────

def sg_logz_reference(a, b, M, oa, ea, ob, eb, flags, affine=True):
    """log Z over the semi-global ensemble: sg_reference's DP with logsumexp for max."""
    ab, ae, bb, be = flags
    if not affine:
        oa = ob = 0.0
    lse = lambda *v: (lambda mx: mx if mx == NINF else mx + np.log(sum(np.exp(x - mx) for x in v)))(max(v))
    m, n = len(a), len(b)
    ia = [ALPHABET.index(c) for c in a]; ib = [ALPHABET.index(c) for c in b]
    FM = [[NINF] * (n + 1) for _ in range(m + 1)]
    FX = [[NINF] * (n + 1) for _ in range(m + 1)]
    FY = [[NINF] * (n + 1) for _ in range(m + 1)]
    FM[0][0] = 0.0
    for j in range(1, n + 1):
        if bb: FM[0][j] = 0.0
        else:  FY[0][j] = -(oa + j * ea)
    for i in range(1, m + 1):
        if ab: FM[i][0] = 0.0
        else:  FX[i][0] = -(ob + i * eb)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            FM[i][j] = lse(FM[i-1][j-1], FX[i-1][j-1], FY[i-1][j-1]) + M[ia[i-1]][ib[j-1]]
            FX[i][j] = lse(FM[i-1][j] - ob - eb, FX[i-1][j] - eb, FY[i-1][j] - ob - eb)
            FY[i][j] = lse(FM[i][j-1] - oa - ea, FX[i][j-1] - oa - ea, FY[i][j-1] - ea)
    ends = [FM[i][j] for i in range(m + 1) for j in range(n + 1)
            if (i == m and j == n) or (i == m and be) or (j == n and ae)]
    ends += [FX[i][j] for i in range(m + 1) for j in range(n + 1)
             if (i == m and j == n) or (i == m and be) or (j == n and ae)]
    ends += [FY[i][j] for i in range(m + 1) for j in range(n + 1)
             if (i == m and j == n) or (i == m and be) or (j == n and ae)]
    return lse(*ends)


@pytest.mark.parametrize("gap_model", ["affine", "linear"])
def test_soft_matches_reference(gmatrix, gap_model):
    M = np.asarray(gmatrix.to_matrix())
    aff = gap_model == "affine"
    rng = random.Random(71)
    h = 1e-6
    for _ in range(6):
        a, b = _rand(rng, 0, 6), _rand(rng, 0, 7)
        oa, ea, ob, eb = 1.5, 0.7, 2.0, 0.4
        for flags in FLAGS:
            p = _params(gmatrix, oa if aff else 0.0, ea, ob if aff else 0.0, eb)
            sp = nwgrad.SeqPairDouble(a, b, p, gap_model=gap_model, mode="semiglobal",
                                      free_ends=flags, grad_mode="soft")
            lz, g = sp.score_and_grad()
            ref = sg_logz_reference(a, b, M, oa, ea, ob, eb, flags, aff)
            assert lz == pytest.approx(ref, rel=1e-10, abs=1e-10), (a, b, flags)
            # the gradient: central differences of the reference's log Z
            gd = g.to_dict()
            for k, (dga, dea, dgb, deb) in (("gap_extend_a", (0, h, 0, 0)), ("gap_extend_b", (0, 0, 0, h)),
                                             ("gap_open_a", (h, 0, 0, 0)), ("gap_open_b", (0, 0, h, 0))):
                if not aff and k.startswith("gap_open"):
                    continue
                up = sg_logz_reference(a, b, M, oa + dga, ea + dea, ob + dgb, eb + deb, flags, aff)
                dn = sg_logz_reference(a, b, M, oa - dga, ea - dea, ob - dgb, eb - deb, flags, aff)
                assert gd[k] == pytest.approx((up - dn) / (2 * h), abs=1e-6), (a, b, flags, k)
            if a and b:
                x, y = ALPHABET.index(a[0]), ALPHABET.index(b[0])
                Mu, Md = M.copy(), M.copy(); Mu[x, y] += h; Md[x, y] -= h
                num = (sg_logz_reference(a, b, Mu, oa, ea, ob, eb, flags, aff)
                       - sg_logz_reference(a, b, Md, oa, ea, ob, eb, flags, aff)) / (2 * h)
                assert np.asarray(gd["matrix"])[x, y] == pytest.approx(num, abs=1e-6), (a, b, flags)


def test_soft_glocal_ends_unchanged(gmatrix):
    # glocal's ends keep the scaled kernels; every other set the log path -- the empty set
    # against mode "global" within the soft tolerance, glocal's bit for bit
    p = _params(gmatrix, 1.5, 0.7, 2.0, 0.4)
    rng = random.Random(72)
    for _ in range(10):
        a, b = _rand(rng, 1, 20), _rand(rng, 1, 30)
        s0, g0 = nwgrad.SeqPairDouble(a, b, p, mode="glocal", grad_mode="soft").score_and_grad()
        s1, g1 = nwgrad.SeqPairDouble(a, b, p, mode="semiglobal", free_ends=("b_begin", "b_end"),
                                      grad_mode="soft").score_and_grad()
        assert s0 == s1 and np.array_equal(np.asarray(g0.to_dict()["matrix"]), np.asarray(g1.to_dict()["matrix"]))
        s0, g0 = nwgrad.SeqPairDouble(a, b, p, mode="global", grad_mode="soft").score_and_grad()
        s1, g1 = nwgrad.SeqPairDouble(a, b, p, mode="semiglobal", free_ends=(), grad_mode="soft").score_and_grad()
        assert s1 == pytest.approx(s0, rel=1e-12)
        np.testing.assert_allclose(np.asarray(g1.to_dict()["matrix"]), np.asarray(g0.to_dict()["matrix"]), atol=1e-10)


@pytest.mark.parametrize("traceback", ["auto", "grid", "hirschberg", "hirschberg_pmax"])
def test_other_tracebacks_run_pointers(gmatrix, traceback):
    p = _params(gmatrix, 2.0, 0.7, 1.5, 0.5)
    rng = random.Random(81)
    for cls in (nwgrad.SeqPairDouble, nwgrad.SeqPair):
        for _ in range(4):
            a, b = _rand(rng, 600, 700), _rand(rng, 600, 900)   # past hb_cutoff
            for flags in [(True, True, True, True), (True, False, False, True), (False, True, False, False)]:
                kw = dict(mode="semiglobal", free_ends=flags)
                assert _same(_run(cls, a, b, p, traceback="pointers", **kw), _run(cls, a, b, p, traceback=traceback, **kw))


# ── Bit identity: the existing modes, every level, score-only ───────────────

def _run(cls, a, b, p, **kw):
    sp = cls(a, b, p, **kw)
    s, g = sp.score_and_grad()
    return s, sp.aligned(), np.asarray(g.to_dict()["matrix"]), [g.to_dict()[k] for k in
            ("gap_open_a", "gap_extend_a", "gap_open_b", "gap_extend_b")]


def _same(r0, r1):
    return r0[0] == r1[0] and r0[1] == r1[1] and np.array_equal(r0[2], r1[2]) and r0[3] == r1[3]


CLASSES = ["SeqPairDouble", "SeqPair", "SeqPairInt32", "SeqPairInt16"]


def _class_params(cls, gmatrix, imatrix, aff):
    if "Int" in cls:
        return _params(imatrix, 3 if aff else 0, 1, 2 if aff else 0, 1)
    return _params(gmatrix, 2.0 if aff else 0.0, 0.7, 1.5 if aff else 0.0, 0.5)


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("gap_model", ["affine", "linear"])
@pytest.mark.parametrize("traceback", ["pointers", "scores"])
def test_flags_reproduce_global_and_glocal(gmatrix, imatrix, cls, gap_model, traceback):
    C = getattr(nwgrad, cls)
    p = _class_params(cls, gmatrix, imatrix, gap_model == "affine")
    rng = random.Random(21)
    for _ in range(30):
        a, b = _rand(rng, 0, 40), _rand(rng, 0, 90)
        kw = dict(gap_model=gap_model, traceback=traceback)
        assert _same(_run(C, a, b, p, mode="global", **kw),
                     _run(C, a, b, p, mode="semiglobal", free_ends=(False,) * 4, **kw)), (a, b)
        gl = _run(C, a, b, p, mode="glocal", **kw)
        assert _same(gl, _run(C, a, b, p, mode="semiglobal", free_ends=("b_begin", "b_end"), **kw))
        assert _same(gl, _run(C, a, b, p, mode="semiglobal", free_ends=(False, False, True, True), **kw))


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("gap_model", ["affine", "linear"])
@pytest.mark.parametrize("traceback", ["pointers", "scores"])
def test_levels_match_scalar(gmatrix, imatrix, cls, gap_model, traceback):
    C = getattr(nwgrad, cls)
    p = _class_params(cls, gmatrix, imatrix, gap_model == "affine")
    levels = [lv for lv in nwgrad.available_isa_levels() if lv != "scalar_fallback"]
    rng = random.Random(31)
    cases = [(_rand(rng, 0, 50), _rand(rng, 0, 130)) for _ in range(12)]
    cases += [("ACGTA", ""), ("", "ACGT"), ("A" * 30, "C" * 3)]
    for flags in FLAGS:
        for a, b in cases:
            kw = dict(gap_model=gap_model, traceback=traceback, mode="semiglobal", free_ends=flags)
            r0 = _run(C, a, b, p, kernel="scalar_fallback", **kw)
            for lv in levels:
                assert _same(r0, _run(C, a, b, p, kernel=lv, **kw)), (lv, flags, a, b)


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("gap_model", ["affine", "linear"])
def test_score_only_equals_the_fill(gmatrix, imatrix, cls, gap_model):
    # grad_mode "none": the score-only kernels (every level's, the scalar copy) give the
    # traceback fill's score bit for bit
    C = getattr(nwgrad, cls)
    p = _class_params(cls, gmatrix, imatrix, gap_model == "affine")
    rng = random.Random(41)
    cases = [(_rand(rng, 0, 45), _rand(rng, 0, 120)) for _ in range(10)] + [("ACG", ""), ("", "AC")]
    for flags in FLAGS:
        for a, b in cases:
            for lv in nwgrad.available_isa_levels():
                kw = dict(gap_model=gap_model, mode="semiglobal", free_ends=flags, kernel=lv)
                ref = C(a, b, p, traceback="pointers", **kw).align()
                assert C(a, b, p, grad_mode="none", **kw).align() == ref, (lv, flags, a, b)


@pytest.mark.parametrize("gap_model", ["affine", "linear"])
@pytest.mark.parametrize("cls", ["SeqPairBatchDouble", "SeqPairBatch", "SeqPairBatchInt32"])
def test_batch_matches_pairs(gmatrix, imatrix, gap_model, cls):
    C = getattr(nwgrad, cls)
    S = getattr(nwgrad, cls.replace("Batch", ""))
    p = _class_params(cls, gmatrix, imatrix, gap_model == "affine")
    rng = random.Random(51)
    A = [_rand(rng, 0, 30) for _ in range(40)]
    B = [_rand(rng, 0, 60) for _ in range(40)]
    sets = [("overlap", NAMED["overlap"]), ("semiglobal", (True, False, True, False)),
            ("semiglobal", (False, True, False, True)), ("semiglobal", (True, False, False, True))]
    for name, flags in sets:
        kw = dict(free_ends=flags) if name == "semiglobal" else {}
        for grad_mode in ("hard", "none"):
            for fill in ("interpair", "striped"):
                bt = C(A, B, p, gap_model=gap_model, mode=name, grad_mode=grad_mode, n_threads=3, **kw)
                bt.fill = fill
                bt.score_and_grad()
                sc = bt.scores()
                for k in range(len(A)):
                    sp = S(A[k], B[k], p, gap_model=gap_model, mode="semiglobal", free_ends=flags)
                    s, g = sp.score_and_grad()
                    assert sc[k] == s, (name, k, grad_mode, fill)
                    if grad_mode == "hard":
                        assert np.array_equal(np.asarray(bt[k].grad.to_dict()["matrix"]),
                                              np.asarray(g.to_dict()["matrix"]))


# ── The names, the flags and the API surface ───────────────────────────────

def test_names_and_flags(gmatrix):
    p = _params(gmatrix, 2.0, 1.0, 1.5, 0.5)
    M = np.asarray(gmatrix.to_matrix())
    rng = random.Random(61)
    pairs = [(_rand(rng, 1, 9), _rand(rng, 1, 12)) for _ in range(20)]
    sp = nwgrad.SeqPairDouble("ACGT", "AACGTT", p, mode="overlap")
    assert sp.mode == "overlap" and sp.free_ends == (True,) * 4
    for a, b in pairs:
        got = nwgrad.SeqPairDouble(a, b, p, mode="overlap").align()
        assert got == pytest.approx(sg_reference(a, b, M, 2.0, 1.0, 1.5, 0.5, (True,) * 4)[0])
    names = ("a_begin", "a_end", "b_begin", "b_end")
    for flags in FLAGS:
        by_name = tuple(n for n, f in zip(names, flags) if f)
        sp = nwgrad.SeqPairDouble("AC", "ACG", p, mode="semiglobal", free_ends=flags)
        assert sp.free_ends == flags
        if by_name:
            sp2 = nwgrad.SeqPairDouble("AC", "ACG", p, mode="semiglobal", free_ends=by_name)
            assert sp2.free_ends == flags
        expect = {(True,) * 4: "overlap", (False, False, True, True): "glocal"}.get(flags, "semiglobal")
        assert sp.mode == expect
        assert nwgrad.SeqPairBatchDouble(["AC"], ["ACG"], p, mode="semiglobal", free_ends=flags).free_ends == flags
    assert nwgrad.SeqPairDouble("AC", "ACG", p, mode="semiglobal", free_ends="a_end").free_ends == (False, True, False, False)
    assert nwgrad.SeqPairDouble("AC", "ACG", p, mode="global").free_ends == (False,) * 4
    assert nwgrad.SeqPairDouble("AC", "ACG", p, mode="local").free_ends is None
    assert nwgrad.SeqPairDouble("AC", "ACG", p, mode="glocal").free_ends == (False, False, True, True)
    # the convenience functions
    s, g = nwgrad.semiglobal_grad_double("ACGT", "TTACG", p, ("a_end", "b_begin"))
    assert s == nwgrad.semiglobal_score_double("ACGT", "TTACG", p, (False, True, True, False))
    assert nwgrad.semiglobal_score_double("ACGT", "TTACG", p) == nwgrad.SeqPairDouble("ACGT", "TTACG", p, mode="overlap").align()


def test_semantics_by_example():
    # match 2 / mismatch -1, a gap of k costs 4 + k
    mat = nwgrad.SubstMatrix(np.eye(4) * 3 - 1, alphabet=ALPHABET)
    p = _params(mat, 4.0, 1.0, 4.0, 1.0)
    q, d = "AAA", "CCCCAAACCCC"
    sc = lambda a, b, fe: nwgrad.SeqPairDouble(a, b, p, mode="semiglobal", free_ends=fe).align()
    assert sc(q, d, ()) == -10                               # global (no free end)
    assert sc(q, d, ("b_begin", "b_end")) == 6               # A within B
    assert sc(d, q, ("a_begin", "a_end")) == 6               # B within A
    assert sc(q, d, ("a_begin", "a_end", "b_begin", "b_end")) == 6
    assert sc(q, d, ("b_begin",)) == -2 and sc(q, d, ("b_end",)) == -2
    assert sc(d, q, ("a_begin",)) == -2 and sc(d, q, ("a_end",)) == -2
    # dovetails: A's suffix over B's prefix, and the other way round
    assert sc("TTTACG", "ACGCCC", ("a_begin", "b_end")) == 6
    assert sc("ACGCCC", "TTTACG", ("a_end", "b_begin")) == 6
    # the empty overlap scores 0: the ends include the border cells (m, 0) and (0, n), as
    # glocal's include (m, 0)
    assert sc(q, d, ("a_begin", "b_end")) == 0
    assert sc(q, d, ("a_end", "b_begin")) == 0


def test_errors():
    p = _params(nwgrad.SubstMatrix(np.eye(4), alphabet=ALPHABET), 1.0, 1.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        nwgrad.SeqPair("A", "C", p, mode="semiglobal")                       # no free_ends
    with pytest.raises(ValueError):
        nwgrad.SeqPair("A", "C", p, mode="global", free_ends=(True,) * 4)   # not semiglobal
    with pytest.raises(ValueError):
        nwgrad.SeqPair("A", "C", p, mode="semiglobal", free_ends=(True,) * 3)
    with pytest.raises(ValueError):
        nwgrad.SeqPair("A", "C", p, mode="semiglobal", free_ends=("a_start",))
    with pytest.raises(ValueError):
        nwgrad.SeqPair("A", "C", p, mode="sg")
    # guide bands take glocal's ends only
    sp = nwgrad.SeqPair("ACG", "CA", p, mode="overlap")
    sp.align()
    with pytest.raises(ValueError):
        sp.banded_align(4)
