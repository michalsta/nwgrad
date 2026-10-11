"""Tracks: K parameter sets, each row of sequence A scored by the set its track selects.

Checked against an exhaustive oracle (track_oracle.py, no DP), against the untracked
library (a track whose sets are all equal must reproduce the untracked result BIT FOR
BIT — the arrays, borders and stacked block are built so it does), and through the
linearity of the score in the parameters (score == grad . params for a hard path).
"""

import itertools

import numpy as np
import pytest

import nwgrad

import track_oracle as orc

ALPHA = "ACGT"
N = len(ALPHA)

# (name, gap model, mode) -> (score fn, hard grad fn), per precision suffix.
FUNCS = {
    ("linear", "global"): ("nw_score", "nw_grad"),
    ("linear", "local"): ("sw_score", "sw_grad"),
    ("affine", "global"): ("nw_score_affine", "nw_affine_grad"),
    ("affine", "local"): ("sw_score_affine", "sw_affine_grad"),
}
CASES = list(FUNCS)
PRECISIONS = ["_double", ""]   # double, float32


def fn(name, prec):
    return getattr(nwgrad, name + prec)


def enc(s):
    return [ALPHA.index(c) for c in s]


def random_params(rng, K, lo=0.3, hi=2.0, integral=False):
    """K random sets.  Real-valued so ties are rare; gaps positive (local is then
    well-posed and the oracle's local candidates hold every optimum)."""
    if integral:
        mats = rng.integers(-3, 4, size=(K, N, N)).astype(float)
        gaps = rng.integers(1, 4, size=(K, 4)).astype(float)
    else:
        mats = rng.normal(0.0, 1.5, size=(K, N, N))
        gaps = rng.uniform(lo, hi, size=(K, 4))
    return mats, gaps


def to_params(mats, gaps):
    return nwgrad.AlignParams.from_arrays(np.ascontiguousarray(mats), np.ascontiguousarray(gaps), ALPHA)


def random_seq(rng, n):
    return "".join(rng.choice(list(ALPHA), size=n))


def random_tracks(rng, m, K, with_gap):
    track = list(rng.integers(0, K, size=m))
    gtrack = list(rng.integers(0, K, size=m + 1)) if with_gap else None
    return track, gtrack


def lib_hard(case, prec, a, b, P, track, gtrack, kernel="auto"):
    gm, mode = case
    s, g = fn(FUNCS[case][1], prec)(a, b, P, kernel=kernel, track=track, gap_track=gtrack)
    return s, g


def grad_arrays(g):
    return g.matrices, g.gaps


# ── The oracle ─────────────────────────────────────────────────────────────────

def _instances(seed, count, max_m=4, max_n=4, Ks=(1, 2, 3)):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(count):
        m = int(rng.integers(0, max_m + 1))
        n = int(rng.integers(0, max_n + 1))
        K = int(rng.choice(Ks))
        a, b = random_seq(rng, m), random_seq(rng, n)
        mats, gaps = random_params(rng, K)
        track, gtrack = random_tracks(rng, m, K, with_gap=bool(rng.integers(0, 2)))
        out.append((a, b, mats, gaps, track, gtrack))
    return out


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("prec", PRECISIONS)
@pytest.mark.parametrize("kernel", ["auto", "scalar_fallback"])
def test_score_and_hard_grad_match_oracle(case, prec, kernel):
    affine, local = case[0] == "affine", case[1] == "local"
    tol = 1e-9 if prec == "_double" else 2e-4
    checked = 0
    for a, b, mats, gaps, track, gtrack in _instances(hash(case) % 1000 + len(prec), 60):
        gt = gtrack if gtrack is not None else orc.default_gap_track(track)
        if local and (len(a) == 0 or len(b) == 0):
            continue
        sc, gm, gg, margin = orc.best(enc(a), enc(b), mats, gaps, track, gt, affine, local)
        P = to_params(mats, gaps)
        s, g = lib_hard(case, prec, a, b, P, track, gtrack, kernel)
        assert s == pytest.approx(sc, abs=tol)
        s2 = fn(FUNCS[case][0], prec)(a, b, P, kernel=kernel, track=track, gap_track=gtrack)
        assert s2 == pytest.approx(sc, abs=tol)
        if margin > 1e-3:
            lm, lg = grad_arrays(g)
            np.testing.assert_array_equal(lm, gm)
            np.testing.assert_array_equal(lg, gg)
            checked += 1
    assert checked > 20


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("prec", PRECISIONS)
def test_hard_grad_dotted_with_params_is_the_score(case, prec):
    """The score is linear in every parameter along a fixed path, so for the path the
    DP returns, sum(grad * params) is exactly its score (here up to rounding)."""
    rng = np.random.default_rng(7)
    for _ in range(40):
        m, n, K = int(rng.integers(1, 30)), int(rng.integers(1, 30)), int(rng.integers(1, 6))
        a, b = random_seq(rng, m), random_seq(rng, n)
        mats, gaps = random_params(rng, K)
        track, gtrack = random_tracks(rng, m, K, with_gap=True)
        P = to_params(mats, gaps)
        s, g = lib_hard(case, prec, a, b, P, track, gtrack)
        dot = float((g.matrices * mats).sum() + (g.gaps * gaps).sum())
        assert dot == pytest.approx(s, abs=1e-9 if prec == "_double" else 1e-3)


# ── Equal sets reproduce the untracked problem, bit for bit ────────────────────

def _base(rng, integral):
    mats, gaps = random_params(rng, 1, integral=integral)
    return mats[0], gaps[0]


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("prec", PRECISIONS)
@pytest.mark.parametrize("integral", [False, True])
@pytest.mark.parametrize("kernel", ["auto", "scalar_fallback"])
def test_equal_sets_are_bit_identical_to_untracked(case, prec, integral, kernel):
    """K copies of one set, any tracks: score bit-identical, and the K-set gradient
    summed over the sets bit-identical to the untracked gradient (integral params
    make ties common, so the paths must match tie for tie)."""
    rng = np.random.default_rng(11 + integral)
    for _ in range(40):
        m, n, K = int(rng.integers(0, 25)), int(rng.integers(0, 25)), int(rng.integers(1, 5))
        a, b = random_seq(rng, m), random_seq(rng, n)
        M, G = _base(rng, integral)
        P1 = nwgrad.AlignParams(M, ALPHA, *G)
        PK = to_params(np.stack([M] * K), np.stack([G] * K))
        track, gtrack = random_tracks(rng, m, K, with_gap=bool(rng.integers(0, 2)))
        s1, g1 = fn(FUNCS[case][1], prec)(a, b, P1, kernel=kernel)
        sk, gk = fn(FUNCS[case][1], prec)(a, b, PK, kernel=kernel, track=track, gap_track=gtrack)
        assert sk == s1  # bitwise (floats compare exactly)
        np.testing.assert_array_equal(gk.matrices.sum(axis=0), g1.to_dict()["matrix"])
        np.testing.assert_array_equal(gk.gaps.sum(axis=0), g1.gaps[0])
        assert fn(FUNCS[case][0], prec)(a, b, PK, kernel=kernel, track=track, gap_track=gtrack) == \
            fn(FUNCS[case][0], prec)(a, b, P1, kernel=kernel)


@pytest.mark.parametrize("case", CASES)
def test_one_set_zero_track_is_untracked(case):
    rng = np.random.default_rng(3)
    for _ in range(20):
        m, n = int(rng.integers(0, 20)), int(rng.integers(0, 20))
        a, b = random_seq(rng, m), random_seq(rng, n)
        M, G = _base(rng, False)
        P = nwgrad.AlignParams(M, ALPHA, *G)
        s1, g1 = fn(FUNCS[case][1], "_double")(a, b, P)
        s2, g2 = fn(FUNCS[case][1], "_double")(a, b, P, track=[0] * m, gap_track=[0] * (m + 1))
        assert s1 == s2
        np.testing.assert_array_equal(g1.matrices, g2.matrices)
        np.testing.assert_array_equal(g1.gaps, g2.gaps)


# ── The gradient is a derivative ───────────────────────────────────────────────

@pytest.mark.parametrize("case", CASES)
def test_gradient_matches_finite_differences(case):
    """Away from ties the hard score is linear near the parameters: perturbing any one
    parameter of any set moves the score by exactly grad * h."""
    rng = np.random.default_rng(5)
    h = 1e-6
    tested = 0
    for _ in range(15):
        m, n, K = int(rng.integers(3, 12)), int(rng.integers(3, 12)), int(rng.integers(2, 5))
        a, b = random_seq(rng, m), random_seq(rng, n)
        mats, gaps = random_params(rng, K)
        track, gtrack = random_tracks(rng, m, K, with_gap=True)
        s0, g = lib_hard(case, "_double", a, b, to_params(mats, gaps), track, gtrack)
        for _ in range(6):
            dm, dg = np.zeros_like(mats), np.zeros_like(gaps)
            if rng.integers(0, 2):
                dm[tuple(rng.integers(0, [K, N, N]))] = h
            else:
                dg[tuple(rng.integers(0, [K, 4]))] = h
            s1, _ = lib_hard(case, "_double", a, b, to_params(mats + dm, gaps + dg), track, gtrack)
            pred = float((g.matrices * dm).sum() + (g.gaps * dg).sum())
            if abs((s1 - s0) - pred) < 1e-9:
                tested += 1
            else:
                # a tie broken differently under the perturbation: the derivative is
                # one-sided there; it must still be a subgradient bound
                assert s1 - s0 >= pred - 1e-9
    assert tested > 60


# ── Semantics pinned on hand-made cases ────────────────────────────────────────

def _two_sets(open_a=(10.0, 10.0), ext_a=(10.0, 10.0), open_b=(10.0, 10.0), ext_b=(10.0, 10.0)):
    """Two sets, match +1 / mismatch -1; penalties not under test default to 10."""
    mats = np.stack([np.eye(N) * 2 - 1, np.eye(N) * 2 - 1])
    gaps = np.array([[open_a[0], ext_a[0], open_b[0], ext_b[0]],
                     [open_a[1], ext_a[1], open_b[1], ext_b[1]]], dtype=float)
    return to_params(mats, gaps)


def _check_oracle(a, b, P, track, gtrack, affine=True, local=False):
    gt = gtrack if gtrack is not None else orc.default_gap_track(track)
    sc, gm, gg, _ = orc.best(enc(a), enc(b), P.matrices, P.gaps, track, gt, affine, local)
    s, g = nwgrad.nw_affine_grad_double(a, b, P, track=track, gap_track=gtrack)
    assert s == pytest.approx(sc)
    np.testing.assert_array_equal(g.matrices, gm)
    np.testing.assert_array_equal(g.gaps, gg)
    return s, g


def _three_sets(gaps):
    """Match +1 / mismatch -1 in every set; `gaps` (3, 4) as given."""
    return to_params(np.stack([np.eye(N) * 2 - 1] * 3), np.array(gaps, dtype=float))


def test_vertical_run_pays_the_open_of_its_first_row():
    # A = "ATTA" against B = "AA": the T's are a gap in B, rows 2 and 3.  Set 2 makes
    # every other gap prohibitive.
    P = _three_sets([[10, 10, 1, .5], [10, 10, 5, .25], [10, 10, 10, 10]])
    s, g = _check_oracle("ATTA", "AA", P, [2, 1, 0, 2], [2] * 5)
    assert s == pytest.approx(2 - (5.0 + 0.25 + 0.5))   # opens in row 2: set 1's open
    np.testing.assert_array_equal(g.gaps[:, 2], [0, -1, 0])
    np.testing.assert_array_equal(g.gaps[:, 3], [-1, -1, 0])
    s, g = _check_oracle("ATTA", "AA", P, [2, 0, 1, 2], [2] * 5)
    assert s == pytest.approx(2 - (1.0 + 0.5 + 0.25))   # opens in row 2: set 0's open
    np.testing.assert_array_equal(g.gaps[:, 2], [-1, 0, 0])


def test_horizontal_gaps_use_the_gap_track_row():
    # A = "AT" against B = "ACCT": the C's are a gap in A within DP row 1.
    P = _three_sets([[1, .5, 10, 10], [3, .75, 10, 10], [10, 10, 10, 10]])
    s, g = _check_oracle("AT", "ACCT", P, [2, 2], [2, 1, 2])
    assert s == pytest.approx(2 - (3.0 + 2 * 0.75))
    np.testing.assert_array_equal(g.gaps[:, 0], [0, -1, 0])
    np.testing.assert_array_equal(g.gaps[:, 1], [0, -2, 0])
    s, g = _check_oracle("AT", "ACCT", P, [2, 2], [2, 0, 2])
    assert s == pytest.approx(2 - (1.0 + 2 * 0.5))
    # the default gap track takes row 1 from track[0]
    s, _ = _check_oracle("AT", "ACCT", P, [0, 2], None)
    s2, _ = _check_oracle("AT", "ACCT", P, [0, 2], [0, 0, 2])
    assert s == s2


def test_leading_and_trailing_rows_have_their_own_gap_sets():
    # A = "A" against B = "CCA" (Global): the C's are a gap in A, either leading (row 0,
    # before the A) or trailing (row 1, after it) — whichever row's set is cheaper.
    P = _two_sets(open_a=(4.0, 0.0), ext_a=(4.0, 0.0))
    s, _ = _check_oracle("A", "CCA", P, [0], [1, 0])
    assert s == pytest.approx(1.0)          # free leading gaps (row 0), then A-A
    s, _ = _check_oracle("A", "AGG", P, [0], [0, 1])
    assert s == pytest.approx(1.0)          # A-A, then free trailing gaps (row 1)
    s, _ = _check_oracle("A", "CCA", P, [0], [0, 1])
    assert s == pytest.approx(-1.0)         # A-C, then the free trailing row 1 takes CA


def test_substitution_uses_the_rows_matrix():
    mats = np.stack([np.full((N, N), -1.0), np.full((N, N), 4.0)])
    P = to_params(mats, np.ones((2, 4)))
    s, g = nwgrad.nw_grad_double("AC", "AC", P, track=[1, 0])
    assert s == pytest.approx(3.0)
    assert g.matrices[1, 0, 0] == 1 and g.matrices[0, 1, 1] == 1


# ── Validation ─────────────────────────────────────────────────────────────────

def test_bad_tracks_raise():
    P = to_params(np.zeros((2, N, N)), np.ones((2, 4)))
    with pytest.raises(ValueError, match="track has 1 entries"):
        nwgrad.nw_grad_double("AC", "AC", P, track=[0])
    with pytest.raises(ValueError, match="gap_track has 2 entries"):
        nwgrad.nw_grad_double("AC", "AC", P, track=[0, 1], gap_track=[0, 0])
    with pytest.raises(ValueError, match="not a parameter set"):
        nwgrad.nw_grad_double("AC", "AC", P, track=[0, 2])
    with pytest.raises(ValueError, match="not a parameter set"):
        nwgrad.nw_grad_double("AC", "AC", P, gap_track=[0, 0, 5])
    with pytest.raises(ValueError, match="not a parameter set index"):
        nwgrad.nw_grad_double("AC", "AC", P, track=[0, -1])
    with pytest.raises(TypeError):
        nwgrad.nw_grad_double("AC", "AC", P, track="xy")


def test_default_track_is_set_zero():
    mats = np.stack([np.eye(N), np.eye(N) * 7])
    P = to_params(mats, np.ones((2, 4)))
    s, g = nwgrad.nw_grad_double("ACGT", "ACGT", P)
    assert s == 4.0
    assert g.matrices[1].sum() == 0


# ── AlignParams with sets ──────────────────────────────────────────────────────

def test_param_sets_api():
    rng = np.random.default_rng(1)
    mats, gaps = random_params(rng, 3)
    P = to_params(mats, gaps)
    assert P.num_sets == 3
    np.testing.assert_array_equal(P.matrices, mats)
    np.testing.assert_array_equal(P.gaps, gaps)
    np.testing.assert_array_equal(P.matrix.to_matrix(), mats[0])
    assert P.gap_open_b == gaps[0, 2]
    for k in range(3):
        np.testing.assert_array_equal(P[k].matrices[0], mats[k])
        np.testing.assert_array_equal(P[k].gaps[0], gaps[k])
    np.testing.assert_array_equal(P[-1].gaps[0], gaps[2])
    S = nwgrad.AlignParams.stack([P[2], P[0], P[1]])
    np.testing.assert_array_equal(S.gaps, gaps[[2, 0, 1]])
    S[1] = P[1]
    np.testing.assert_array_equal(S.gaps, gaps[[2, 1, 1]])
    Q = P + P * 2.0 - P
    np.testing.assert_allclose(Q.matrices, 2 * mats)
    np.testing.assert_allclose(Q.gaps, 2 * gaps)
    P.gaps = gaps * 3
    np.testing.assert_array_equal(P.gaps, gaps * 3)
    P.matrices = mats * 0
    assert not P.matrices.any()
    d = P.to_dict()
    assert d["num_sets"] == 3 and d["matrices"].shape == (3, N, N) and d["gaps"].shape == (3, 4)
    with pytest.raises(ValueError, match="numbers of sets"):
        P + P[0]
    with pytest.raises(IndexError):
        P[3]
    with pytest.raises(ValueError, match=r"shape \(K, 4\)"):
        nwgrad.AlignParams.from_arrays(mats, gaps[:2], ALPHA)
    one = P[0]
    assert set(one.to_dict()) == {"matrix", "alphabet", "gap_open_a", "gap_extend_a",
                                  "gap_open_b", "gap_extend_b"}


def test_param_sets_gradient_accumulates_in_place():
    P = to_params(np.zeros((2, N, N)), np.ones((2, 4)))
    s, g = nwgrad.nw_grad_double("ACGT", "AGT", P, track=[0, 1, 1, 0])
    acc = g * 0.0
    acc += g
    acc += g
    np.testing.assert_array_equal(acc.matrices, 2 * g.matrices)
    np.testing.assert_array_equal(acc.gaps, 2 * g.gaps)


# ── Banded ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("prec", PRECISIONS)
def test_banded_equal_sets_bit_identical_and_wide_band_is_full(case, prec):
    rng = np.random.default_rng(21)
    for _ in range(30):
        m, n, K = int(rng.integers(1, 25)), int(rng.integers(1, 25)), int(rng.integers(1, 4))
        a, b = random_seq(rng, m), random_seq(rng, n)
        M, G = _base(rng, bool(rng.integers(0, 2)))
        P1 = nwgrad.AlignParams(M, ALPHA, *G)
        PK = to_params(np.stack([M] * K), np.stack([G] * K))
        track, gtrack = random_tracks(rng, m, K, with_gap=True)
        band = int(rng.integers(1, 6))
        s1, g1 = fn(FUNCS[case][1], prec)(a, b, P1, band=band)
        sk, gk = fn(FUNCS[case][1], prec)(a, b, PK, band=band, track=track, gap_track=gtrack)
        assert sk == s1
        np.testing.assert_array_equal(gk.matrices.sum(axis=0), g1.matrices[0])
        np.testing.assert_array_equal(gk.gaps.sum(axis=0), g1.gaps[0])
        # distinct sets: a band wider than B is the full DP
        mats, gaps = random_params(rng, K)
        PD = to_params(mats, gaps)
        sf, gf = fn(FUNCS[case][1], prec)(a, b, PD, track=track, gap_track=gtrack)
        sb, gb = fn(FUNCS[case][1], prec)(a, b, PD, band=n + 1, track=track, gap_track=gtrack)
        assert sb == sf
        np.testing.assert_array_equal(gb.matrices, gf.matrices)
        np.testing.assert_array_equal(gb.gaps, gf.gaps)
