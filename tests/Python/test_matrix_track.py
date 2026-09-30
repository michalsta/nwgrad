"""The matrix track: a different substitution matrix per position of sequence A.

AlignParams holds K >= 1 matrix SLOTS over one alphabet; a per-problem TRACK names one
slot per residue of sequence A.  Internally this is a widening of the DP's ROW ALPHABET
from "residue" to "(slot, residue)" — see Aligner::build_row_alphabet — so the claims
worth pinning are:

  1. K == 1 with no track is the old library, EXACTLY.  So is an all-zeros track, and so
     is any track over slots that happen to be equal: the reduction must not perturb the
     arithmetic, only the indexing.  Asserted with `==`, not approx.
  2. A tracked score really is the per-position model, checked against an exhaustive
     enumeration of every alignment path (bruteforce.all_paths) and against an
     independent Python DP written from the recurrence rather than from the kernel.
  3. The gradient lands in the RIGHT SLOT — every position that selected slot k
     contributes to slot k and to no other — checked against the same oracles and
     against finite differences of the score.
  4. It survives every traceback mode, the banded path, and the batch paths.

Run under NWGRAD_FORCE_KERNEL=simd and every assertion here becomes a scalar-vs-simd
bit-identity check at each ISA level, which is the property the whole library rests on.
"""

import itertools
import math

import numpy as np
import pytest

import nwgrad
from bruteforce import all_paths

ALPHA = "ACDEFGHIKLMNPQRSTVWY"
N = len(ALPHA)

GO_A, GE_A, GO_B, GE_B = 3.0, 1.0, 2.5, 0.75


# ── Fixtures: distinguishable matrices ───────────────────────────────────────


def _matrix(seed, alphabet=ALPHA):
    """A dense, asymmetric, unambiguous matrix.  Distinct seeds give matrices that
    disagree on essentially every cell, so a track that reads the wrong slot cannot
    coincidentally produce the right score."""
    rng = np.random.default_rng(seed)
    arr = rng.normal(0.0, 2.0, size=(len(alphabet), len(alphabet)))
    return nwgrad.SubstMatrix(np.ascontiguousarray(arr), alphabet)


def _params(mats, alphabet=ALPHA):
    mats = list(mats)
    return nwgrad.AlignParams(mats, GO_A, GE_A, GO_B, GE_B)


def _seq(seed, n, alphabet=ALPHA):
    rng = np.random.default_rng(seed)
    return "".join(rng.choice(list(alphabet), size=n))


# ── Oracle 1: exhaustive enumeration of every path ───────────────────────────


def _tracked_path_score(path, a, b, mats, track, affine):
    """Score one path under the per-position model, from the definition."""
    score = 0.0
    i = j = 0
    for step, run in itertools.groupby(path):
        k = len(list(run))
        if step == "D":
            for _ in range(k):
                score += mats[track[i]].score(a[i], b[j])
                i += 1
                j += 1
        elif step == "X":          # gap in B, consumes A, uses the b params
            score -= k * GE_B + (GO_B if affine else 0.0)
            i += k
        else:                       # gap in A, consumes B, uses the a params
            score -= k * GE_A + (GO_A if affine else 0.0)
            j += k
    return score


def _exhaustive(a, b, mats, track, affine):
    return max(_tracked_path_score(p, a, b, mats, track, affine)
               for p in all_paths(len(a), len(b)))


def _exhaustive_argmax(a, b, mats, track, affine):
    """(best path, best score).  Only meaningful when the optimum is unique."""
    scored = [(p, _tracked_path_score(p, a, b, mats, track, affine))
              for p in all_paths(len(a), len(b))]
    return max(scored, key=lambda ps: ps[1])


def _path_slot_counts(path, a, b, track, nslots):
    """Per-slot matrix pair counts along one path: counts[k][ai, bj]."""
    counts = [np.zeros((N, N)) for _ in range(nslots)]
    i = j = 0
    for step in path:
        if step == "D":
            counts[track[i]][ALPHA.index(a[i]), ALPHA.index(b[j])] += 1
            i += 1
            j += 1
        elif step == "X":
            i += 1
        else:
            j += 1
    return counts


# ── Oracle 2: an independent affine DP, written from the recurrence ──────────


def _nw_affine_tracked(a, b, mats, track):
    """Global affine NW with a per-position matrix.  Plain Python, no shared code with
    the engine under test; transcribed from the model in aligner.hpp (an X run followed
    by a Y run pays both opens, and the borders charge one open plus k extends)."""
    neg = float("-inf")
    m, n = len(a), len(b)
    M = [[neg] * (n + 1) for _ in range(m + 1)]
    X = [[neg] * (n + 1) for _ in range(m + 1)]
    Y = [[neg] * (n + 1) for _ in range(m + 1)]
    M[0][0] = 0.0
    for i in range(1, m + 1):
        X[i][0] = -(GO_B + i * GE_B)
    for j in range(1, n + 1):
        Y[0][j] = -(GO_A + j * GE_A)
    for i in range(1, m + 1):
        s = mats[track[i - 1]]
        for j in range(1, n + 1):
            M[i][j] = max(M[i - 1][j - 1], X[i - 1][j - 1], Y[i - 1][j - 1]) \
                + s.score(a[i - 1], b[j - 1])
            X[i][j] = max(M[i - 1][j] - GO_B, X[i - 1][j], Y[i - 1][j] - GO_B) - GE_B
            Y[i][j] = max(M[i][j - 1] - GO_A, X[i][j - 1] - GO_A, Y[i][j - 1]) - GE_A
    return max(M[m][n], X[m][n], Y[m][n])


def test_the_dp_oracle_agrees_with_exhaustive_enumeration():
    """Self-check on the oracle itself, before anything is tested against it."""
    mats = [_matrix(1), _matrix(2)]
    a, b = _seq(10, 6), _seq(11, 5)
    track = [0, 1, 1, 0, 1, 0]
    assert _nw_affine_tracked(a, b, mats, track) == pytest.approx(
        _exhaustive(a, b, mats, track, affine=True))


# ── 1. The reduction must not perturb anything ───────────────────────────────


def test_all_zeros_track_is_bit_identical_to_no_track():
    """The widened row alphabet must be pure re-indexing: identical doubles, in a
    different order, producing the identical score.  Exact equality, not approx."""
    p = _params([_matrix(1)])
    a, b = _seq(20, 60), _seq(21, 55)
    plain, gplain = nwgrad.nw_affine_grad_double(a, b, p)
    trk, gtrk = nwgrad.nw_affine_grad_double(a, b, p, track=[0] * len(a))
    assert trk == plain
    np.testing.assert_array_equal(gtrk.to_dict()["matrix"], gplain.to_dict()["matrix"])
    assert (gtrk.gap_open_a, gtrk.gap_extend_a, gtrk.gap_open_b, gtrk.gap_extend_b) == \
           (gplain.gap_open_a, gplain.gap_extend_a, gplain.gap_open_b, gplain.gap_extend_b)


def test_identical_slots_are_bit_identical_to_one_slot():
    """Three slots holding the same matrix, tracked arbitrarily, is the one-slot problem.
    This isolates the INDEXING from the arithmetic: if compaction or the profile rebuild
    changed a single value, this is where it shows."""
    m = _matrix(3)
    one = _params([m])
    three = _params([m, m, m])
    a, b = _seq(30, 70), _seq(31, 64)
    rng = np.random.default_rng(99)
    track = [int(k) for k in rng.integers(0, 3, size=len(a))]

    s1, g1 = nwgrad.nw_affine_grad_double(a, b, one)
    s3, g3 = nwgrad.nw_affine_grad_double(a, b, three, track=track)
    assert s3 == s1
    # The three slot gradients partition the one-slot gradient exactly.
    slots = g3.to_dict()["matrices"]
    assert len(slots) == 3
    np.testing.assert_array_equal(sum(slots), g1.to_dict()["matrix"])


def test_untracked_result_is_unchanged_by_the_feature_existing():
    """A K == 1 call with no track at all must still match the independent DP."""
    mats = [_matrix(4)]
    a, b = _seq(40, 40), _seq(41, 37)
    got = nwgrad.nw_score_affine_double(a, b, _params(mats))
    assert got == pytest.approx(_nw_affine_tracked(a, b, mats, [0] * len(a)))


# ── 2. A tracked score really is the per-position model ──────────────────────


@pytest.mark.parametrize("nslots", [2, 3])
def test_score_matches_exhaustive_enumeration(nslots):
    mats = [_matrix(50 + k) for k in range(nslots)]
    a, b = _seq(60, 7), _seq(61, 6)
    rng = np.random.default_rng(7)
    track = [int(k) for k in rng.integers(0, nslots, size=len(a))]
    got = nwgrad.nw_score_affine_double(a, b, _params(mats), track=track)
    assert got == pytest.approx(_exhaustive(a, b, mats, track, affine=True))


def test_score_matches_exhaustive_linear_gaps():
    mats = [_matrix(70), _matrix(71)]
    a, b = _seq(72, 6), _seq(73, 6)
    track = [0, 1, 0, 1, 1, 0]
    got = nwgrad.nw_score_double(a, b, _params(mats), track=track)
    assert got == pytest.approx(_exhaustive(a, b, mats, track, affine=False))


def test_local_score_matches_exhaustive():
    """Smith-Waterman: the best global score over every pair of substrings, with the
    track sliced along with sequence A."""
    mats = [_matrix(80), _matrix(81)]
    a, b = _seq(82, 6), _seq(83, 6)
    track = [1, 0, 0, 1, 0, 1]
    best = 0.0
    for i0, i1 in itertools.combinations(range(len(a) + 1), 2):
        for j0, j1 in itertools.combinations(range(len(b) + 1), 2):
            best = max(best, _exhaustive(a[i0:i1], b[j0:j1], mats,
                                         track[i0:i1], affine=True))
    got = nwgrad.sw_score_affine_double(a, b, _params(mats), track=track)
    assert got == pytest.approx(best)


@pytest.mark.parametrize("length", [40, 120, 300])
def test_score_matches_the_independent_dp_at_length(length):
    """Long enough to leave the Hirschberg base case and actually split."""
    mats = [_matrix(90), _matrix(91), _matrix(92)]
    a, b = _seq(100 + length, length), _seq(200 + length, length - 7)
    rng = np.random.default_rng(length)
    track = [int(k) for k in rng.integers(0, 3, size=len(a))]
    got = nwgrad.nw_score_affine_double(a, b, _params(mats), track=track)
    assert got == pytest.approx(_nw_affine_tracked(a, b, mats, track))


def test_a_track_actually_changes_the_answer():
    """Liveness.  Without this the whole file could pass by ignoring the track."""
    mats = [_matrix(110), _matrix(111)]
    a, b = _seq(112, 50), _seq(113, 48)
    p = _params(mats)
    all_slot0 = nwgrad.nw_score_affine_double(a, b, p, track=[0] * len(a))
    all_slot1 = nwgrad.nw_score_affine_double(a, b, p, track=[1] * len(a))
    mixed = nwgrad.nw_score_affine_double(a, b, p, track=[i % 2 for i in range(len(a))])
    assert all_slot0 != all_slot1
    assert mixed not in (all_slot0, all_slot1)


# ── 3. The gradient lands in the right slot ──────────────────────────────────


def test_hard_grad_matches_exhaustive_per_slot_counts():
    mats = [_matrix(120), _matrix(121)]
    a, b = _seq(122, 7), _seq(123, 6)
    track = [0, 1, 1, 0, 1, 0, 1]
    best_path, best_score = _exhaustive_argmax(a, b, mats, track, affine=True)

    score, grad = nwgrad.nw_affine_grad_double(a, b, _params(mats), track=track)
    assert score == pytest.approx(best_score)

    want = _path_slot_counts(best_path, a, b, track, len(mats))
    got = grad.to_dict()["matrices"]
    assert len(got) == len(want)
    for k, (g, w) in enumerate(zip(got, want)):
        np.testing.assert_allclose(g, w, atol=1e-9, err_msg=f"slot {k}")


def test_soft_grad_matches_exhaustive_expected_per_slot_counts():
    mats = [_matrix(130), _matrix(131)]
    a, b = _seq(132, 6), _seq(133, 5)
    track = [1, 0, 1, 1, 0, 0]
    scored = [(p, _tracked_path_score(p, a, b, mats, track, affine=True))
              for p in all_paths(len(a), len(b))]
    hi = max(s for _, s in scored)
    weights = [math.exp(s - hi) for _, s in scored]
    z = sum(weights)
    want = [np.zeros((N, N)) for _ in mats]
    for (path, _), w in zip(scored, weights):
        for k, c in enumerate(_path_slot_counts(path, a, b, track, len(mats))):
            want[k] += c * w / z

    log_z, grad = nwgrad.nw_affine_soft_grad_double(a, b, _params(mats), track=track)
    assert log_z == pytest.approx(hi + math.log(z))
    for k, (g, w) in enumerate(zip(grad.to_dict()["matrices"], want)):
        np.testing.assert_allclose(g, w, atol=1e-9, err_msg=f"slot {k}")


def test_a_slot_no_position_selects_gets_a_zero_gradient():
    """Slot 2 is present in the params but absent from the track: its gradient must be
    identically zero, and the other two must be unaffected by its existence."""
    mats = [_matrix(140), _matrix(141), _matrix(142)]
    a, b = _seq(143, 30), _seq(144, 28)
    track = [i % 2 for i in range(len(a))]
    _, g3 = nwgrad.nw_affine_grad_double(a, b, _params(mats), track=track)
    slots = g3.to_dict()["matrices"]
    assert np.count_nonzero(slots[2]) == 0
    assert np.count_nonzero(slots[0]) > 0 and np.count_nonzero(slots[1]) > 0

    _, g2 = nwgrad.nw_affine_grad_double(a, b, _params(mats[:2]), track=track)
    for got, want in zip(slots[:2], g2.to_dict()["matrices"]):
        np.testing.assert_array_equal(got, want)


def test_hard_grad_matches_finite_differences_per_slot():
    """Perturb one entry of one slot and watch the score move by exactly the gradient.
    Independent of every oracle above, and the check that would catch a gradient
    credited to the wrong slot."""
    mats = [_matrix(150), _matrix(151)]
    a, b = _seq(152, 45), _seq(153, 42)
    rng = np.random.default_rng(3)
    track = [int(k) for k in rng.integers(0, 2, size=len(a))]
    p = _params(mats)
    base, grad = nwgrad.nw_affine_grad_double(a, b, p, track=track)
    slots = grad.to_dict()["matrices"]

    eps = 1e-6
    tested = 0
    for k in range(2):
        nz = np.argwhere(slots[k] > 0)
        for (r, c) in nz[:4]:
            arrs = [m.to_matrix() for m in mats]
            arrs[k][r, c] += eps
            bumped = _params([nwgrad.SubstMatrix(np.ascontiguousarray(x), ALPHA)
                              for x in arrs])
            moved = nwgrad.nw_score_affine_double(a, b, bumped, track=track)
            assert (moved - base) / eps == pytest.approx(slots[k][r, c], abs=1e-4)
            tested += 1
    assert tested >= 4, "fixture produced no non-zero gradient entries to perturb"


# ── The literal per-position reading: one matrix for every residue of A ──────


def test_one_matrix_per_position():
    """K == len(A) with track = range(len(A)) — nothing tied, a genuinely distinct
    substitution matrix at every position of sequence A."""
    a, b = _seq(160, 24), _seq(161, 22)
    mats = [_matrix(300 + i) for i in range(len(a))]
    track = list(range(len(a)))
    p = _params(mats)
    got = nwgrad.nw_score_affine_double(a, b, p, track=track)
    assert got == pytest.approx(_nw_affine_tracked(a, b, mats, track))

    _, grad = nwgrad.nw_affine_grad_double(a, b, p, track=track)
    slots = grad.to_dict()["matrices"]
    assert len(slots) == len(a)
    # Each position of A matches at most once, and each slot serves exactly one
    # position, so no slot can accumulate more than a single count.
    for k, s in enumerate(slots):
        assert s.sum() <= 1.0 + 1e-12, f"slot {k} used by more than one position"


def test_one_matrix_per_position_counts_match_the_alignment():
    a, b = _seq(170, 20), _seq(171, 18)
    mats = [_matrix(400 + i) for i in range(len(a))]
    p = _params(mats)
    sp = nwgrad.SeqPairDouble(a, b, p, track=list(range(len(a))))
    sp.alloc_dp()
    sp.align_full()
    sp.compute_grad()
    ga, gb = sp.aligned()
    n_match = sum(1 for x, y in zip(ga, gb) if x != "-" and y != "-")
    total = sum(s.sum() for s in sp.grad.to_dict()["matrices"])
    assert total == pytest.approx(n_match)


# ── 4. Every traceback mode, the banded path, the batch paths ────────────────


@pytest.mark.parametrize("traceback", ["pointers", "scores", "hirschberg",
                                       "hirschberg_pmax"])
def test_traceback_modes_agree_under_a_track(traceback):
    """Every mode must find the same optimum on a tracked problem.  hirschberg* may
    return a different optimal PATH at a tie, so this pins the SCORE (which none of them
    is allowed to get wrong) and the path's validity."""
    mats = [_matrix(180), _matrix(181), _matrix(182)]
    a, b = _seq(183, 250), _seq(184, 240)
    rng = np.random.default_rng(5)
    track = [int(k) for k in rng.integers(0, 3, size=len(a))]
    p = _params(mats)

    sp = nwgrad.SeqPairDouble(a, b, p, traceback=traceback, track=track)
    sp.hb_cutoff = 32               # force the recursion to actually split
    sp.alloc_dp()
    sp.align_full()
    assert sp.score == pytest.approx(_nw_affine_tracked(a, b, mats, track))

    ga, gb = sp.aligned()
    assert len(ga) == len(gb)
    assert ga.replace("-", "") == a and gb.replace("-", "") == b


def test_banded_realignment_respects_the_track():
    mats = [_matrix(190), _matrix(191)]
    a, b = _seq(192, 120), _seq(193, 118)
    rng = np.random.default_rng(6)
    track = [int(k) for k in rng.integers(0, 2, size=len(a))]
    sp = nwgrad.SeqPairDouble(a, b, _params(mats), track=track)
    sp.alloc_dp()
    sp.align_full()
    full = sp.score
    sp.realign_banded(len(a))       # a band this wide cannot exclude the optimum
    assert sp.score == pytest.approx(full)
    assert full == pytest.approx(_nw_affine_tracked(a, b, mats, track))


def test_seq_pair_reports_its_track():
    p = _params([_matrix(200), _matrix(201)])
    a = _seq(202, 12)
    track = [i % 2 for i in range(len(a))]
    assert list(nwgrad.SeqPair(a, _seq(203, 10), p, track=track).track) == track
    assert list(nwgrad.SeqPair(a, _seq(203, 10), p).track) == []


def test_batch_aligner_tracks():
    mats = [_matrix(210), _matrix(211)]
    p = _params(mats)
    seqs_a = [_seq(220 + i, 30 + i) for i in range(5)]
    seqs_b = [_seq(230 + i, 28 + i) for i in range(5)]
    rng = np.random.default_rng(8)
    tracks = [[int(k) for k in rng.integers(0, 2, size=len(s))] for s in seqs_a]

    ba = nwgrad.BatchAlignerDouble(p, n_threads=3)
    res = ba.align(seqs_a, seqs_b, tracks=tracks)
    for s, sa, sb, t in zip(res.scores, seqs_a, seqs_b, tracks):
        assert s == pytest.approx(_nw_affine_tracked(sa, sb, mats, t))

    # The batch gradient is the sum of the per-pair gradients, slot by slot.
    want = [np.zeros((N, N)) for _ in mats]
    for sa, sb, t in zip(seqs_a, seqs_b, tracks):
        _, g = nwgrad.nw_affine_grad_double(sa, sb, p, track=t)
        for k, arr in enumerate(g.to_dict()["matrices"]):
            want[k] += arr
    for k, got in enumerate(res.grad.to_dict()["matrices"]):
        np.testing.assert_allclose(got, want[k], atol=1e-9, err_msg=f"slot {k}")


def test_seq_pair_batch_add_many_tracks():
    mats = [_matrix(240), _matrix(241)]
    p = _params(mats)
    seqs_a = [_seq(250 + i, 40) for i in range(6)]
    seqs_b = [_seq(260 + i, 38) for i in range(6)]
    rng = np.random.default_rng(12)
    tracks = [[int(k) for k in rng.integers(0, 2, size=len(s))] for s in seqs_a]

    batch = nwgrad.SeqPairBatchDouble(n_threads=3)
    batch.add_many(seqs_a, seqs_b, p, tracks=tracks)
    batch.alloc_dp()
    batch.align_full()
    for i, (sa, sb, t) in enumerate(zip(seqs_a, seqs_b, tracks)):
        assert batch[i].score == pytest.approx(_nw_affine_tracked(sa, sb, mats, t))
        assert list(batch[i].track) == t

    grad = batch.compute_grad()
    assert len(grad.to_dict()["matrices"]) == 2


def test_batch_sums_gradients_over_pairs_with_different_tracks():
    """Pairs in one batch may carry DIFFERENT tracks — that is the point of a track —
    but they must agree on how many slots exist."""
    mats = [_matrix(270), _matrix(271)]
    p = _params(mats)
    a0, b0 = _seq(272, 25), _seq(273, 24)
    a1, b1 = _seq(274, 25), _seq(275, 24)
    batch = nwgrad.SeqPairBatchDouble(n_threads=2)
    batch.add_many([a0, a1], [b0, b1], p,
                   tracks=[[0] * len(a0), [1] * len(a1)])
    batch.alloc_dp()
    batch.align_full()
    assert batch[0].score == pytest.approx(_nw_affine_tracked(a0, b0, mats, [0] * len(a0)))
    assert batch[1].score == pytest.approx(_nw_affine_tracked(a1, b1, mats, [1] * len(a1)))


# ── AlignParams: the stack itself ────────────────────────────────────────────


def test_slot_accessors():
    m0, m1 = _matrix(280), _matrix(281)
    p = nwgrad.AlignParams(m0, GO_A, GE_A, GO_B, GE_B)
    assert p.matrix_count == 1
    assert p.add_matrix(m1) == 1
    assert p.matrix_count == 2
    np.testing.assert_array_equal(p.matrix_at(0).to_matrix(), m0.to_matrix())
    np.testing.assert_array_equal(p.matrix_at(1).to_matrix(), m1.to_matrix())
    np.testing.assert_array_equal(p.matrices[1].to_matrix(), m1.to_matrix())
    # matrices is a list of copies; writing to it must not touch the params.
    p.matrices[1] = m0
    np.testing.assert_array_equal(p.matrix_at(1).to_matrix(), m1.to_matrix())
    p.set_matrix_at(1, m0)
    np.testing.assert_array_equal(p.matrix_at(1).to_matrix(), m0.to_matrix())


def test_arithmetic_is_elementwise_over_every_slot():
    m0, m1 = _matrix(290), _matrix(291)
    p = _params([m0, m1])
    doubled = p + p
    for k, arr in enumerate(doubled.to_dict()["matrices"]):
        np.testing.assert_allclose(arr, 2.0 * [m0, m1][k].to_matrix())
    assert doubled.gap_open_a == pytest.approx(2 * GO_A)
    scaled = p * -1.0
    for k, arr in enumerate(scaled.to_dict()["matrices"]):
        np.testing.assert_allclose(arr, -[m0, m1][k].to_matrix())


# ── Errors: every one of these used to be a silently wrong answer ────────────


def test_track_length_must_match_sequence_a():
    p = _params([_matrix(1), _matrix(2)])
    with pytest.raises(Exception, match="track"):
        nwgrad.nw_score_affine_double("ACDE", "ACDE", p, track=[0, 1, 0])
    with pytest.raises(Exception, match="track"):
        nwgrad.SeqPair("ACDE", "ACDE", p, track=[0, 1, 0])


def test_track_entry_must_name_an_existing_slot():
    p = _params([_matrix(1), _matrix(2)])
    with pytest.raises(Exception, match="slot"):
        nwgrad.nw_score_affine_double("ACDE", "ACDE", p, track=[0, 1, 2, 0])
    with pytest.raises(Exception, match="slot"):
        nwgrad.nw_score_affine_double("ACDE", "ACDE", p, track=[0, -1, 1, 0])


def test_slots_must_share_an_alphabet():
    p = nwgrad.AlignParams(_matrix(1), GO_A, GE_A, GO_B, GE_B)
    dna = _matrix(2, "ACGT")
    with pytest.raises(Exception, match="alphabet"):
        p.add_matrix(dna)


def test_combining_different_slot_counts_raises():
    one = _params([_matrix(1)])
    two = _params([_matrix(1), _matrix(2)])
    with pytest.raises(Exception, match="slot"):
        one + two


def test_set_params_cannot_change_the_slot_count():
    one = _params([_matrix(1)])
    two = _params([_matrix(1), _matrix(2)])
    sp = nwgrad.SeqPair("ACDEFG", "ACDEFG", two, track=[0, 1, 0, 1, 0, 1])
    with pytest.raises(Exception, match="slot"):
        sp.set_params(one)


def test_batch_rejects_pairs_with_different_slot_counts():
    one = _params([_matrix(1)])
    two = _params([_matrix(1), _matrix(2)])
    batch = nwgrad.SeqPairBatch(n_threads=1)
    a, b = _seq(500, 10), _seq(501, 10)
    sp1 = nwgrad.SeqPair(a, b, one)
    sp2 = nwgrad.SeqPair(a, b, two)
    batch.add(sp1)
    with pytest.raises(Exception, match="slot"):
        batch.add(sp2)


def test_empty_matrix_list_raises():
    with pytest.raises(Exception):
        nwgrad.AlignParams([], GO_A, GE_A, GO_B, GE_B)
