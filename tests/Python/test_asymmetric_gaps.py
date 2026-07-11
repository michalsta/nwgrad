"""Asymmetric gap penalties: gap_open_a/gap_extend_a vs gap_open_b/gap_extend_b.

The rest of the suite always passes the same gap costs for both sequences, so a
regression that swapped the "a" and "b" parameters — or ignored one side — would
go unnoticed.  Biopython cannot serve as the oracle here because it only exposes
a single gap cost for both sequences, so these tests check against `bruteforce`,
an independent exhaustive enumerator, plus a transpose-invariance property.

Convention under test (see aligner.hpp):
  - the "a" penalties charge gaps in A (a '-' in A, consuming B)
  - the "b" penalties charge gaps in B (a '-' in B, consuming A)
"""

import random

import numpy as np
import pytest

import nwgrad

import bruteforce as bf

ALPHABET = "ACGT"


@pytest.fixture(scope="module")
def matrix():
    return nwgrad.SubstMatrix(np.eye(4) * 2 - 1, alphabet=ALPHABET)


@pytest.fixture(scope="module")
def generic_matrix():
    """An asymmetric matrix with no two entries equal.

    The plain match/mismatch matrix ties a lot of alignments on short sequences,
    and a tie makes the hard subgradient a free choice among optima — untestable.
    Irregular entries make the optimum unique almost always.
    """
    rng = np.random.default_rng(20240711)
    return nwgrad.SubstMatrix(rng.normal(size=(4, 4)) * 2.0, alphabet=ALPHABET)


def params(matrix, gap_open_a, gap_extend_a, gap_open_b, gap_extend_b):
    return nwgrad.AlignParams(matrix,
                              gap_open_a=gap_open_a, gap_extend_a=gap_extend_a,
                              gap_open_b=gap_open_b, gap_extend_b=gap_extend_b)


def _corpus(n=25, seed=11):
    """Short random sequence pairs with lopsided gap costs.

    Kept tiny: the oracle enumerates every alignment path.
    """
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        a = "".join(rng.choice(ALPHABET) for _ in range(rng.randint(1, 5)))
        b = "".join(rng.choice(ALPHABET) for _ in range(rng.randint(1, 5)))
        out.append((a, b,
                    rng.choice([0.5, 2.0, 4.0]), rng.choice([0.5, 1.0, 2.0]),
                    rng.choice([0.5, 2.0, 4.0]), rng.choice([0.5, 1.0, 2.0])))
    return out


CORPUS = _corpus()


# ── Scores against the exhaustive oracle ─────────────────────────────────────

@pytest.mark.parametrize("a,b,oa,ea,ob,eb", CORPUS)
def test_global_affine_matches_bruteforce(matrix, a, b, oa, ea, ob, eb):
    p = params(matrix, oa, ea, ob, eb)
    assert nwgrad.nw_score_affine(a, b, p) == pytest.approx(
        bf.global_score(a, b, p, affine=True))


@pytest.mark.parametrize("a,b,oa,ea,ob,eb", CORPUS)
def test_local_affine_matches_bruteforce(matrix, a, b, oa, ea, ob, eb):
    p = params(matrix, oa, ea, ob, eb)
    assert nwgrad.sw_score_affine(a, b, p) == pytest.approx(
        bf.local_score(a, b, p, affine=True))


@pytest.mark.parametrize("a,b,oa,ea,ob,eb", CORPUS)
def test_global_linear_matches_bruteforce(matrix, a, b, oa, ea, ob, eb):
    p = params(matrix, 0.0, ea, 0.0, eb)  # linear ignores the open penalties
    assert nwgrad.nw_score(a, b, p) == pytest.approx(
        bf.global_score(a, b, p, affine=False))


@pytest.mark.parametrize("a,b,oa,ea,ob,eb", CORPUS)
def test_local_linear_matches_bruteforce(matrix, a, b, oa, ea, ob, eb):
    p = params(matrix, 0.0, ea, 0.0, eb)
    assert nwgrad.sw_score(a, b, p) == pytest.approx(
        bf.local_score(a, b, p, affine=False))


@pytest.mark.parametrize("a,b,oa,ea,ob,eb", CORPUS)
def test_log_partition_matches_bruteforce(matrix, a, b, oa, ea, ob, eb):
    p = params(matrix, oa, ea, ob, eb)
    log_z, _ = nwgrad.nw_affine_soft_grad(a, b, p)
    assert log_z == pytest.approx(bf.global_log_z(a, b, p, affine=True))


# ── Transpose invariance ─────────────────────────────────────────────────────
# Swapping the sequences and simultaneously swapping the a/b penalties must
# leave the score unchanged.  This is the property that fails loudly if the two
# parameter sets are ever crossed.

@pytest.mark.parametrize("a,b,oa,ea,ob,eb", CORPUS)
@pytest.mark.parametrize("score_fn", [nwgrad.nw_score_affine, nwgrad.sw_score_affine])
def test_swapping_sequences_and_penalties_preserves_score(
        matrix, score_fn, a, b, oa, ea, ob, eb):
    forward = score_fn(a, b, params(matrix, oa, ea, ob, eb))
    mirrored = score_fn(b, a, params(matrix, ob, eb, oa, ea))
    assert forward == pytest.approx(mirrored)


def test_crossed_penalties_actually_change_the_score(matrix):
    """Guards the test above from being vacuous.

    If a and b were interchangeable, swapping only the penalties would be a
    no-op and transpose invariance would hold trivially.
    """
    a, b = "ACGTACGTAAACGT", "ACGTTTTTACGT"
    a_heavy = nwgrad.nw_score_affine(a, b, params(matrix, 10.0, 1.0, 1.0, 1.0))
    b_heavy = nwgrad.nw_score_affine(a, b, params(matrix, 1.0, 1.0, 10.0, 1.0))
    assert a_heavy != pytest.approx(b_heavy)


# ── Each parameter drives the side it claims to ──────────────────────────────

def test_a_penalties_suppress_gaps_in_a(matrix):
    a, b = "AC", "ACGTGT"  # b is longer, so alignment needs gaps in a
    cheap = nwgrad.SeqPair(a, b, params(matrix, 0.5, 0.5, 20.0, 20.0),
                           gap_model="affine", mode="global", grad_mode="none")
    cheap.alloc_dp()
    cheap.align_full()
    aligned_a, _ = cheap.aligned()
    assert "-" in aligned_a  # gaps land in a, which is where they are cheap


def test_b_penalties_suppress_gaps_in_b(matrix):
    a, b = "ACGTGT", "AC"  # mirror image: now a is longer
    cheap = nwgrad.SeqPair(a, b, params(matrix, 20.0, 20.0, 0.5, 0.5),
                           gap_model="affine", mode="global", grad_mode="none")
    cheap.alloc_dp()
    cheap.align_full()
    _, aligned_b = cheap.aligned()
    assert "-" in aligned_b


@pytest.mark.parametrize("gap_open_b", [0.0, 50.0])
def test_b_open_penalty_is_not_ignored(matrix, gap_open_b):
    """A free vs a punitive gap-open on the b side must not score the same."""
    p = params(matrix, 1.0, 1.0, gap_open_b, 1.0)
    score = nwgrad.nw_score_affine("ACGTACGT", "AGTACT", p)
    if gap_open_b == 0.0:
        assert score > 0
    else:
        assert score < -40


# ── Gradients ────────────────────────────────────────────────────────────────

def _grad_arrays(grad):
    d = grad.to_dict()
    return (d["matrix"], d["gap_open_a"], d["gap_extend_a"],
            d["gap_open_b"], d["gap_extend_b"])


def _pairs_to_matrix(pairs):
    m = np.zeros((len(ALPHABET), len(ALPHABET)))
    for (x, y), v in pairs.items():
        m[ALPHABET.index(x), ALPHABET.index(y)] += v
    return m


def _optimum_is_unique(a, b, p, affine):
    """Ties make the hard subgradient implementation-defined; skip those cases."""
    scores = [s for _, s in bf._scored_paths(a, b, p, affine)]
    best = max(scores)
    return sum(1 for s in scores if s > best - 1e-9) == 1


@pytest.mark.parametrize("a,b,oa,ea,ob,eb", CORPUS)
def test_hard_gradient_counts_match_bruteforce(generic_matrix, a, b, oa, ea, ob, eb):
    p = params(generic_matrix, oa, ea, ob, eb)
    if not _optimum_is_unique(a, b, p, affine=True):
        pytest.skip("optimal alignment is not unique; subgradient is a free choice")

    _, grad = nwgrad.nw_affine_grad(a, b, p)
    got_matrix, got_oa, got_ea, got_ob, got_eb = _grad_arrays(grad)
    pairs, exp_oa, exp_ea, exp_ob, exp_eb = bf.global_hard_counts(a, b, p, affine=True)

    np.testing.assert_allclose(got_matrix, _pairs_to_matrix(pairs), atol=1e-9)
    assert (got_oa, got_ea, got_ob, got_eb) == pytest.approx(
        (exp_oa, exp_ea, exp_ob, exp_eb))


@pytest.mark.parametrize("a,b,oa,ea,ob,eb", CORPUS)
def test_soft_gradient_expected_counts_match_bruteforce(
        generic_matrix, a, b, oa, ea, ob, eb):
    # asymmetric matrix *and* asymmetric gaps at once
    p = params(generic_matrix, oa, ea, ob, eb)
    _, grad = nwgrad.nw_affine_soft_grad(a, b, p)
    got_matrix, got_oa, got_ea, got_ob, got_eb = _grad_arrays(grad)
    pairs, exp_oa, exp_ea, exp_ob, exp_eb = bf.global_soft_counts(a, b, p, affine=True)

    np.testing.assert_allclose(got_matrix, _pairs_to_matrix(pairs), atol=1e-9)
    assert (got_oa, got_ea, got_ob, got_eb) == pytest.approx(
        (exp_oa, exp_ea, exp_ob, exp_eb))


@pytest.mark.parametrize("gap_param", ["gap_open_a", "gap_extend_a",
                                       "gap_open_b", "gap_extend_b"])
def test_gap_gradient_is_a_count_not_a_score_derivative(matrix, gap_param):
    """Pins the sign convention of the gap components of the gradient.

    The matrix component is a true derivative of the score, but the four gap
    components are reported as *counts* — i.e. the negation of d(score)/d(param),
    because the score subtracts the penalties.  The two halves of AlignParams
    therefore carry opposite signs, which matters to anyone writing the update
    loop that align_params.hpp advertises.  This test exists to make any change
    to that convention deliberate rather than silent.
    """
    a, b = "ACGTACGT", "AGTACT"
    base = dict(gap_open_a=5.0, gap_extend_a=1.0, gap_open_b=5.0, gap_extend_b=1.0)

    score, grad = nwgrad.nw_affine_grad(a, b, params(matrix, **base))

    eps = 1e-6
    bumped = dict(base)
    bumped[gap_param] += eps
    score_eps = nwgrad.nw_score_affine(a, b, params(matrix, **bumped))
    d_score = (score_eps - score) / eps

    assert grad.to_dict()[gap_param] == pytest.approx(-d_score, abs=1e-4)


def test_matrix_gradient_is_a_true_score_derivative(matrix):
    """Counterpart to the test above: the matrix half has the opposite sign."""
    a, b = "ACGTACGT", "AGTACT"
    kw = dict(gap_open_a=5.0, gap_extend_a=1.0, gap_open_b=5.0, gap_extend_b=1.0)
    score, grad = nwgrad.nw_affine_grad(a, b, params(matrix, **kw))

    eps = 1e-6
    bumped_arr = matrix.to_matrix().copy()
    bumped_arr[0, 0] += eps
    bumped = nwgrad.AlignParams(
        nwgrad.SubstMatrix(bumped_arr, alphabet=ALPHABET), **kw)
    d_score = (nwgrad.nw_score_affine(a, b, bumped) - score) / eps

    assert grad.to_dict()["matrix"][0, 0] == pytest.approx(d_score, abs=1e-4)


# ── Asymmetric penalties survive the higher-level entry points ───────────────

@pytest.mark.parametrize("mode", ["global", "local"])
@pytest.mark.parametrize("gap_model", ["linear", "affine"])
@pytest.mark.parametrize("grad_mode", ["hard", "soft", "none"])
def test_seq_pair_agrees_with_single_pair_functions(matrix, mode, gap_model, grad_mode):
    a, b = "ACGTACGTAA", "AGTACTG"
    p = params(matrix, 4.0, 0.5, 1.5, 2.0)  # deliberately lopsided

    sp = nwgrad.SeqPair(a, b, p, gap_model=gap_model, mode=mode, grad_mode=grad_mode)
    sp.alloc_dp()
    sp.align_full()

    fn = {
        ("global", "linear"): nwgrad.nw_score, ("local", "linear"): nwgrad.sw_score,
        ("global", "affine"): nwgrad.nw_score_affine,
        ("local", "affine"): nwgrad.sw_score_affine,
    }[(mode, gap_model)]
    soft_fn = {
        ("global", "linear"): nwgrad.nw_soft_grad,
        ("local", "linear"): nwgrad.sw_soft_grad,
        ("global", "affine"): nwgrad.nw_affine_soft_grad,
        ("local", "affine"): nwgrad.sw_affine_soft_grad,
    }[(mode, gap_model)]

    # in soft mode SeqPair.score is the log-partition function, not the Viterbi score
    expected = soft_fn(a, b, p)[0] if grad_mode == "soft" else fn(a, b, p)
    assert sp.score == pytest.approx(expected)


@pytest.mark.parametrize("grad_mode", ["hard", "soft"])
def test_batch_aligner_accumulates_asymmetric_gap_gradients(matrix, grad_mode):
    pairs = [("ACGTACGTAA", "AGTACTG"), ("AC", "ACGTGT"), ("ACGTGT", "AC")]
    p = params(matrix, 4.0, 0.5, 1.5, 2.0)

    batch = nwgrad.BatchAligner(p, gap_model="affine", mode="global",
                                grad_mode=grad_mode, n_threads=2)
    result = batch.align([a for a, _ in pairs], [b for _, b in pairs])

    single = {"hard": nwgrad.nw_affine_grad, "soft": nwgrad.nw_affine_soft_grad}[grad_mode]
    per_pair = [single(a, b, p) for a, b in pairs]

    np.testing.assert_allclose(result.scores, [s for s, _ in per_pair], atol=1e-9)

    got = result.grad.to_dict()
    for key in ("gap_open_a", "gap_extend_a", "gap_open_b", "gap_extend_b"):
        expected = sum(g.to_dict()[key] for _, g in per_pair)
        assert got[key] == pytest.approx(expected), key
    expected_matrix = sum(g.to_dict()["matrix"] for _, g in per_pair)
    np.testing.assert_allclose(got["matrix"], expected_matrix, atol=1e-9)
