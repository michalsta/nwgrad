"""Arithmetic on AlignParams: +, +=, -, -=, unary -, *, *=, and scalar * params.

align_params.hpp advertises these operators as the gradient-descent update API
("params = params - learning_rate * grad") and defaults every field to zero so an
AlignParams can serve as a gradient accumulator.  Nothing else in the suite
exercised them, so the whole update path a training loop depends on was unchecked.

Every operator must act element-wise across the 256x256 substitution table *and*
all four gap fields, and must carry the alphabet through to the result.
"""

import numpy as np
import pytest

import nwgrad

ALPHABET = "ACGT"

GAP_FIELDS = ("gap_open_a", "gap_extend_a", "gap_open_b", "gap_extend_b")


def make(seed, **gaps):
    rng = np.random.default_rng(seed)
    arr = rng.normal(size=(4, 4))
    defaults = dict(gap_open_a=11.0, gap_extend_a=1.0,
                    gap_open_b=7.0, gap_extend_b=0.5)
    defaults.update(gaps)
    return nwgrad.AlignParams(
        nwgrad.SubstMatrix(arr, alphabet=ALPHABET), **defaults)


def parts(p):
    d = p.to_dict()
    return d["matrix"], np.array([d[f] for f in GAP_FIELDS])


def assert_equals(got, want_matrix, want_gaps, alphabet=ALPHABET):
    got_matrix, got_gaps = parts(got)
    np.testing.assert_allclose(got_matrix, want_matrix, atol=1e-12)
    np.testing.assert_allclose(got_gaps, want_gaps, atol=1e-12)
    assert got.to_dict()["alphabet"] == alphabet


# ── Binary operators ─────────────────────────────────────────────────────────

def test_add():
    x, y = make(0), make(1, gap_open_a=2.0, gap_extend_a=3.0,
                         gap_open_b=4.0, gap_extend_b=5.0)
    xm, xg = parts(x)
    ym, yg = parts(y)
    assert_equals(x + y, xm + ym, xg + yg)


def test_sub():
    x, y = make(0), make(1, gap_open_a=2.0, gap_extend_a=3.0,
                         gap_open_b=4.0, gap_extend_b=5.0)
    xm, xg = parts(x)
    ym, yg = parts(y)
    assert_equals(x - y, xm - ym, xg - yg)


def test_mul_by_scalar():
    x = make(0)
    xm, xg = parts(x)
    assert_equals(x * 2.5, xm * 2.5, xg * 2.5)


def test_rmul_scalar_on_the_left():
    """`learning_rate * grad` — the form the header's update rule actually uses."""
    x = make(0)
    xm, xg = parts(x)
    assert_equals(2.5 * x, xm * 2.5, xg * 2.5)


def test_mul_and_rmul_agree():
    x = make(0)
    lhs_m, lhs_g = parts(x * 0.1)
    rhs_m, rhs_g = parts(0.1 * x)
    np.testing.assert_allclose(lhs_m, rhs_m, atol=1e-12)
    np.testing.assert_allclose(lhs_g, rhs_g, atol=1e-12)


def test_neg():
    x = make(0)
    xm, xg = parts(x)
    assert_equals(-x, -xm, -xg)


def test_neg_is_multiplication_by_minus_one():
    x = make(0)
    neg_m, neg_g = parts(-x)
    mul_m, mul_g = parts(x * -1.0)
    np.testing.assert_allclose(neg_m, mul_m, atol=1e-12)
    np.testing.assert_allclose(neg_g, mul_g, atol=1e-12)


# ── In-place operators ───────────────────────────────────────────────────────

def test_iadd_mutates_in_place():
    x, y = make(0), make(1)
    xm, xg = parts(x)
    ym, yg = parts(y)
    before = x
    x += y
    assert x is before  # true in-place, not a rebind to a fresh object
    assert_equals(x, xm + ym, xg + yg)


def test_isub_mutates_in_place():
    x, y = make(0), make(1)
    xm, xg = parts(x)
    ym, yg = parts(y)
    before = x
    x -= y
    assert x is before
    assert_equals(x, xm - ym, xg - yg)


def test_imul_mutates_in_place():
    x = make(0)
    xm, xg = parts(x)
    before = x
    x *= 3.0
    assert x is before
    assert_equals(x, xm * 3.0, xg * 3.0)


# ── Aliasing: the out-of-place operators must not touch their operands ───────

@pytest.mark.parametrize("op", [
    lambda x, y: x + y,
    lambda x, y: x - y,
    lambda x, y: x * 2.0,
    lambda x, y: 2.0 * x,
    lambda x, y: -x,
])
def test_out_of_place_operators_leave_operands_untouched(op):
    x, y = make(0), make(1)
    xm, xg = parts(x)
    ym, yg = parts(y)

    result = op(x, y)
    assert result is not x and result is not y

    assert_equals(x, xm, xg)
    assert_equals(y, ym, yg)


# ── Algebraic identities ─────────────────────────────────────────────────────

def test_zero_params_is_an_additive_identity():
    """The documented use of AlignParams as a zero-initialised accumulator."""
    zero = nwgrad.AlignParams(nwgrad.SubstMatrix(np.zeros((4, 4)), alphabet=ALPHABET))
    zg = parts(zero)[1]
    np.testing.assert_allclose(zg, np.zeros(4), atol=1e-12)

    x = make(0)
    xm, xg = parts(x)
    assert_equals(x + zero, xm, xg)


def test_subtracting_self_yields_zero():
    x = make(0)
    assert_equals(x - x, np.zeros((4, 4)), np.zeros(4))


def test_addition_is_commutative():
    x, y = make(0), make(1)
    lhs_m, lhs_g = parts(x + y)
    rhs_m, rhs_g = parts(y + x)
    np.testing.assert_allclose(lhs_m, rhs_m, atol=1e-12)
    np.testing.assert_allclose(lhs_g, rhs_g, atol=1e-12)


def test_scalar_multiplication_distributes_over_addition():
    x, y = make(0), make(1)
    lhs_m, lhs_g = parts((x + y) * 2.0)
    rhs_m, rhs_g = parts(x * 2.0 + y * 2.0)
    np.testing.assert_allclose(lhs_m, rhs_m, atol=1e-12)
    np.testing.assert_allclose(lhs_g, rhs_g, atol=1e-12)


# ── The advertised update rule ───────────────────────────────────────────────

def test_gradient_descent_step_updates_matrix_and_gaps_together():
    """`params = params - learning_rate * grad`, verbatim from align_params.hpp."""
    matrix = nwgrad.SubstMatrix(np.eye(4) * 2 - 1, alphabet=ALPHABET)
    p = nwgrad.AlignParams(matrix, gap_open_a=5.0, gap_extend_a=1.0,
                           gap_open_b=5.0, gap_extend_b=1.0)
    before_matrix, before_gaps = parts(p)

    _, grad = nwgrad.nw_affine_grad("ACGTACGT", "AGTACT", p)
    grad_matrix, grad_gaps = parts(grad)

    learning_rate = 0.1
    stepped = p - learning_rate * grad

    assert_equals(stepped,
                  before_matrix - learning_rate * grad_matrix,
                  before_gaps - learning_rate * grad_gaps)


def test_accumulating_gradients_over_a_batch_matches_summing_them():
    """The `+=` accumulator pattern, against a plain sum of per-pair gradients."""
    matrix = nwgrad.SubstMatrix(np.eye(4) * 2 - 1, alphabet=ALPHABET)
    p = nwgrad.AlignParams(matrix, gap_open_a=4.0, gap_extend_a=0.5,
                           gap_open_b=1.5, gap_extend_b=2.0)
    pairs = [("ACGTACGTAA", "AGTACTG"), ("AC", "ACGTGT"), ("ACGTGT", "AC")]

    total = nwgrad.AlignParams(nwgrad.SubstMatrix(np.zeros((4, 4)), alphabet=ALPHABET))
    for a, b in pairs:
        total += nwgrad.nw_affine_grad(a, b, p)[1]

    grads = [nwgrad.nw_affine_grad(a, b, p)[1] for a, b in pairs]
    expected_matrix = sum(parts(g)[0] for g in grads)
    expected_gaps = sum(parts(g)[1] for g in grads)

    assert_equals(total, expected_matrix, expected_gaps)


def test_result_of_arithmetic_is_usable_for_alignment():
    """A stepped AlignParams must still align — i.e. the alphabet survived."""
    matrix = nwgrad.SubstMatrix(np.eye(4) * 2 - 1, alphabet=ALPHABET)
    p = nwgrad.AlignParams(matrix, gap_open_a=5.0, gap_extend_a=1.0,
                           gap_open_b=5.0, gap_extend_b=1.0)

    _, grad = nwgrad.nw_affine_grad("ACGTACGT", "AGTACT", p)
    stepped = p - 0.01 * grad

    assert stepped.matrix.alphabet == ALPHABET
    score = nwgrad.nw_score_affine("ACGTACGT", "AGTACT", stepped)
    assert np.isfinite(score)
