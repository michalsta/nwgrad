"""A one-cell oracle for local soft gradients, including all four gap fields."""

import numpy as np
import pytest
import nwgrad


@pytest.mark.parametrize("suffix", ["", "_double"], ids=["float32", "double"])
@pytest.mark.parametrize("gap_model", ["linear", "affine"])
@pytest.mark.parametrize("field", ["gap_open_a", "gap_extend_a",
                                   "gap_open_b", "gap_extend_b"])
def test_one_cell_local_gap_derivative(suffix, gap_model, field):
    fn = getattr(nwgrad, ("sw_soft_grad" if gap_model == "linear"
                         else "sw_affine_soft_grad") + suffix)
    gaps = dict(gap_open_a=2.0, gap_extend_a=1.0,
                gap_open_b=1.5, gap_extend_b=0.5)
    matrix = np.array([[2.0]])
    score, grad = fn("A", "A", nwgrad.AlignParams(matrix, alphabet="A", **gaps))
    # The local forward recurrence has four unit restart/border terms, one
    # match, and one interior step for each gap direction. Border cells are
    # constants, so there are no paid transitions along those borders.
    cost_a, cost_b = gaps["gap_extend_a"], gaps["gap_extend_b"]
    if gap_model == "affine":
        cost_a += gaps["gap_open_a"]
        cost_b += gaps["gap_open_b"]
    z = 4 + np.exp(2.0) + np.exp(-cost_a) + np.exp(-cost_b)
    assert score == pytest.approx(np.log(z), abs=1e-12)
    expected = -np.exp(-cost_a if field.endswith("_a") else -cost_b) / z
    if gap_model == "linear" and "open" in field:
        expected = 0.0
    step = 1e-5
    plus, minus = gaps.copy(), gaps.copy()
    plus[field] += step
    minus[field] -= step
    numerical = (fn("A", "A", nwgrad.AlignParams(matrix, alphabet="A", **plus))[0]
                 - fn("A", "A", nwgrad.AlignParams(matrix, alphabet="A", **minus))[0]) / (2 * step)
    assert numerical == pytest.approx(expected, abs=2e-10)
    assert getattr(grad, field) == pytest.approx(expected, abs=1e-12)


FIELDS = ["gap_open_a", "gap_extend_a", "gap_open_b", "gap_extend_b"]


def _soft_fn(gap_model, mode, suffix):
    prefix = "nw" if mode == "global" else "sw"
    name = f"{prefix}_soft_grad" if gap_model == "linear" else f"{prefix}_affine_soft_grad"
    return getattr(nwgrad, name + suffix)


@pytest.mark.parametrize("suffix", ["", "_double"], ids=["float32", "double"])
@pytest.mark.parametrize("gap_model", ["linear", "affine"])
@pytest.mark.parametrize("mode", ["global", "local"])
@pytest.mark.parametrize("band", [0, 2], ids=["full", "banded"])
def test_soft_gap_gradient_matches_finite_difference(suffix, gap_model, mode, band):
    # Multi-cell problems, so interior, border-adjacent and (banded) band-edge
    # transitions all contribute; asymmetric gaps catch a crossed field.
    rng = np.random.default_rng(7)
    matrix = rng.normal(0.0, 1.0, (4, 4)) + 2.0 * np.eye(4)
    gaps = dict(gap_open_a=1.25, gap_extend_a=0.75, gap_open_b=0.5, gap_extend_b=1.1)
    fn = _soft_fn(gap_model, mode, suffix)
    for a, b in [("ACGTTGCA", "ACGTGCA"), ("GATTACA", "TTGATTACAGG"), ("CA", "TCAG")]:
        _, grad = fn(a, b, nwgrad.AlignParams(matrix, alphabet="ACGT", **gaps), band=band)
        step = 1e-5
        for field in FIELDS:
            plus, minus = gaps.copy(), gaps.copy()
            plus[field] += step
            minus[field] -= step
            numerical = (fn(a, b, nwgrad.AlignParams(matrix, alphabet="ACGT", **plus), band=band)[0]
                         - fn(a, b, nwgrad.AlignParams(matrix, alphabet="ACGT", **minus), band=band)[0]
                         ) / (2 * step)
            assert getattr(grad, field) == pytest.approx(numerical, abs=1e-8), (a, b, field)


@pytest.mark.parametrize("suffix", ["", "_double"], ids=["float32", "double"])
@pytest.mark.parametrize("gap_model", ["linear", "affine"])
@pytest.mark.parametrize("a,b", [("", "ACGT"), ("ACGT", ""), ("", "")])
def test_local_soft_gap_gradient_is_zero_on_empty_input(suffix, gap_model, a, b):
    # With an empty side there is no interior cell, so no gap move exists: log Z
    # counts only the constant border cells and is independent of every gap cost.
    fn = _soft_fn(gap_model, "local", suffix)
    params = nwgrad.AlignParams(2.0 * np.eye(4) - 1.0, alphabet="ACGT",
                                gap_open_a=2.0, gap_extend_a=1.0,
                                gap_open_b=1.5, gap_extend_b=0.5)
    score, grad = fn(a, b, params)
    assert score == pytest.approx(np.log(len(a) + len(b) + 1), abs=1e-12)
    for field in FIELDS:
        assert getattr(grad, field) == 0.0
