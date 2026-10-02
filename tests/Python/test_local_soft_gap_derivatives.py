"""A one-cell oracle for local soft gradients, including all four gap fields."""

import numpy as np
import pytest
import nwgrad


@pytest.mark.parametrize("suffix", ["", "_double"], ids=["float32", "double"])
@pytest.mark.parametrize("gap_model", ["linear", "affine"])
@pytest.mark.parametrize("field", ["gap_open_a", "gap_extend_a",
                                   "gap_open_b", "gap_extend_b"])
def test_one_cell_local_gap_derivative(suffix, gap_model, field, request):
    broken = ((gap_model == "linear" and "extend" in field)
              or (gap_model == "affine" and "open" in field))
    if broken:
        request.node.add_marker(pytest.mark.xfail(
            strict=True, reason="local soft gap gradients count nonexistent border transitions"))
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
