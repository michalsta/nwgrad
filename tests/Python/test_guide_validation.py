"""Malformed external guides must be rejected before DP indexes their rows."""

import pytest

from conftest import describe, run_isolated


@pytest.mark.parametrize("precision", ["", "Double"], ids=["float32", "double"])
@pytest.mark.parametrize("entry", ["convenience", "batch"])
@pytest.mark.parametrize("grad_mode", ["hard", "soft"])
def test_short_guide_raises_value_error(precision, entry, grad_mode):
    # Both aligned strings are well formed, but describe only one of the four
    # input residues. The resulting two-element guide is indexed through row 4.
    suffix = "_double" if precision else ""
    function = "nw_affine_grad" if grad_mode == "hard" else "nw_affine_soft_grad"
    proc = run_isolated(f"""
import nwgrad as n
import numpy as np
p = n.AlignParams(3 * np.eye(4) - 1, alphabet="ACGT",
                  gap_open_a=2, gap_extend_a=1, gap_open_b=2, gap_extend_b=1)
try:
    if {entry!r} == "convenience":
        n.{function}{suffix}("ACGT", "ACGT", p, band=1,
                             aligned_a="A", aligned_b="A")
    else:
        batch = n.SeqPairBatch{precision}(1, grad_mode={grad_mode!r})
        batch.align(["ACGT"], ["ACGT"], p, band=1, aligned_a=["A"], aligned_b=["A"])
except ValueError:
    pass
else:
    raise AssertionError("a guide shorter than the input sequence was accepted")
""")
    assert proc.returncode == 0, describe(proc)
