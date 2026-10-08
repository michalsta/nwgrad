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


@pytest.mark.parametrize("cls", ["SeqPairBatch", "SeqPairBatchDouble"])
def test_align_validates_each_guide_pair(cls):
    """align()'s gapped strings go through guide_j_from_aligned, as BatchAligner.align's
    did: one empty string against a non-empty one is a caller error, not "no guide"
    (which would silently band around the diagonal instead)."""
    import numpy as np
    import nwgrad
    p = nwgrad.AlignParams(3 * np.eye(4) - 1, alphabet="ACGT", gap_open_a=2, gap_extend_a=1,
                           gap_open_b=2, gap_extend_b=1)
    batch = getattr(nwgrad, cls)(1, grad_mode="hard")
    with pytest.raises(ValueError):
        batch.align(["ACGT"], ["ACGT"], p, band=2, aligned_a=["AC-GT"], aligned_b=[""])



def _dna_params(nwgrad):
    import numpy as np
    return nwgrad.AlignParams(3 * np.eye(4) - 1, alphabet="ACGT", gap_open_a=2, gap_extend_a=1,
                              gap_open_b=2, gap_extend_b=1)


@pytest.mark.parametrize("which", ["a", "b"])
def test_convenience_rejects_half_a_guide(which):
    """Only one of aligned_a / aligned_b used to mean "no guide": the string given was
    never even validated (scan B8)."""
    import nwgrad
    p = _dna_params(nwgrad)
    kw = {"aligned_a": "malformed"} if which == "a" else {"aligned_b": "malformed"}
    with pytest.raises(ValueError, match="both aligned_a and aligned_b"):
        nwgrad.nw_score_affine("ACGT", "ACGT", p, band=1, **kw)


@pytest.mark.parametrize("cls", ["SeqPairBatch", "SeqPairBatchDouble"])
def test_align_rejects_half_a_guide_list(cls):
    import nwgrad
    batch = getattr(nwgrad, cls)(1, grad_mode="hard")
    with pytest.raises(ValueError):
        batch.align(["ACGT"], ["ACGT"], _dna_params(nwgrad), band=1, aligned_b=["malformed"])


def test_negative_band_is_rejected_everywhere():
    """band < 0 used to select the Full DP, where the band check never ran (scan B6)."""
    import nwgrad
    p = _dna_params(nwgrad)
    with pytest.raises(ValueError, match="band must be >= 0"):
        nwgrad.nw_score_affine("ACGT", "ACGT", p, band=-1)
    with pytest.raises(ValueError, match="band must be >= 0"):
        nwgrad.SeqPairBatchDouble(1, grad_mode="hard").align(["ACGT"], ["ACGT"], p, band=-1)


def test_huge_band_equals_full():
    import nwgrad
    p = _dna_params(nwgrad)
    assert nwgrad.nw_score_affine_double("ACGTAC", "AGTTACG", p, band=2**31 - 1) == \
        nwgrad.nw_score_affine_double("ACGTAC", "AGTTACG", p)
