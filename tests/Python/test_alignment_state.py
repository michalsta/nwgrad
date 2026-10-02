"""Composition of soft DP, traceback, and parameter replacement.

Crash-prone checks run in children so one bad traceback cannot kill pytest.
Known defects are xfailed until fixed; --runxfail exposes the reproductions.
"""

import pytest

from conftest import describe, run_isolated


PRELUDE = """
import gc
import numpy as np
import nwgrad as n

def params(k=3, alphabet="ACGT"):
    return n.AlignParams(k * np.eye(4) - 1, alphabet=alphabet,
                         gap_open_a=2, gap_extend_a=1,
                         gap_open_b=2, gap_extend_b=1)
"""


@pytest.mark.parametrize("precision", ["", "Double"], ids=["float32", "double"])
@pytest.mark.parametrize("mode", ["global", "local"])
@pytest.mark.parametrize("kernel", ["scalar_fallback", "auto"])
@pytest.mark.parametrize("traceback,cutoff", [
    ("scores", 512), ("pointers", 512),
    ("hirschberg", 512), ("hirschberg", 1),
])
def test_soft_pair_retains_viterbi_alignment(precision, mode, kernel, traceback,
                                           cutoff, request):
    # F/B are row-major, but changing their layout must not change how the
    # separately retained Viterbi scores/directions are read.
    import nwgrad
    simd = kernel == "auto" and nwgrad.get_isa_level() != "scalar_fallback"
    if ((traceback == "scores" and simd) or traceback == "pointers"
            or (traceback == "hirschberg" and cutoff >= 4)):
        request.node.add_marker(pytest.mark.xfail(
            strict=True, reason="forward-backward overwrites Viterbi layout metadata"))
    proc = run_isolated(PRELUDE + f"""
sp = n.SeqPair{precision}("ACGT", "ACGT", params(), mode={mode!r},
                          grad_mode="soft", kernel={kernel!r}, traceback={traceback!r})
sp.hb_cutoff = {cutoff}
score, grad = sp.score_and_grad()
assert score >= 8
assert np.isfinite(grad.matrix.to_matrix()).all()
assert sp.aligned() == ("ACGT", "ACGT"), sp.aligned()
assert sp.formatted(0) == "ACGT\\n||||\\nACGT"
# Exercise reuse, including the Full -> GuideBanded transition.
sp.realign_banded(4)
sp.compute_grad()
assert sp.aligned() == ("ACGT", "ACGT"), sp.aligned()
""")
    assert proc.returncode == 0, describe(proc)


@pytest.mark.parametrize("precision", ["", "Double"], ids=["float32", "double"])
@pytest.mark.parametrize("traceback", ["scores", "pointers"])
@pytest.mark.parametrize("realign", [False, True], ids=["cached-path", "realigned"])
def test_traceback_after_replacing_and_releasing_params(precision, traceback,
                                                       realign, request):
    if not realign:
        # Allocator reuse makes this fail on normal builds; ASan detects the
        # dangling read regardless of reuse. Do not make an allocator-dependent
        # XPASS break a normal build on another platform.
        request.node.add_marker(pytest.mark.xfail(
            strict=False, reason="set_params leaves traceback pointing at freed params"))
    proc = run_isolated(PRELUDE + f"""
sp = n.SeqPair{precision}("ACGT", "ACGT", params(),
                          traceback={traceback!r}, kernel="scalar_fallback")
# Constructor params are pinned separately. Replace once before computing, so
# the parameters used by the DP can actually die on the second replacement.
sp.set_params(params(4))
sp.score_and_grad()
expected = sp.aligned()
guide = list(sp.guide_j)
sp.set_params(params(5))
assert not sp.score_valid and not sp.grad_valid
assert list(sp.guide_j) == guide
gc.collect()
trash = [params(99, "TGCA") for _ in range(500)]
if {realign!r}:
    sp.realign_banded(4)
    sp.compute_grad()
    assert sp.score == 16
    assert sp.aligned() == expected
else:
    # Safely rejecting a stale traceback is also acceptable; reading freed
    # params or decoding the sequence using a different alphabet is not.
    try:
        got = sp.aligned()
    except RuntimeError:
        pass
    else:
        assert got == expected, (got, expected)
""")
    assert proc.returncode == 0, describe(proc)
