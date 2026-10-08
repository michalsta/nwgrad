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
                                           cutoff):
    # F/B are row-major, but changing their layout must not change how the
    # separately retained Viterbi scores/directions are read.
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
                                                       realign):
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
    # set_params() invalidates the retained tables: they and the traceback
    # belong to the replaced params, which may already be freed.
    assert not sp.dp_valid
    for call in (sp.aligned, lambda: sp.formatted(0)):
        try:
            call()
        except RuntimeError:
            pass
        else:
            raise AssertionError("stale traceback was readable after set_params()")
""")
    assert proc.returncode == 0, describe(proc)


@pytest.mark.parametrize("precision", ["", "Double"], ids=["float32", "double"])
@pytest.mark.parametrize("traceback", ["scores", "pointers"])
@pytest.mark.parametrize("realign", [False, True], ids=["cached-path", "realigned"])
def test_batch_set_params_invalidates_owned_pair_traceback(precision, traceback,
                                                           realign):
    # Batch-owned pairs (add_many) borrow the batch's params cell, so a batch
    # set_params() releases the params their tables were computed under.
    proc = run_isolated(PRELUDE + f"""
b = n.SeqPairBatch{precision}(n_threads=1, traceback={traceback!r})
b.add_many(["ACGT", "ACGTT"], ["ACGT", "ACGT"], params(), kernel="scalar_fallback")
b.set_params(params(4))
b.align_full()
expected = [b[i].aligned() for i in range(len(b))]
b.set_params(params(5))
gc.collect()
trash = [params(99, "TGCA") for _ in range(500)]
for i in range(len(b)):
    assert not b[i].dp_valid
    try:
        b[i].aligned()
    except RuntimeError:
        pass
    else:
        raise AssertionError("stale traceback was readable after batch.set_params()")
if {realign!r}:
    b.realign_banded(4)
    assert [b[i].aligned() for i in range(len(b))] == expected
""")
    assert proc.returncode == 0, describe(proc)
