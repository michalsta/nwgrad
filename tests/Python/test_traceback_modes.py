"""traceback="pointers" must be indistinguishable from traceback="scores".

The two differ only in how a cell's predecessor is recovered:

  "scores"    retain VM/VX/VY for the whole DP and re-derive the argmax by repeating
              the forward pass's comparison chain.  12 B/cell (float32).
  "pointers"  record the predecessor as one byte per cell per state during the fill
              and follow it; scores then need only two rolling rows.  3 B/cell, and
              measured 1.4-2.2x faster because the retained footprint is what drives
              the page-fault cost in the memory-bound regime.

The contract is not "close enough" but *identical*: same score, same alignment
strings, same guide_j, same gradient, ties included.  A different-but-valid
subgradient is precisely what disqualified Hirschberg, so pointers is held to the
stricter standard.

The matrix here is integral (+4 / -1) with integral gap costs, so **ties are common**.
Under a random real-valued matrix exact ties essentially never occur and a broken
tie-break would sail straight through.
"""

import numpy as np
import pytest

import nwgrad

AA = "ARNDCQEGHILKMFPSTWYV"
MODES = ["pointers", "scores"]


@pytest.fixture(scope="module")
def params():
    m = np.full((20, 20), -1.0)
    np.fill_diagonal(m, 4.0)                      # integral => ties are common
    mat = nwgrad.SubstMatrix(m, alphabet=AA)
    return nwgrad.AlignParams(mat, gap_open_a=11.0, gap_extend_a=1.0,
                              gap_open_b=11.0, gap_extend_b=1.0)


def _seqs(n, lo, hi, seed):
    rng = np.random.default_rng(seed)
    return ["".join(rng.choice(list(AA), int(l))) for l in rng.integers(lo, hi, n)]


def _batch(seqs_a, seqs_b, params, mode, traceback, grad_mode="hard", threads=4):
    b = nwgrad.SeqPairBatch(n_threads=threads, traceback=traceback)
    b.add_many(seqs_a, seqs_b, params, gap_model="affine", mode=mode,
               grad_mode=grad_mode, kernel="auto")
    return b


def test_pointers_is_the_default():
    assert nwgrad.SeqPairBatch(n_threads=1).traceback == "pointers"


def test_traceback_is_fixed_at_construction(params):
    """It decides what the DP retains, so it is deliberately not flippable later."""
    b = nwgrad.SeqPairBatch(n_threads=1)
    with pytest.raises(AttributeError):
        b.traceback = "scores"
    sp = nwgrad.SeqPair("ACDE", "ACDE", params)
    with pytest.raises(AttributeError):
        sp.traceback = "scores"


def test_unknown_traceback_rejected():
    with pytest.raises(ValueError, match="unknown traceback"):
        nwgrad.SeqPairBatch(n_threads=1, traceback="bogus")


@pytest.mark.parametrize("mode", ["global", "local"])
def test_score_and_grad_identical(params, mode):
    a_seqs = _seqs(200, 1, 120, 1)
    b_seqs = _seqs(200, 1, 120, 2)
    out = {}
    for tb in MODES:
        batch = _batch(a_seqs, b_seqs, params, mode, tb)
        out[tb] = (batch.score_and_grad(),
                   [batch[i].score for i in range(len(a_seqs))],
                   batch.compute_grad().to_dict())
    assert out["scores"][0] == out["pointers"][0]
    assert out["scores"][1] == out["pointers"][1]
    gs, gp = out["scores"][2], out["pointers"][2]
    np.testing.assert_array_equal(gs["matrix"], gp["matrix"])
    for f in ("gap_open_a", "gap_extend_a", "gap_open_b", "gap_extend_b"):
        assert gs[f] == gp[f], f


@pytest.mark.parametrize("mode", ["global", "local"])
def test_aligned_strings_identical(params, mode):
    """align_full() + aligned() — the pair's OWN buffers.

    Regression: "pointers" skips allocating VM/VX/VY, and `aligned_affine` used to
    read them anyway.  That was a SEGFAULT, not a wrong answer, and nothing covered
    it because the batch path drops the pair's tables and raises instead.
    """
    a_seqs = _seqs(60, 1, 90, 3)
    b_seqs = _seqs(60, 1, 90, 4)
    out = {}
    for tb in MODES:
        batch = _batch(a_seqs, b_seqs, params, mode, tb, threads=1)
        batch.alloc_dp()
        out[tb] = (batch.align_full(),
                   [batch[i].aligned() for i in range(len(a_seqs))],
                   [list(batch[i].guide_j) for i in range(len(a_seqs))])
    assert out["scores"] == out["pointers"]


@pytest.mark.parametrize("mode", ["global", "local"])
@pytest.mark.parametrize("band", [8, 32])
def test_banded_grad_identical(params, mode, band):
    """Pointers governs the full DP; the banded aligner always keeps score tables.
    The two must still compose — establish guides, then re-align banded around them.

    Lengths are deliberately unequal, so the band is often narrower than |m - n| —
    the regime that used to segfault the traceback (see
    test_guided_banded.py::test_narrow_band_does_not_crash).
    """
    a_seqs = _seqs(120, 20, 150, 5)
    b_seqs = _seqs(120, 20, 150, 6)
    out = {}
    for tb in MODES:
        batch = _batch(a_seqs, b_seqs, params, mode, tb)
        batch.score_and_grad()
        out[tb] = (batch.banded_grad(band),
                   [batch[i].score for i in range(len(a_seqs))])
    assert out["scores"] == out["pointers"]


@pytest.mark.parametrize("mode", ["global", "local"])
def test_degenerate_shapes(params, mode):
    """Empty and single-residue sequences walk the BORDER cells of the pointer
    tables — the ones the recurrence never visits, so the fill must write them."""
    odd = ["", "A", "AC", "ACDEFGHIK"]
    pairs = [(x, y) for x in odd for y in odd]
    out = {}
    for tb in MODES:
        batch = _batch([x for x, _ in pairs], [y for _, y in pairs], params, mode, tb,
                       threads=1)
        out[tb] = (batch.score_and_grad(),
                   [batch[i].score for i in range(len(pairs))])
    assert out["scores"] == out["pointers"]


def test_soft_gradient_unaffected(params):
    """The soft path is forward-backward in double and has no pointers; asking for
    them must not change — or corrupt — it."""
    a_seqs = _seqs(40, 5, 60, 7)
    b_seqs = _seqs(40, 5, 60, 8)
    out = {}
    for tb in MODES:
        batch = _batch(a_seqs, b_seqs, params, "global", tb, grad_mode="soft")
        out[tb] = (batch.score_and_grad(), batch.compute_grad().to_dict()["matrix"])
    assert out["scores"][0] == pytest.approx(out["pointers"][0], rel=1e-12)
    np.testing.assert_allclose(out["scores"][1], out["pointers"][1], atol=1e-10)


def test_seq_pair_traceback_argument(params):
    """The per-pair constructor takes it too, and pairs added by add() keep it."""
    a, b = "ACDEFGHIKLMNPQ", "ACDWFGHIKLMNPQ"
    got = {}
    for tb in MODES:
        sp = nwgrad.SeqPair(a, b, params, gap_model="affine", mode="global",
                            grad_mode="hard", traceback=tb)
        assert sp.traceback == tb
        batch = nwgrad.SeqPairBatch(n_threads=1)
        batch.add(sp)
        batch.alloc_dp()
        got[tb] = (batch.align_full(), sp.aligned())
    assert got["scores"] == got["pointers"]
