"""SeqPair.coordinates(): the alignment as Biopython-style coordinates.

Two checks: the coordinates rebuild exactly the gapped strings aligned() returns
(Bio.Align.Alignment round trip, every gap model, mode and traceback), and they are
Biopython's own coordinates wherever Biopython reports a single optimal alignment.
"""

import numpy as np
import pytest

import nwgrad

Bio = pytest.importorskip("Bio")
from Bio.Align import Alignment, PairwiseAligner, substitution_matrices  # noqa: E402

DNA = "ACGT"


def _seqs(n, lo, hi, seed):
    rng = np.random.default_rng(seed)
    return ["".join(rng.choice(list(DNA), int(l))) for l in rng.integers(lo, hi, n)]


def _params(seed, go, ge):
    m = np.random.default_rng(seed).normal(size=(4, 4))
    return nwgrad.AlignParams(nwgrad.SubstMatrix(m, alphabet=DNA), gap_open_a=go,
                              gap_extend_a=ge, gap_open_b=go, gap_extend_b=ge), m


def _pair(a, b, p, gap_model, mode, traceback="pointers"):
    sp = nwgrad.SeqPairDouble(a, b, p, gap_model=gap_model, mode=mode, traceback=traceback)
    sp.alloc_dp()
    sp.align_full()
    return sp


@pytest.mark.parametrize("gap_model", ["affine", "linear"])
@pytest.mark.parametrize("mode", ["local", "global"])
@pytest.mark.parametrize("traceback", ["pointers", "scores", "hirschberg"])
def test_round_trip(gap_model, mode, traceback):
    if traceback == "hirschberg" and gap_model == "linear":
        pytest.skip("Hirschberg is affine only")
    p, _ = _params(1, 1.5 if gap_model == "affine" else 0.0, 0.6)
    for a, b in zip(_seqs(60, 1, 40, 2), _seqs(60, 1, 60, 3)):
        sp = _pair(a, b, p, gap_model, mode, traceback)
        c = sp.coordinates()
        assert c.dtype == np.int64 and c.shape[0] == 2
        sa, sb = sp.aligned()
        aln = Alignment([a, b], c)
        assert (aln[0], aln[1]) == (sa, sb)
        if mode == "global":
            assert c[:, 0].tolist() == [0, 0] and c[:, -1].tolist() == [len(a), len(b)]


def test_round_trip_after_banded_realignment():
    p1, _ = _params(4, 1.5, 0.6)
    p2, _ = _params(5, 1.5, 0.6)
    for a, b in zip(_seqs(40, 5, 40, 6), _seqs(40, 5, 50, 7)):
        sp = _pair(a, b, p1, "affine", "local")
        sp.set_params(p2)
        sp.realign_banded(3)
        aln = Alignment([a, b], sp.coordinates())
        assert (aln[0], aln[1]) == sp.aligned()


@pytest.mark.parametrize("gap_model", ["affine", "linear"])
@pytest.mark.parametrize("mode", ["local", "global"])
def test_matches_biopython_where_the_optimum_is_unique(gap_model, mode):
    go, ge = (1.5, 0.6) if gap_model == "affine" else (0.0, 0.6)
    p, m = _params(8, go, ge)
    aligner = PairwiseAligner()
    aligner.mode = mode
    aligner.substitution_matrix = substitution_matrices.Array(alphabet=DNA, data=m)
    # nwgrad charges a gap of length k  go + k*ge;  Biopython  open + (k-1)*extend.
    aligner.open_gap_score = -(go + ge)
    aligner.extend_gap_score = -ge
    compared = 0
    for a, b in zip(_seqs(200, 3, 30, 9), _seqs(200, 3, 40, 10)):
        alns = aligner.align(a, b)
        if len(alns) != 1:
            continue
        sp = _pair(a, b, p, gap_model, mode)
        assert sp.score == pytest.approx(alns.score, abs=1e-9)
        np.testing.assert_array_equal(sp.coordinates(), alns[0].coordinates)
        compared += 1
    # The fixture must actually exercise the comparison.  Global linear has the
    # fewest unique optima (a linear gap slides freely along equal residues): ~19/200.
    assert compared >= 10
