"""
Cross-validation of nwgrad.nw_score_affine against Bio.Align.PairwiseAligner (global NW).
"""

import numpy as np
import pytest

pytest.importorskip("Bio", reason="biopython not installed")

from Bio import Align
from Bio.Align import substitution_matrices
import nwgrad

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"


@pytest.fixture(scope="module")
def bio_blosum62():
    full = substitution_matrices.load("BLOSUM62")
    m = substitution_matrices.Array(alphabet=AA_ORDER, dims=2)
    for a in AA_ORDER:
        for b in AA_ORDER:
            m[a, b] = full[a, b]
    return m


@pytest.fixture(scope="module")
def blosum_arr(bio_blosum62):
    return np.array(
        [[bio_blosum62[a, b] for b in AA_ORDER] for a in AA_ORDER],
        dtype=np.float64,
    )


def make_params(blosum_arr, gap_extend, gap_open=0.0):
    return nwgrad.AlignParams(blosum_arr,
                               gap_open_a=gap_open, gap_extend_a=gap_extend,
                               gap_open_b=gap_open, gap_extend_b=gap_extend)


def make_bio_aligner(bio_matrix, gap_open: float, gap_extend: float) -> Align.PairwiseAligner:
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = bio_matrix
    aligner.open_gap_score   = -(gap_open + gap_extend)
    aligner.extend_gap_score = -gap_extend
    return aligner


PAIRS = [
    ("A",           "A"),
    ("A",           "C"),
    ("A",           "AC"),
    ("ACDE",        "ACDE"),
    ("ACDE",        "ACDF"),
    ("ACDEFG",      "ACDE"),
    ("PLEASANTLY",  "MEANLY"),
    ("ACDEFGHIKL",  "CDEFGHIKLM"),
    ("MADEEKLF",    "MADEEKLF"),
    ("MADEEKLF",    "MADE"),
    ("ACDEFGHIKLMN","ACDEFGHIKLMN"),
    ("ACDEFGHIKLMN","NPQRSTVWY"),
]

GAP_PARAMS = [(0.0, 0.5), (0.0, 1.0), (1.0, 0.5), (2.0, 1.0), (5.0, 2.0), (11.0, 1.0)]


@pytest.mark.parametrize("gap_open,gap_extend", GAP_PARAMS)
@pytest.mark.parametrize("a,b", PAIRS)
def test_score_matches_biopython(a, b, gap_open, gap_extend, blosum_arr, bio_blosum62):
    bio_aligner = make_bio_aligner(bio_blosum62, gap_open, gap_extend)
    expected = bio_aligner.score(a, b)
    got = nwgrad.nw_score_affine(a, b, make_params(blosum_arr, gap_extend, gap_open))
    assert got == pytest.approx(expected, abs=1e-9), (
        f"nw_score_affine({a!r},{b!r}, open={gap_open}, ext={gap_extend}): "
        f"got {got}, expected {expected}"
    )


@pytest.mark.parametrize("gap_open,gap_extend", GAP_PARAMS)
@pytest.mark.parametrize("a,b", PAIRS)
def test_symmetric_vs_biopython(a, b, gap_open, gap_extend, blosum_arr, bio_blosum62):
    bio_aligner = make_bio_aligner(bio_blosum62, gap_open, gap_extend)
    p = make_params(blosum_arr, gap_extend, gap_open)
    fwd = nwgrad.nw_score_affine(a, b, p)
    rev = nwgrad.nw_score_affine(b, a, p)
    assert fwd == pytest.approx(bio_aligner.score(a, b), abs=1e-9)
    assert rev == pytest.approx(bio_aligner.score(b, a), abs=1e-9)
    assert fwd == pytest.approx(rev, abs=1e-9)
