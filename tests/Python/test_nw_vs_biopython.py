"""
Cross-validation of nwgrad.nw_score against Bio.Align.PairwiseAligner (global NW).

Gap model mapping
-----------------
Our model:      gap_cost(k) = gap_extend * k
BioPython model: gap_cost(k) = open + extend * (k-1)

Setting open = extend = -gap_extend makes the two models identical.

BioPython cannot align empty sequences, so empty-string edge cases are excluded here.
"""

import numpy as np
import pytest

pytest.importorskip("Bio", reason="biopython not installed")

from Bio import Align
from Bio.Align import substitution_matrices
import nwgrad

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def bio_blosum62():
    """BioPython's built-in BLOSUM62, restricted to our 20-AA alphabet."""
    full = substitution_matrices.load("BLOSUM62")
    m = substitution_matrices.Array(alphabet=AA_ORDER, dims=2)
    for a in AA_ORDER:
        for b in AA_ORDER:
            m[a, b] = full[a, b]
    return m


@pytest.fixture(scope="module")
def blosum(bio_blosum62):
    """nwgrad SubstMatrix built from the same values as bio_blosum62."""
    arr = np.array(
        [[bio_blosum62[a, b] for b in AA_ORDER] for a in AA_ORDER],
        dtype=np.float64,
    )
    return nwgrad.SubstMatrix(arr)


def make_bio_aligner(bio_matrix, gap_extend: float) -> Align.PairwiseAligner:
    """
    Build a BioPython global NW aligner with linear gap penalty matching our model.
    Our gap_cost(k) = gap_extend * k
    BioPython gap_cost(k) = open + extend*(k-1)  →  open = extend = -gap_extend
    """
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = bio_matrix
    aligner.open_gap_score   = -gap_extend
    aligner.extend_gap_score = -gap_extend
    return aligner


# ── Test pairs ───────────────────────────────────────────────────────────────

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

GAP_EXTENDS = [0.5, 1.0, 2.0, 5.0]


@pytest.mark.parametrize("gap_extend", GAP_EXTENDS)
@pytest.mark.parametrize("a,b", PAIRS)
def test_score_matches_biopython(a, b, gap_extend, blosum, bio_blosum62):
    bio_aligner = make_bio_aligner(bio_blosum62, gap_extend)
    expected = bio_aligner.score(a, b)
    got = nwgrad.nw_score(a, b, blosum, gap_extend)
    assert got == pytest.approx(expected, abs=1e-9), (
        f"nw_score({a!r}, {b!r}, gap={gap_extend}): got {got}, expected {expected}"
    )


@pytest.mark.parametrize("gap_extend", GAP_EXTENDS)
@pytest.mark.parametrize("a,b", PAIRS)
def test_symmetric_vs_biopython(a, b, gap_extend, blosum, bio_blosum62):
    """Both implementations should give the same score when sequences are swapped."""
    bio_aligner = make_bio_aligner(bio_blosum62, gap_extend)
    fwd = nwgrad.nw_score(a, b, blosum, gap_extend)
    rev = nwgrad.nw_score(b, a, blosum, gap_extend)
    bio_fwd = bio_aligner.score(a, b)
    bio_rev = bio_aligner.score(b, a)
    assert fwd  == pytest.approx(bio_fwd,  abs=1e-9)
    assert rev  == pytest.approx(bio_rev,  abs=1e-9)
    assert fwd  == pytest.approx(rev,      abs=1e-9)
