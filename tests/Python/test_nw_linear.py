"""Global NW score tests — correctness verified against Bio.Align.PairwiseAligner."""

import numpy as np
import pytest
import nwgrad
from Bio import Align
from Bio.Align import substitution_matrices
from test_subst_matrix import BLOSUM62

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"

pytest.importorskip("Bio", reason="biopython not installed")


def make_params(gap_extend, gap_open=0.0):
    return nwgrad.AlignParams(BLOSUM62, gap_open_a=gap_open, gap_extend_a=gap_extend,
                                        gap_open_b=gap_open, gap_extend_b=gap_extend)


@pytest.fixture(scope="module")
def bio_blosum62():
    m = substitution_matrices.Array(alphabet=AA_ORDER, dims=2)
    for i, a in enumerate(AA_ORDER):
        for j, b in enumerate(AA_ORDER):
            m[a, b] = BLOSUM62[i, j]
    return m


def make_bio_aligner(bio_matrix, gap_extend):
    """Global NW aligner with linear gap penalty matching our model (gap_cost = k * gap_extend)."""
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = bio_matrix
    aligner.open_gap_score = -gap_extend
    aligner.extend_gap_score = -gap_extend
    return aligner


# ── Correctness vs. BioPython ─────────────────────────────────────────────────

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
def test_matches_biopython(a, b, gap_extend, bio_blosum62):
    expected = make_bio_aligner(bio_blosum62, gap_extend).score(a, b)
    got = nwgrad.nw_score(a, b, make_params(gap_extend))
    assert got == pytest.approx(expected, abs=1e-9), (
        f"nw_score({a!r}, {b!r}, gap={gap_extend}): got {got}, expected {expected}"
    )


@pytest.mark.parametrize("gap_extend", GAP_EXTENDS)
@pytest.mark.parametrize("a,b", PAIRS)
def test_symmetric_vs_biopython(a, b, gap_extend, bio_blosum62):
    bio = make_bio_aligner(bio_blosum62, gap_extend)
    p = make_params(gap_extend)
    assert nwgrad.nw_score(a, b, p) == pytest.approx(bio.score(a, b), abs=1e-9)
    assert nwgrad.nw_score(b, a, p) == pytest.approx(bio.score(b, a), abs=1e-9)
    assert nwgrad.nw_score(a, b, p) == pytest.approx(nwgrad.nw_score(b, a, p), abs=1e-9)


# ── Identity score ────────────────────────────────────────────────────────────

def test_identity_score():
    """Aligning a sequence with itself yields the sum of its self-substitution scores."""
    seq = "ACDEFG"
    expected = sum(BLOSUM62[AA_ORDER.index(c), AA_ORDER.index(c)] for c in seq)
    assert nwgrad.nw_score(seq, seq, make_params(1.0)) == pytest.approx(expected)


# ── Gap penalty ───────────────────────────────────────────────────────────────

def test_gap_penalty_scaling():
    """Higher gap penalty should reduce (or equal) the score for mismatched-length seqs."""
    a, b = "ACDEFG", "ACDE"
    assert nwgrad.nw_score(a, b, make_params(0.1)) >= nwgrad.nw_score(a, b, make_params(10.0))


# ── Edge cases (empty sequences — BioPython cannot align these) ───────────────

def test_empty_vs_empty():
    assert nwgrad.nw_score("", "", make_params(1.0)) == pytest.approx(0.0)


def test_empty_vs_seq():
    seq = "ACDE"
    assert nwgrad.nw_score("", seq, make_params(2.0)) == pytest.approx(-len(seq) * 2.0)
    assert nwgrad.nw_score(seq, "", make_params(2.0)) == pytest.approx(-len(seq) * 2.0)
