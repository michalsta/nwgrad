"""Spot-checks for predefined substitution matrices in nwgrad.matrices.

Values cross-checked against:
  - BLOSUM/PAM/NUC44: NCBI BLAST FTP ftp.ncbi.nlm.nih.gov/blast/matrices/
  - VTML: wrpearson/fasta36 GitHub repository (data/ directory)
"""
import numpy as np
import pytest
import nwgrad
from nwgrad import matrices
from nwgrad.matrices import (
    BLOSUM45, BLOSUM50, BLOSUM62, BLOSUM80, BLOSUM90,
    PAM30, PAM70, PAM250,
    VTML40, VTML80, VTML160, VTML200,
    NUC44,
)


# Parametrize over matrix *names* rather than the SubstMatrix objects themselves:
# passing bound instances as parametrize values makes pytest retain them for the
# whole session, which trips nanobind's leak checker at interpreter shutdown.
_AA_MAT_NAMES = ["BLOSUM45", "BLOSUM50", "BLOSUM62", "BLOSUM80", "BLOSUM90",
                 "PAM30", "PAM70", "PAM250", "VTML40", "VTML80", "VTML160", "VTML200"]
_AA_ALPHABET = "ARNDCQEGHILKMFPSTWYVBZX"


class TestShapesAndAlphabets:
    @pytest.mark.parametrize("name", _AA_MAT_NAMES)
    def test_aa_alphabet(self, name):
        assert getattr(matrices, name).alphabet == _AA_ALPHABET

    @pytest.mark.parametrize("name", _AA_MAT_NAMES)
    def test_aa_shape(self, name):
        assert getattr(matrices, name).to_matrix().shape == (23, 23)

    def test_nuc44_alphabet(self):
        assert NUC44.alphabet == "ATGCSWRYKMBVHDN"

    def test_nuc44_shape(self):
        assert NUC44.to_matrix().shape == (15, 15)


class TestBLOSUM62:
    # Known values from NCBI BLOSUM62
    def test_aa_diagonal(self):
        assert BLOSUM62.score('A', 'A') == 4.0
        assert BLOSUM62.score('C', 'C') == 9.0
        assert BLOSUM62.score('W', 'W') == 11.0

    def test_conservative_substitution(self):
        assert BLOSUM62.score('D', 'E') == 2.0
        assert BLOSUM62.score('I', 'V') == 3.0
        assert BLOSUM62.score('K', 'R') == 2.0

    def test_dissimilar_substitution(self):
        assert BLOSUM62.score('A', 'W') == -3.0
        assert BLOSUM62.score('C', 'P') == -3.0

    def test_symmetry(self):
        # BLOSUM62 is symmetric
        mat = BLOSUM62.to_matrix()
        np.testing.assert_array_equal(mat, mat.T)

    def test_roundtrip(self):
        mat = BLOSUM62.to_matrix()
        recovered = nwgrad.SubstMatrix(mat, _AA_ALPHABET)
        np.testing.assert_array_equal(mat, recovered.to_matrix())


class TestBLOSUM45:
    def test_diagonal(self):
        assert BLOSUM45.score('A', 'A') == 5.0
        assert BLOSUM45.score('W', 'W') == 15.0

    def test_off_diagonal(self):
        assert BLOSUM45.score('D', 'E') == 2.0


class TestBLOSUM80:
    def test_diagonal(self):
        assert BLOSUM80.score('A', 'A') == 7.0


class TestBLOSUM90:
    def test_diagonal(self):
        assert BLOSUM90.score('A', 'A') == 5.0
        assert BLOSUM90.score('C', 'C') == 9.0


class TestPAM250:
    def test_diagonal(self):
        assert PAM250.score('A', 'A') == 2.0
        assert PAM250.score('W', 'W') == 17.0

    def test_conservative(self):
        assert PAM250.score('D', 'E') == 3.0

    def test_symmetry(self):
        mat = PAM250.to_matrix()
        np.testing.assert_array_equal(mat, mat.T)


class TestPAM30:
    def test_diagonal(self):
        assert PAM30.score('A', 'A') == 6.0
        assert PAM30.score('W', 'W') == 13.0


class TestPAM70:
    def test_diagonal(self):
        assert PAM70.score('A', 'A') == 5.0


class TestVTML:
    def test_vtml200_alanine(self):
        assert VTML200.score('A', 'A') == 4.0

    def test_vtml80_alanine(self):
        assert VTML80.score('A', 'A') == 5.0

    def test_vtml40_alanine(self):
        assert VTML40.score('A', 'A') == 6.0

    def test_vtml160_alanine(self):
        assert VTML160.score('A', 'A') == 5.0


class TestNUC44:
    def test_match(self):
        assert NUC44.score('A', 'A') == 5.0
        assert NUC44.score('T', 'T') == 5.0

    def test_mismatch(self):
        assert NUC44.score('A', 'T') == -4.0
        assert NUC44.score('A', 'G') == -4.0

    def test_ambiguity_n(self):
        assert NUC44.score('N', 'A') == -2.0
        assert NUC44.score('A', 'N') == -2.0
        assert NUC44.score('N', 'N') == -1.0


class TestUsableWithAlignParams:
    def test_blosum62_alignment(self):
        params = nwgrad.AlignParams(
            BLOSUM62.to_matrix(),
            alphabet=BLOSUM62.alphabet,
            gap_open_a=11.0, gap_extend_a=1.0,
            gap_open_b=11.0, gap_extend_b=1.0,
        )
        sp = nwgrad.SeqPair("PLEASANTLY", "MEANLY", params,
                            gap_model="affine", mode="global", grad_mode="hard")
        sp.alloc_dp()
        sp.align_full()
        assert sp.score == pytest.approx(-3.0)

    def test_nuc44_alignment(self):
        params = nwgrad.AlignParams(
            NUC44.to_matrix(),
            alphabet=NUC44.alphabet,
            gap_extend_a=2.0, gap_extend_b=2.0,
        )
        sp = nwgrad.SeqPair("ACGT", "ACGT", params,
                            gap_model="linear", mode="global", grad_mode="none")
        sp.alloc_dp()
        sp.align_full()
        assert sp.score == pytest.approx(20.0)  # 4 × 5.0
