"""Spot-checks for predefined substitution matrices in nwgrad.matrices.

Values cross-checked against:
  - BLOSUM/PAM/NUC44: NCBI BLAST FTP ftp.ncbi.nlm.nih.gov/blast/matrices/
  - VTML: wrpearson/fasta36 GitHub repository (data/ directory)
"""
import hashlib

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

# Full-table references downloaded and compared cell-for-cell on 2026-10-09:
# BLOSUM*, PAM*, NUC44: https://ftp.ncbi.nlm.nih.gov/blast/matrices/
#   (NUC44's filename is NUC.4.4; protein stop-codon rows/columns are omitted).
# VTML*: https://github.com/wrpearson/fasta36/tree/6d6718aaf949483197dd48ea9baecfb7d0f73a41/data
#   (filenames VTML_40.mat, VTML_80.mat, VTML_160.mat, VTML_200.mat).
# Hash alphabet ASCII + NUL + the source table reordered to that alphabet as
# row-major little-endian float64. No network access is needed during pytest.
_REFERENCE_SHA256 = {
    "BLOSUM45": "c436c9eee6277f3a9d02a8bee9224f79085d6ff5075110d873dbb0680fd11a27",
    "BLOSUM50": "59ab3a456e612b25877254624c367f2302f575e96dc63b4540242d37c08e153b",
    "BLOSUM62": "da22fab0e8bb547e456f03dd6e89d7e5d79bdcf1ea73c33f8c1047d6c7a106af",
    "BLOSUM80": "fe3e3b28eb02271c87287c39cfb63ea9f062d46625f7a5dba8d4cebd36ae7c7e",
    "BLOSUM90": "d8133d90063e1472cbbdd8a88b1a9e917015872e1bd6fb5b39e7775215555e43",
    "PAM30": "21a882f410c09af56aeff91e06294c2046e27769fc01bdb83329b8185d35b97f",
    "PAM70": "e0452ee87becc4e993acbff67bebfb00ca8d9000f386dc5f9eeeebcbd6b2a978",
    "PAM250": "43afb2f706eba85581925444ad93efcc740a20dc1aa0a6f4e30acbe09e647809",
    "VTML40": "543d7e25add26121744147aaeb46b56242cbf576e7914b91310b296d76cc421f",
    "VTML80": "b640e42989fd4f5f8e4376019bbc2f1d893efcca79c8098e8d474cee7e9a332d",
    "VTML160": "9338b20ff87403b298ffa31eedf7e121af410a004aa93e7f4a415434b06659d8",
    "VTML200": "18df50c1567697f62fa4e13f9e3800f471f724d6650ee1db5d6857fe07aabc05",
    "NUC44": "069c6bd85045b38bf9666da88a60742b6089a551bb1fba8461bad4d02d2d7386",
}


@pytest.mark.parametrize("name", list(_REFERENCE_SHA256))
def test_every_cell_matches_online_reference(name):
    matrix = getattr(matrices, name)
    values = np.ascontiguousarray(matrix.to_matrix(), dtype="<f8")
    digest = hashlib.sha256(matrix.alphabet.encode("ascii") + b"\0" + values.tobytes())
    assert digest.hexdigest() == _REFERENCE_SHA256[name]


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
        sp.align_full()
        assert sp.score == pytest.approx(20.0)  # 4 × 5.0
