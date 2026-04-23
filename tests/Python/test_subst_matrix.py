import numpy as np
import pytest
import nwgrad

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"

# Minimal BLOSUM62 diagonal + a few off-diagonal values for spot-checks.
# Full 20×20 BLOSUM62 (canonical AA order: ACDEFGHIKLMNPQRSTVWY).
BLOSUM62 = np.array([
    # A   C   D   E   F   G   H   I   K   L   M   N   P   Q   R   S   T   V   W   Y
    [ 4,  0, -2, -1, -2,  0, -2, -1, -1, -1, -1, -2, -1, -1, -1,  1,  0,  0, -3, -2],  # A
    [ 0,  9, -3, -4, -2, -3, -3, -1, -3, -1, -1, -3, -3, -3, -3, -1, -1, -1, -2, -2],  # C
    [-2, -3,  6,  2, -3, -1, -1, -3, -1, -4, -3,  1, -1,  0, -2,  0, -1, -3, -4, -3],  # D
    [-1, -4,  2,  5, -3, -2,  0, -3,  1, -3, -2,  0, -1,  2,  0,  0, -1, -2, -3, -2],  # E
    [-2, -2, -3, -3,  6, -3, -1,  0, -3,  0,  0, -3, -4, -3, -3, -2, -2, -1,  1,  3],  # F
    [ 0, -3, -1, -2, -3,  6, -2, -4, -2, -4, -3,  0, -2, -2, -2,  0, -2, -3, -2, -3],  # G
    [-2, -3, -1,  0, -1, -2,  8, -3, -1, -3, -2,  1, -2,  0,  0, -1, -2, -3, -2,  2],  # H
    [-1, -1, -3, -3,  0, -4, -3,  4, -3,  2,  1, -3, -3, -3, -3, -2, -1,  3, -3, -1],  # I
    [-1, -3, -1,  1, -3, -2, -1, -3,  5, -2, -1,  0, -1,  1,  2,  0, -1, -2, -3, -2],  # K
    [-1, -1, -4, -3,  0, -4, -3,  2, -2,  4,  2, -3, -3, -2, -2, -2, -1,  1, -2, -1],  # L
    [-1, -1, -3, -2,  0, -3, -2,  1, -1,  2,  5, -2, -2,  0, -1, -1, -1,  1, -1, -1],  # M
    [-2, -3,  1,  0, -3,  0,  1, -3,  0, -3, -2,  6, -2,  0,  0,  1,  0, -3, -4, -2],  # N
    [-1, -3, -1, -1, -4, -2, -2, -3, -1, -3, -2, -2,  7, -1, -2, -1, -1, -2, -4, -3],  # P
    [-1, -3,  0,  2, -3, -2,  0, -3,  1, -2,  0,  0, -1,  5,  1,  0, -1, -2, -2, -1],  # Q
    [-1, -3, -2,  0, -3, -2,  0, -3,  2, -2, -1,  0, -2,  1,  5, -1, -1, -3, -3, -2],  # R
    [ 1, -1,  0,  0, -2,  0, -1, -2,  0, -2, -1,  1, -1,  0, -1,  4,  1, -2, -3, -2],  # S
    [ 0, -1, -1, -1, -2, -2, -2, -1, -1, -1, -1,  0, -1, -1, -1,  1,  5,  0, -2, -2],  # T
    [ 0, -1, -3, -2, -1, -3, -3,  3, -2,  1,  1, -3, -2, -2, -3, -2,  0,  4, -3, -1],  # V
    [-3, -2, -4, -3,  1, -2, -2, -3, -3, -2, -1, -4, -4, -2, -3, -3, -2, -3, 11,  2],  # W
    [-2, -2, -3, -2,  3, -3,  2, -1, -2, -1, -1, -2, -3, -1, -2, -2, -2, -1,  2,  7],  # Y
], dtype=np.float64)


@pytest.fixture
def mat():
    return nwgrad.SubstMatrix(BLOSUM62)


def test_round_trip(mat):
    recovered = mat.to_matrix()
    assert recovered.shape == (20, 20)
    assert recovered.dtype == np.float64
    np.testing.assert_array_equal(recovered, BLOSUM62)


def test_score_diagonal(mat):
    for i, aa in enumerate(AA_ORDER):
        expected = BLOSUM62[i, i]
        assert mat.score(aa, aa) == pytest.approx(expected), f"diagonal mismatch for {aa}"


def test_score_spot_checks(mat):
    # A-C: 0, W-W: 11, D-E: 2
    assert mat.score('A', 'C') == pytest.approx(0.0)
    assert mat.score('W', 'W') == pytest.approx(11.0)
    assert mat.score('D', 'E') == pytest.approx(2.0)


def test_blosum62_is_symmetric(mat):
    # BLOSUM62 happens to be symmetric; verify round-trip preserves that.
    for i, a in enumerate(AA_ORDER):
        for j, b in enumerate(AA_ORDER):
            assert mat.score(a, b) == pytest.approx(mat.score(b, a)), \
                f"BLOSUM62 symmetry failed for ({a}, {b})"


def test_asymmetric_matrix():
    # SubstMatrix must preserve asymmetric values exactly.
    arr = np.zeros((20, 20), dtype=np.float64)
    arr[0, 1] = 3.0   # A→C
    arr[1, 0] = 7.0   # C→A
    sm = nwgrad.SubstMatrix(arr)
    assert sm.score('A', 'C') == pytest.approx(3.0)
    assert sm.score('C', 'A') == pytest.approx(7.0)


def test_score_bad_input(mat):
    with pytest.raises(Exception):
        mat.score('AA', 'C')
    with pytest.raises(Exception):
        mat.score('A', '')


def test_independent_instances():
    # Modifying the source array after construction must not affect SubstMatrix.
    arr = BLOSUM62.copy()
    sm = nwgrad.SubstMatrix(arr)
    arr[0, 0] = 999.0
    assert sm.score('A', 'A') == pytest.approx(4.0)
