"""Manual tests for the soft gradient where the expected value is derived
analytically — without running the DP implementation as oracle.

Three kinds of tests
--------------------
1. Single-character global: only one path exists, grad[X][Y] == 1.0 exactly.

2. Brute-force path enumeration for short sequences (global linear only):
   Recursively enumerate every valid global-NW path, compute its score, derive
   the soft gradient as expected (weighted) counts.  This is O(3^(m+n)) but is
   completely independent of the forward–backward implementation.

3. Closed-form formula for "AA…A" vs "AA…A" with an all-zero substitution
   matrix (global linear, length 2):
     Paths: 1 with score 0 (D D), 6 with score -2g, 6 with score -4g.
     Expected diagonal steps = (2 + 6 e^{-2g}) / (1 + 6 e^{-2g} + 6 e^{-4g})
"""

import math
import itertools
import numpy as np
import pytest
import nwgrad

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"

# ── helpers ───────────────────────────────────────────────────────────────────

def make_mat(arr20):
    return nwgrad.SubstMatrix(np.asarray(arr20, dtype=np.float64))


def zero_mat():
    return make_mat(np.zeros((20, 20)))


def unit_mat():
    """Diagonal = 1, off-diagonal = 0."""
    m = np.zeros((20, 20))
    np.fill_diagonal(m, 1.0)
    return make_mat(m)


def aa(c):
    return AA_ORDER.index(c)


def score_from_mat20(arr20, c1, c2):
    return float(arr20[aa(c1), aa(c2)])


# ── brute-force reference (global linear NW) ─────────────────────────────────

def brute_force_nw_linear_soft_grad(seq_a, seq_b, mat20, gap_extend):
    """Enumerate every global-NW path and return (log_Z, expected_counts dict).

    A path is a list of moves: 'D' (diagonal), 'V' (vertical — gap in seq_b),
    'H' (horizontal — gap in seq_a).  From (0,0) to (m,n), where m=len(a),
    n=len(b), every move must advance at least one index.

    Score:
        +s(a[i], b[j])  for each D step at position (i,j)
        -gap_extend      for each V or H step

    This is O(3^(m+n)) — only valid for very short sequences.
    """
    m, n = len(seq_a), len(seq_b)

    def s(c1, c2):
        return score_from_mat20(mat20, c1, c2)

    # Enumerate paths via DFS; returns list of (score, list_of_diag_pairs)
    results = []

    def dfs(i, j, score, pairs):
        if i == m and j == n:
            results.append((score, list(pairs)))
            return
        # Diagonal
        if i < m and j < n:
            pairs.append((seq_a[i], seq_b[j]))
            dfs(i + 1, j + 1, score + s(seq_a[i], seq_b[j]), pairs)
            pairs.pop()
        # Vertical (gap in seq_b, consume seq_a[i])
        if i < m:
            dfs(i + 1, j, score - gap_extend, pairs)
        # Horizontal (gap in seq_a, consume seq_b[j])
        if j < n:
            dfs(i, j + 1, score - gap_extend, pairs)

    dfs(0, 0, 0.0, [])

    # Compute log-partition in a numerically stable way
    scores = [r[0] for r in results]
    max_s = max(scores)
    log_z = max_s + math.log(sum(math.exp(s - max_s) for s in scores))

    # Expected counts
    expected = {}
    for score, pairs in results:
        w = math.exp(score - log_z)
        for c1, c2 in pairs:
            expected[(c1, c2)] = expected.get((c1, c2), 0.0) + w

    return log_z, expected


def grad_to_dict(grad_arr):
    """Convert a (20,20) numpy grad array to {(c1,c2): value} for nonzero entries."""
    d = {}
    for i, c1 in enumerate(AA_ORDER):
        for j, c2 in enumerate(AA_ORDER):
            v = float(grad_arr[i, j])
            if abs(v) > 1e-15:
                d[(c1, c2)] = v
    return d


# ══════════════════════════════════════════════════════════════════════════════
# 1. Single-character global alignment — closed-form expected values
#
#   For length-1 sequences there are exactly 3 global-NW paths:
#     1. Diagonal:   (0,0)→(1,1),         score = s(X,Y)
#     2. V then H:   (0,0)→(1,0)→(1,1),   score = −2·gap_extend   (linear)
#                                                = −2·(go+ge)       (affine)
#     3. H then V:   (0,0)→(0,1)→(1,1),   score = −2·gap_extend   (linear)
#                                                = −2·(go+ge)       (affine)
#
#   Z     = log( exp(s) + 2·exp(−2g) )      (linear, g = gap_extend)
#         = log( exp(s) + 2·exp(−2(go+ge)) ) (affine)
#   grad[X][Y] = exp(s) / (exp(s) + 2·exp(−2g))
# ══════════════════════════════════════════════════════════════════════════════

from test_subst_matrix import BLOSUM62

SINGLE_CHAR_GLOBAL = [
    # (seq_a, seq_b, pair)
    ("A", "A", ("A", "A")),
    ("A", "C", ("A", "C")),
    ("W", "Y", ("W", "Y")),
    ("G", "G", ("G", "G")),
    ("K", "R", ("K", "R")),
]


@pytest.mark.parametrize("a,b,pair", SINGLE_CHAR_GLOBAL)
def test_single_char_nw_linear_closed_form(a, b, pair):
    """Single-char global linear: closed-form Z and grad."""
    ge = 1.0
    s = score_from_mat20(BLOSUM62, a, b)
    denom     = math.exp(s) + 2 * math.exp(-2 * ge)
    z_exp     = math.log(denom)
    grad_exp  = math.exp(s) / denom

    log_z, g = nwgrad.nw_soft_grad(a, b, make_mat(BLOSUM62), gap_extend=ge)
    g = np.array(g)

    assert log_z == pytest.approx(z_exp, rel=1e-9)
    assert g[aa(pair[0]), aa(pair[1])] == pytest.approx(grad_exp, rel=1e-9)
    # Only the one pair can appear in a length-1 alignment.
    other = g.copy()
    other[aa(pair[0]), aa(pair[1])] = 0.0
    assert other.sum() == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("a,b,pair", SINGLE_CHAR_GLOBAL)
def test_single_char_nw_affine_closed_form(a, b, pair):
    """Single-char global affine: closed-form Z and grad."""
    go, ge = 11.0, 1.0
    s = score_from_mat20(BLOSUM62, a, b)
    denom    = math.exp(s) + 2 * math.exp(-2 * (go + ge))
    z_exp    = math.log(denom)
    grad_exp = math.exp(s) / denom

    log_z, g = nwgrad.nw_affine_soft_grad(
        a, b, make_mat(BLOSUM62), gap_open=go, gap_extend=ge
    )
    g = np.array(g)

    assert log_z == pytest.approx(z_exp, rel=1e-9)
    assert g[aa(pair[0]), aa(pair[1])] == pytest.approx(grad_exp, rel=1e-9)
    other = g.copy()
    other[aa(pair[0]), aa(pair[1])] = 0.0
    assert other.sum() == pytest.approx(0.0, abs=1e-12)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Closed-form: "AA" vs "AA", all-zero matrix, global linear
#
#   All paths from (0,0) to (2,2) and their scores (gap_extend = g):
#     1 path  with 2 diagonals: score 0
#     6 paths with 1 diagonal:  score -2g
#     6 paths with 0 diagonals: score -4g
#
#   Z     = log(1 + 6 e^{-2g} + 6 e^{-4g})
#   E[#D] = (2 + 6 e^{-2g}) / (1 + 6 e^{-2g} + 6 e^{-4g})
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("g", [0.5, 1.0, 2.0, 5.0])
def test_closed_form_aa_vs_aa_zero_matrix(g):
    """'AA' vs 'AA', all-zero matrix, global linear: closed-form expected counts."""
    e2 = math.exp(-2 * g)
    e4 = math.exp(-4 * g)
    z_expected    = math.log(1 + 6 * e2 + 6 * e4)
    grad_expected = (2 + 6 * e2) / (1 + 6 * e2 + 6 * e4)

    mat = zero_mat()
    log_z, grad = nwgrad.nw_soft_grad("AA", "AA", mat, gap_extend=g)
    grad = np.array(grad)

    assert log_z == pytest.approx(z_expected, rel=1e-9)
    assert grad[aa("A"), aa("A")] == pytest.approx(grad_expected, rel=1e-9)
    # All non-A entries must be zero since the sequences contain only A.
    non_a = grad.copy()
    non_a[aa("A"), aa("A")] = 0.0
    assert non_a.sum() == pytest.approx(0.0, abs=1e-12)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Brute-force vs implementation (global linear, short sequences)
# ══════════════════════════════════════════════════════════════════════════════

BRUTE_FORCE_CASES = [
    # (seq_a, seq_b, mat20_label, gap_extend)
    ("A",   "A",    "zero",   1.0),
    ("A",   "C",    "zero",   1.0),
    ("AC",  "AC",   "zero",   1.0),
    ("AC",  "CA",   "zero",   1.0),
    ("AG",  "GA",   "unit",   1.0),
    ("AC",  "AC",   "unit",   0.5),
    ("ACD", "ACD",  "zero",   1.0),
    ("ACD", "CDA",  "zero",   2.0),
    ("ACD", "ACD",  "unit",   2.0),
    # non-trivial BLOSUM62 values, very short
    ("A",   "A",    "blosum", 1.0),
    ("AC",  "AC",   "blosum", 1.0),
    ("AC",  "CA",   "blosum", 2.0),
]


def _mat_by_label(label):
    if label == "zero":
        return np.zeros((20, 20))
    if label == "unit":
        m = np.zeros((20, 20))
        np.fill_diagonal(m, 1.0)
        return m
    if label == "blosum":
        return BLOSUM62.copy()
    raise ValueError(label)


@pytest.mark.parametrize(
    "a,b,mat_label,gap", BRUTE_FORCE_CASES,
    ids=[f"{a}|{b}|{ml}|g{g}" for a, b, ml, g in BRUTE_FORCE_CASES],
)
def test_brute_force_nw_linear(a, b, mat_label, gap):
    """Compare nw_soft_grad against full path enumeration for short sequences."""
    mat20 = _mat_by_label(mat_label)

    ref_logz, ref_counts = brute_force_nw_linear_soft_grad(a, b, mat20, gap)
    impl_logz, impl_grad = nwgrad.nw_soft_grad(a, b, make_mat(mat20), gap_extend=gap)
    impl_dict = grad_to_dict(np.array(impl_grad))

    assert impl_logz == pytest.approx(ref_logz, rel=1e-9, abs=1e-12), \
        f"log_Z mismatch: impl={impl_logz:.8g}, ref={ref_logz:.8g}"

    # Every pair in the reference should match the implementation.
    all_pairs = set(ref_counts) | set(impl_dict)
    for pair in all_pairs:
        ref_v  = ref_counts.get(pair, 0.0)
        impl_v = impl_dict.get(pair, 0.0)
        assert impl_v == pytest.approx(ref_v, rel=1e-9, abs=1e-12), \
            f"grad[{pair}]: impl={impl_v:.8g}, ref={ref_v:.8g}"


# ══════════════════════════════════════════════════════════════════════════════
# 4. Single-character local (SW): soft log_Z and expected count
#
#   For SW soft with seq "X" and "X", gap_extend=g, substitution s=s(X,X):
#
#   F[i][j] table (2x2 — including boundary row/col):
#     F[0][0] = F[1][0] = F[0][1] = 0   (free start)
#     F[1][1] = lse(F[0][0]+s, F[0][1]-g, F[1][0]-g, 0)
#             = lse(s, -g, -g, 0)
#             = lse(s, 0)   [for s > 0, g > 0]
#
#   Z = lse(F[0][0], F[1][0], F[0][1], F[1][1])
#     = lse(0, 0, 0, lse(s, 0))
#     = lse(0, lse(s, 0))   since lse(0,0,0,x) = lse(lse(0,0,0), x) = lse(log(3), x)
#
#   Expected count grad[X][X]:
#     = exp(F[0][0] + s + B[1][1] - Z)   where B[1][1] = 0
#     = exp(s - Z)
#
#   Z = log(3 + exp(lse(s,0))) = log(3 + exp(s) + 1) = log(4 + exp(s))
#   grad[X][X] = exp(s - log(4 + exp(s))) = exp(s) / (4 + exp(s))
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("char,s_val,g", [
    ("A", 4.0, 1.0),   # BLOSUM62 s(A,A) = 4
    ("C", 9.0, 1.0),   # BLOSUM62 s(C,C) = 9
    ("W", 11.0, 2.0),  # BLOSUM62 s(W,W) = 11
])
def test_single_char_sw_linear_closed_form(char, s_val, g):
    """SW soft, single char 'X' vs 'X': closed-form for log_Z and grad[X][X].

    F table (with free-start boundary):
      F[0][0] = F[1][0] = F[0][1] = 0
      F[1][1] = lse(s, -g, -g, 0)     <- lse includes free-restart 0

    Z = lse(0, 0, 0, F[1][1])
      = log(3 + exp(F[1][1]))
      = log(3 + exp(s) + 2*exp(-g) + 1)   [exp(lse(s,-g,-g,0)) = exp(s)+2*exp(-g)+1]
      = log(4 + exp(s) + 2*exp(-g))

    grad[X][X] = exp(F[0][0] + s + B[1][1] - Z)  where B[1][1] = 0 (free end)
               = exp(s) / (4 + exp(s) + 2*exp(-g))
    """
    denom         = 4 + math.exp(s_val) + 2 * math.exp(-g)
    z_expected    = math.log(denom)
    grad_expected = math.exp(s_val) / denom

    mat20 = np.zeros((20, 20))
    mat20[aa(char), aa(char)] = s_val

    log_z, grad = nwgrad.sw_soft_grad(char, char, make_mat(mat20), gap_extend=g)
    grad = np.array(grad)

    assert log_z == pytest.approx(z_expected, rel=1e-9)
    assert grad[aa(char), aa(char)] == pytest.approx(grad_expected, rel=1e-9)
    # All other entries zero (sequences contain only one distinct char).
    other = grad.copy()
    other[aa(char), aa(char)] = 0.0
    assert other.sum() == pytest.approx(0.0, abs=1e-12)


# ══════════════════════════════════════════════════════════════════════════════
# 5. Disjoint alphabets — alignments are uniquely determined
#
#   "AC" vs "DG" with a matrix where s(A,D)=a, s(C,G)=b, all other entries=0,
#   and gap_extend very large.  With no gaps possible, the only surviving path
#   is the diagonal, so:
#     Z ≈ a + b
#     grad[A][D] = 1,  grad[C][G] = 1
#
#   We verify the exact closed form without the large-gap approximation:
#
#   F[0][0]=0, F[1][0]=-g, F[2][0]=-2g, F[0][1]=-g, F[0][2]=-2g
#   F[1][1] = lse(s(A,D), -2g, -2g)
#   F[1][2] = lse(F[0][1]+s(A,G), F[0][2]-g, F[1][1]-g)
#           = lse(-g+0, -3g, F[1][1]-g)   [s(A,G)=0]
#   F[2][1] = lse(F[1][0]+s(C,D), F[1][1]-g, F[2][0]-g)
#           = lse(-g+0, F[1][1]-g, -3g)   [s(C,D)=0]
#   F[2][2] = lse(F[1][1]+s(C,G), F[1][2]-g, F[2][1]-g)
#
#   Backward:
#   B[2][2]=0
#   B[1][1] gets: B[2][2] + s(C,G) = b  (from diagonal (1,1)→(2,2))
#             + B[1][2] - g via j-decrease, B[2][1] - g via i-decrease
#
# Rather than manually solving the full backward, we verify:
#   (a) grad[A][D] + grad[C][G] ≈ 2  (both positions counted)
#   (b) all other grad entries ≈ 0   (no off-diagonal substitutions occur)
# for very large g (where only the diagonal path survives).
# ══════════════════════════════════════════════════════════════════════════════

def test_disjoint_alphabet_large_gap_nw_linear():
    """'AC' vs 'DG', sparse matrix, large gap: both pairs counted once."""
    a_val, b_val = 3.0, 5.0
    g = 100.0

    mat20 = np.zeros((20, 20))
    mat20[aa("A"), aa("D")] = a_val
    mat20[aa("D"), aa("A")] = a_val  # symmetry
    mat20[aa("C"), aa("G")] = b_val
    mat20[aa("G"), aa("C")] = b_val

    log_z, grad = nwgrad.nw_soft_grad("AC", "DG", make_mat(mat20), gap_extend=g)
    grad = np.array(grad)

    # With g=100 the gap paths have negligible weight, so grad ≈ hard subgradient.
    assert grad[aa("A"), aa("D")] == pytest.approx(1.0, abs=1e-6)
    assert grad[aa("C"), aa("G")] == pytest.approx(1.0, abs=1e-6)
    assert grad.sum() == pytest.approx(2.0, abs=1e-6)
    assert log_z == pytest.approx(a_val + b_val, abs=1e-6)


def test_disjoint_alphabet_exact_nw_linear():
    """'AC' vs 'DG', sparse matrix: analytically exact log_Z and gradient.

    With s(A,D)=a, s(C,G)=b, all other entries=0, gap_extend=g:

    F[1][1] = lse(a, -2g)
    F[1][2] = lse(-g, lse(a,-2g)-g) = lse(-g, F[1][1]-g)
    F[2][1] = lse(-g, F[1][1]-g)                          (same by symmetry of 0s)
    F[2][2] = lse(F[1][1]+b, F[1][2]-g, F[2][1]-g)
            = lse(F[1][1]+b, 2*(lse(-g, F[1][1]-g)-g))

    Backward from B[2][2]=0:
      B[1][1] = b   (only significant path: diagonal to (2,2) with score b)
        + tiny corrections from gap paths at (1,2) and (2,1) — included numerically

    grad[A][D] = exp(F[0][0] + a + B[1][1] - Z)  = exp(a + b - Z) (approximately)
    grad[C][G] = exp(F[1][1] + b + B[2][2] - Z)  = exp(F[1][1] + b - Z)
               = exp(lse(a,-2g) + b - Z)
    """
    a_val, b_val, g = 3.0, 5.0, 1.0

    mat20 = np.zeros((20, 20))
    mat20[aa("A"), aa("D")] = a_val
    mat20[aa("D"), aa("A")] = a_val
    mat20[aa("C"), aa("G")] = b_val
    mat20[aa("G"), aa("C")] = b_val

    # Compute reference using brute-force enumeration
    ref_logz, ref_counts = brute_force_nw_linear_soft_grad(
        "AC", "DG", mat20, g
    )

    log_z, grad = nwgrad.nw_soft_grad("AC", "DG", make_mat(mat20), gap_extend=g)
    grad = np.array(grad)

    assert log_z == pytest.approx(ref_logz, rel=1e-9)
    for pair, ref_v in ref_counts.items():
        assert grad[aa(pair[0]), aa(pair[1])] == pytest.approx(ref_v, rel=1e-9)
