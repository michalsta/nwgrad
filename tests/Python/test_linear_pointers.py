"""traceback="pointers" for the linear gap model (Aligner::viterbi_linear_ptr).

It retains one direction byte per cell instead of the H table, and must walk exactly
the path the H walk ("scores") re-derives by equality — score, alignment strings,
guide and gradient bit-identical, ties included.  Since it is also the default
("auto" resolves to pointers for linear), every other linear test exercises it too.
"""

import numpy as np
import pytest

import nwgrad

DNA = "ACGT"
CLS = {"double": (nwgrad.SeqPairDouble, nwgrad.SeqPairBatchDouble),
       "float32": (nwgrad.SeqPair, nwgrad.SeqPairBatch)}


def _params(kind):
    if kind == "ties":
        m = np.full((4, 4), -1.0)
        np.fill_diagonal(m, 2.0)
        ge = 1.0
    elif kind == "unrepresentable":
        m = np.random.default_rng(3).normal(size=(4, 4))
        ge = 0.1
    else:
        m = np.random.default_rng(4).normal(size=(4, 4))
        ge = 0.7
    return nwgrad.AlignParams(nwgrad.SubstMatrix(m, alphabet=DNA), 0.0, ge, 0.0, ge * 1.3)


def _seqs(n, lo, hi, seed):
    rng = np.random.default_rng(seed)
    return ["".join(rng.choice(list(DNA), int(k))) for k in rng.integers(lo, hi, n)]


def _pair(cls, a, b, p, mode, tb):
    sp = cls(a, b, p, gap_model="linear", mode=mode, grad_mode="hard", traceback=tb)
    sp.alloc_dp()
    sp.align_full()
    sp.compute_grad()
    g = sp.grad
    return (sp.score, sp.aligned(), list(sp.guide_j), np.asarray(g.matrix.to_matrix()),
            (g.gap_extend_a, g.gap_extend_b))


@pytest.mark.parametrize("prec", ["double", "float32"])
@pytest.mark.parametrize("mode", ["global", "local"])
@pytest.mark.parametrize("kind", ["ties", "unrepresentable", "random"])
def test_pointers_equal_scores(prec, mode, kind):
    A, B = _seqs(60, 0, 40, 1), _seqs(60, 0, 40, 2)
    p = _params(kind)
    for a, b in zip(A, B):
        x = _pair(CLS[prec][0], a, b, p, mode, "scores")
        y = _pair(CLS[prec][0], a, b, p, mode, "pointers")
        assert x[0] == y[0] and x[1] == y[1] and x[2] == y[2] and x[4] == y[4]
        assert np.array_equal(x[3], y[3])


def test_auto_resolves_to_pointers():
    sp = nwgrad.SeqPairDouble("ACGT", "AGT", _params("ties"), gap_model="linear", mode="local")
    assert sp.traceback == "pointers"


@pytest.mark.parametrize("prec", ["double", "float32"])
@pytest.mark.parametrize("mode", ["global", "local"])
def test_batch_pointers_equal_scores(prec, mode):
    A, B = _seqs(200, 0, 120, 5), _seqs(200, 0, 120, 6)
    p = _params("ties")
    out = []
    for tb in ("scores", "pointers"):
        b = CLS[prec][1](n_threads=3, traceback=tb)
        b.add_many(A, B, p, gap_model="linear", mode=mode, grad_mode="hard")
        b.score_and_grad()
        out.append((b.scores(), *b.grads(), [list(b[i].guide_j) for i in range(len(b))]))
    for x, y in zip(out[0][:3], out[1][:3]):
        assert np.array_equal(x, y)
    assert out[0][3] == out[1][3]


@pytest.mark.parametrize("mode", ["global", "local"])
def test_auto_switches_fill_by_size_same_path(mode):
    """Under "auto" a small linear pair keeps H (faster), a large one direction bytes
    (past 2 MiB); explicit "pointers" always the latter.  The path is the same."""
    p = _params("ties")
    for L in (40, 600):   # H table: 13 KB / 2.9 MB at double
        a, b = _seqs(2, L, L + 1, 7 + L)
        out = []
        for tb in ("auto", "pointers", "scores"):
            sp = nwgrad.SeqPairDouble(a, b, p, gap_model="linear", mode=mode, traceback=tb)
            sp.alloc_dp(); sp.align_full()
            out.append((sp.score, sp.aligned(), list(sp.guide_j)))
        assert out[0] == out[1] == out[2]
