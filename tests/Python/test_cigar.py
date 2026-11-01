"""SeqPair.cigar() / SeqPairBatch.cigars(): the alignment as a SAM CIGAR string.

seq_a is the query, seq_b the reference.  Checked against a CIGAR rebuilt in Python
from aligned() and coordinates(), for every gap model, mode and traceback, plus a few
hand-checked cases.
"""

import re
from itertools import groupby

import numpy as np
import pytest

import nwgrad

DNA = "ACGT"


def _seqs(n, lo, hi, seed):
    rng = np.random.default_rng(seed)
    return ["".join(rng.choice(list(DNA), int(l))) for l in rng.integers(lo, hi, n)]


def _params(seed, go, ge):
    m = np.random.default_rng(seed).normal(size=(4, 4))
    return nwgrad.AlignParams(nwgrad.SubstMatrix(m, alphabet=DNA), gap_open_a=go,
                              gap_extend_a=ge, gap_open_b=go, gap_extend_b=ge)


def _identity(go=2.0, ge=1.0):
    m = np.where(np.eye(4, dtype=bool), 2.0, -3.0)
    return nwgrad.AlignParams(nwgrad.SubstMatrix(m, alphabet=DNA), gap_open_a=go,
                              gap_extend_a=ge, gap_open_b=go, gap_extend_b=ge)


def _expected(sp, extended, soft_clip):
    sa, sb = sp.aligned()
    ops = []
    for x, y in zip(sa, sb):
        if y == "-":
            ops.append("I")
        elif x == "-":
            ops.append("D")
        else:
            ops.append(("=" if x == y else "X") if extended else "M")
    c = sp.coordinates()
    if soft_clip:
        ops = ["S"] * int(c[0, 0]) + ops + ["S"] * (len(sp.seq_a) - int(c[0, -1]))
    return "".join(f"{len(list(g))}{op}" for op, g in groupby(ops))


def _consumed(cigar):
    q = r = 0
    for n, op in re.findall(r"(\d+)([MIDSX=])", cigar):
        q += int(n) if op in "MI=XS" else 0
        r += int(n) if op in "MD=X" else 0
    return q, r


@pytest.mark.parametrize("gap_model", ["affine", "linear"])
@pytest.mark.parametrize("mode", ["local", "global"])
@pytest.mark.parametrize("traceback", ["pointers", "scores", "hirschberg"])
def test_matches_aligned(gap_model, mode, traceback):
    if traceback == "hirschberg" and gap_model == "linear":
        pytest.skip("Hirschberg is affine only")
    p = _params(1, 1.5 if gap_model == "affine" else 0.0, 0.6)
    for a, b in zip(_seqs(60, 1, 40, 2), _seqs(60, 1, 60, 3)):
        sp = nwgrad.SeqPairDouble(a, b, p, gap_model=gap_model, mode=mode, traceback=traceback)
        sp.align()
        for ext in (False, True):
            for clip in (False, True):
                cg = sp.cigar(extended=ext, soft_clip=clip)
                assert cg == _expected(sp, ext, clip)
                assert re.fullmatch(r"(\d+[MIDSX=])*", cg)
                q, r = _consumed(cg)
                c = sp.coordinates()
                assert r == c[1, -1] - c[1, 0]
                assert q == (len(a) if clip else c[0, -1] - c[0, 0])
        if mode == "global":
            assert _consumed(sp.cigar()) == (len(a), len(b))
            assert sp.cigar(soft_clip=True) == sp.cigar()


def test_hand_checked():
    p = _identity()
    sp = nwgrad.SeqPairDouble("ACGTACGT", "ACGTACGT", p)
    sp.align()
    assert sp.cigar() == "8M" and sp.cigar(extended=True) == "8="

    sp = nwgrad.SeqPairDouble("ACGTTTACGT", "ACGTACGT", p)   # TT extra in the query
    sp.align()
    assert sp.cigar() == "3M2I5M"           # the gap slides; this tie-break puts it first

    sp = nwgrad.SeqPairDouble("ACGTACGT", "ACGTTTACGT", p)   # TT missing from the query
    sp.align()
    assert sp.cigar() == "3M2D5M"

    sp = nwgrad.SeqPairDouble("ACGAACGT", "ACGTACGT", p)
    sp.align()
    assert sp.cigar() == "8M" and sp.cigar(extended=True) == "3=1X4="


def test_local_soft_clip():
    p = _identity(go=4.0, ge=2.0)
    sp = nwgrad.SeqPairDouble("TTTTGATTACAGGGG", "CCGATTACACC", p, mode="local")
    sp.align()
    assert sp.cigar() == "7M"
    assert sp.cigar(soft_clip=True) == "4S7M4S"
    assert sp.coordinates()[:, 0].tolist() == [4, 2]


def test_needs_a_stored_path():
    sp = nwgrad.SeqPairDouble("ACGT", "ACGA", _identity())
    with pytest.raises(RuntimeError):
        sp.cigar()
    sp.align(keep_paths=False)
    with pytest.raises(RuntimeError):
        sp.cigar()
    sp.align()
    sp.cigar()
    sp.drop_paths()
    with pytest.raises(RuntimeError):
        sp.cigar()


@pytest.mark.parametrize("mode", ["local", "global"])
def test_batch_cigars(mode):
    p = _params(4, 1.5, 0.6)
    sa, sb = _seqs(50, 1, 40, 5), _seqs(50, 1, 50, 6)
    batch = nwgrad.SeqPairBatchDouble(sa, sb, p, mode=mode, n_threads=2)
    batch.score_and_grad(keep_paths=True)
    for ext in (False, True):
        for clip in (False, True):
            got = batch.cigars(extended=ext, soft_clip=clip)
            assert got == [batch[i].cigar(extended=ext, soft_clip=clip) for i in range(len(sa))]
    batch.drop_paths()
    with pytest.raises(RuntimeError):
        batch.cigars()
