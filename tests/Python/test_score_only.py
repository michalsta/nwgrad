"""Score only: the convenience score functions, align() and score_and_grad() in
grad_mode="none" run Aligner::compute_score — no table, no traceback.

The contract, checked here through the Python API:
  * the score is the exact optimum, bit-identical to a traceback="pointers" pair's
    (also where float32's `auto` default, hirschberg_pmax, would replay a path's score);
  * a none-mode batch's guides are deferred, and resolve to exactly the guides the eager
    path stores — under the params SCORED, even after set_params();
  * banded_grad() after a none-mode score_and_grad() equals the eager path's.
Lengths straddle hb_cutoff (512), where the float32 default switches to the prefix-max
carry, and the inter-pair groups (short pairs) as well as the per-pair kernel are hit.
"""
import numpy as np
import pytest

import nwgrad

PRECISIONS = [("SeqPair", "SeqPairBatch", ""), ("SeqPairDouble", "SeqPairBatchDouble", "_double")]
TYPES = [("affine", "global"), ("affine", "local"), ("linear", "global"), ("linear", "local")]
SCORE_FN = {("affine", "global"): "nw_score_affine", ("affine", "local"): "sw_score_affine",
            ("linear", "global"): "nw_score", ("linear", "local"): "sw_score"}


def _params(alpha, rng, ge=0.3):
    n = len(alpha)
    m = rng.normal(-0.6, 0.6, (n, n))
    np.fill_diagonal(m, rng.normal(2.0, 0.3, n))
    return nwgrad.AlignParams(nwgrad.SubstMatrix(m, alphabet=alpha), 3.1, ge, 2.7, ge * 1.3)


def _pairs(alpha, rng, count, lo, hi):
    A, B = [], []
    for _ in range(count):
        a = "".join(rng.choice(list(alpha), int(rng.integers(lo, hi + 1))))
        b = list(a)
        for _ in range(max(1, len(a) // 8)):          # mutate: substitutions and indels
            k = int(rng.integers(0, max(1, len(b))))
            r = rng.random()
            if r < 0.5 and b:
                b[min(k, len(b) - 1)] = str(rng.choice(list(alpha)))
            elif r < 0.75 and b:
                del b[min(k, len(b) - 1)]
            else:
                b.insert(k, str(rng.choice(list(alpha))))
        A.append(a); B.append("".join(b))
    return A, B


def _pointer_scores(sp_cls, A, B, p, gm, md):
    cls = getattr(nwgrad, sp_cls)
    out = []
    for a, b in zip(A, B):
        sp = cls(a, b, p, gap_model=gm, mode=md, grad_mode="hard", traceback="pointers")
        sp.score_and_grad()
        out.append(sp.score)
    return out


@pytest.mark.parametrize("sp_cls,spb_cls,sfx", PRECISIONS, ids=["float32", "double"])
@pytest.mark.parametrize("gm,md", TYPES)
@pytest.mark.parametrize("alpha", ["ACGT", "ACDEFGHIKLMNPQRSTVWY"], ids=["dna", "protein"])
def test_convenience_score_is_the_exact_optimum(sp_cls, spb_cls, sfx, gm, md, alpha):
    rng = np.random.default_rng(1)
    p = _params(alpha, rng)
    A, B = _pairs(alpha, rng, 6, 1, 90)
    A2, B2 = _pairs(alpha, rng, 2, 520, 700)              # past hb_cutoff
    A, B = A + A2 + ["", alpha[:4]], B + B2 + ["A", ""]
    fn = getattr(nwgrad, SCORE_FN[(gm, md)] + sfx)
    got = [fn(a, b, p) for a, b in zip(A, B)]
    assert got == _pointer_scores(sp_cls, A, B, p, gm, md)


@pytest.mark.parametrize("sp_cls,spb_cls,sfx", PRECISIONS, ids=["float32", "double"])
@pytest.mark.parametrize("gm,md", TYPES)
@pytest.mark.parametrize("fill", ["interpair", "striped"])
def test_align_none_is_the_exact_optimum(sp_cls, spb_cls, sfx, gm, md, fill):
    rng = np.random.default_rng(2)
    p = _params("ACGT", rng, ge=0.1)
    A, B = _pairs("ACGT", rng, 40, 1, 60)                  # inter-pair groups
    A2, B2 = _pairs("ACGT", rng, 3, 530, 600)              # per-pair kernel, past hb_cutoff
    A, B = A + A2, B + B2
    batch = getattr(nwgrad, spb_cls)(3, gap_model=gm, mode=md, grad_mode="none")
    batch.fill = fill
    r = batch.align(A, B, p)
    assert list(r.scores) == _pointer_scores(sp_cls, A, B, p, gm, md)


@pytest.mark.parametrize("sp_cls,spb_cls,sfx", PRECISIONS, ids=["float32", "double"])
@pytest.mark.parametrize("gm,md", TYPES)
@pytest.mark.parametrize("fill", ["interpair", "striped"])
def test_batch_none_scores_and_deferred_guides(sp_cls, spb_cls, sfx, gm, md, fill):
    rng = np.random.default_rng(3)
    alpha = "ACGT"
    p0, p1 = _params(alpha, rng, ge=0.1), _params(alpha, rng, ge=0.4)
    A, B = _pairs(alpha, rng, 30, 1, 60)
    A2, B2 = _pairs(alpha, rng, 2, 520, 560)
    A, B = A + A2, B + B2

    def make():
        b = getattr(nwgrad, spb_cls)(2, gap_model=gm, mode=md, grad_mode="none")
        b.fill = fill
        b.add_many(A, B, p0)
        return b

    lazy, eager = make(), make()
    lazy.score_and_grad()                          # score only, guides deferred
    eager.score_and_grad(keep_paths=True)          # the traceback path, guides stored now

    assert list(lazy.scores()) == _pointer_scores(sp_cls, A, B, p0, gm, md)

    # The deferred guides belong to the params SCORED (p0), even after set_params(p1).
    lazy.set_params(p1)
    eager.set_params(p1)
    for i in range(len(A)):
        assert lazy[i].guide_j == eager[i].guide_j, i

    # ... so the banded step around them is the eager path's, bit for bit.
    lazy2, eager2 = make(), make()
    lazy2.score_and_grad()
    eager2.score_and_grad(keep_paths=True)
    lazy2.set_params(p1)
    eager2.set_params(p1)
    assert lazy2.banded_grad(4) == eager2.banded_grad(4)
    assert list(lazy2.scores()) == list(eager2.scores())


def test_score_only_ignores_stored_path_state():
    """A none-mode score_and_grad() after a keep_paths one drops the stored paths (their
    scores are recomputed, their alignments are not)."""
    rng = np.random.default_rng(4)
    p = _params("ACGT", rng)
    A, B = _pairs("ACGT", rng, 5, 5, 30)
    b = nwgrad.SeqPairBatchDouble(1, gap_model="affine", mode="global", grad_mode="none")
    b.add_many(A, B, p)
    b.score_and_grad(keep_paths=True)
    b[0].aligned()
    b.score_and_grad()
    with pytest.raises(Exception):
        b[0].aligned()
