"""nwgrad.logistic: a binary logistic link over per-pair scores (DiscrimAlign's likelihood)."""

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import expit

import nwgrad
import nwgrad.logistic as L

from test_weighted_grad import FIELDS, SEQS_A, SEQS_B, dna_params


def data(seed=0, n=3 * 4096 + 17, offset=0.0):
    rng = np.random.default_rng(seed)
    s = rng.normal(-3.0, 2.0, n) + offset
    y = (rng.random(n) < expit(s - offset + 1.5)).astype(float)
    return s, y


def root(s, y, center=0.0, width=400.0):
    return brentq(lambda a: np.sum(y - expit(a + s)), center - width, center + width, xtol=1e-14)


def test_probabilities_are_scipy_expit_bit_for_bit():
    s, _ = data(1)
    for alpha in (0.0, -1.3, 7.5):
        assert np.array_equal(L.probabilities(s, alpha), expit(alpha + s))


@pytest.mark.parametrize("start", [0.0, 1e-3, -0.7, 5.0, -50.0, 500.0, -500.0])
def test_fit_alpha_finds_the_root_from_any_start(start):
    s, y = data(3)
    ref = root(s, y)
    assert L.fit_alpha(s, y, ref + start) == pytest.approx(ref, abs=1e-10)


@pytest.mark.parametrize("offset", [300.0, -300.0, 1600.0])
def test_fit_alpha_follows_a_large_jump(offset):
    s, y = data(4, offset=offset)
    assert L.fit_alpha(s, y, 0.0) == pytest.approx(root(s, y, center=-offset), abs=1e-9)


def test_results_do_not_depend_on_threads():
    s, y = data(5)
    results = {L.fit_alpha(s, y, -1.0, n_threads=t) for t in (1, 2, 3, 8)}
    assert len(results) == 1
    assert len({L.probabilities(s, 0.3, n_threads=t).tobytes() for t in (1, 3, 8)}) == 1


def test_soft_labels_fit_the_root_of_the_cross_entropy_gradient():
    """Labels are not checked, and soft labels in [0, 1] are supported as such:
    alpha is the root of sum(y - p) for fractional y too."""
    s, _ = data(8)
    y = np.random.default_rng(8).random(s.size)
    assert L.fit_alpha(s, y, 0.0) == pytest.approx(root(s, y), abs=1e-10)


def test_labels_are_not_checked():
    """Labels are the caller's responsibility: out-of-range or single-class labels
    give a meaningless intercept, not an error."""
    s = np.arange(5.0)
    for y in (np.ones(5), np.array([0.0, 1.0, 2.0, 1.0, 0.0])):
        assert np.isfinite(L.fit_alpha(s, y, 0.0))


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        L.fit_alpha(np.zeros(4), np.array([0.0, 1.0, 0.0]), 0.0)


# ── step(batch, labels, alpha0) ──────────────────────────────────────────────

def many_pairs(n, seed=6):
    rng = np.random.default_rng(seed)
    letters = np.array(list("ACGT"))
    a = ["".join(rng.choice(letters, size=rng.integers(4, 13))) for _ in range(n)]
    b = ["".join(rng.choice(letters, size=rng.integers(4, 17))) for _ in range(n)]
    y = rng.integers(0, 2, n).astype(float)
    y[0], y[1] = 0.0, 1.0
    return a, b, y


@pytest.mark.parametrize("cls", [nwgrad.SeqPairBatch, nwgrad.SeqPairBatchDouble])
def test_step_is_its_parts_composed(cls):
    a, b, y = many_pairs(2 * 4096 + 5)
    batch = cls(n_threads=4)
    batch.add_many(a, b, dna_params(), gap_model="affine", mode="local")
    batch.score_and_grad()
    alpha0 = -0.4
    st = L.step(batch, y, alpha0)
    s = batch.scores()
    assert st.alpha == L.fit_alpha(s, y, alpha0, n_threads=4)
    expected = batch.weighted_grad(y - L.probabilities(s, st.alpha)).to_dict()
    got = st.grad.to_dict()
    assert np.array_equal(got["matrix"], expected["matrix"])
    assert all(got[f] == expected[f] for f in FIELDS)


def test_step_does_not_depend_on_threads():
    a, b, y = many_pairs(2 * 4096 + 5, seed=7)
    out = []
    for t in (1, 3, 8):
        batch = nwgrad.SeqPairBatchDouble(n_threads=t)
        batch.add_many(a, b, dna_params(), gap_model="linear", mode="global")
        batch.score_and_grad()
        st = L.step(batch, y, 0.2)
        g = st.grad.to_dict()
        out.append((st.alpha, g["matrix"].tobytes(), tuple(g[f] for f in FIELDS)))
    assert len(set(out)) == 1


def test_step_needs_one_label_per_pair_and_cached_scores():
    batch = nwgrad.SeqPairBatchDouble(n_threads=2)
    batch.add_many(SEQS_A, SEQS_B, dna_params(), gap_model="affine", mode="local")
    y = np.array([0.0, 1.0] * 3)
    with pytest.raises(RuntimeError, match="score not valid"):
        L.step(batch, y, 0.0)
    batch.score_and_grad()
    with pytest.raises(ValueError, match="one label per pair"):
        L.step(batch, y[:-1], 0.0)


def test_log_likelihood_is_not_provided():
    """Removed: it clipped probabilities. Evaluate it from the logits instead."""
    assert not hasattr(L, "log_likelihood")
    assert not hasattr(L.Step, "loglik_at_alpha0")


def test_fit_alpha_rejects_nan_scores():
    """NaN scores used to "converge" to a plausible intercept (scan B5)."""
    import numpy as np
    import nwgrad.logistic as L
    with pytest.raises(ValueError, match="non-finite"):
        L.fit_alpha(np.array([np.nan, 0.0]), np.array([0.0, 1.0]), 0.0, n_threads=1)


def test_step_with_soft_labels_is_its_parts_composed():
    a, b, _ = many_pairs(2 * 4096 + 5, seed=9)
    y = np.random.default_rng(9).random(len(a))
    batch = nwgrad.SeqPairBatchDouble(n_threads=4)
    batch.add_many(a, b, dna_params(), gap_model="affine", mode="local")
    batch.score_and_grad()
    st = L.step(batch, y, -0.4)
    s = batch.scores()
    assert st.alpha == L.fit_alpha(s, y, -0.4, n_threads=4)
    expected = batch.weighted_grad(y - L.probabilities(s, st.alpha)).to_dict()
    got = st.grad.to_dict()
    assert np.array_equal(got["matrix"], expected["matrix"])
    assert all(got[f] == expected[f] for f in FIELDS)
