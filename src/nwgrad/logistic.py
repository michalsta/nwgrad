"""
A binary logistic link over per-pair alignment scores: P(y = 1) = expit(alpha + score).

This is the likelihood DiscrimAlign maximises. ``step(batch, labels, alpha0)`` does
one optimisation iteration's work after ``batch.score_and_grad()``: the
log-likelihood at ``alpha0``, the fitted intercept, and the gradient
``sum_i (y_i - p_i) grad_i``. ``fit_alpha``, ``log_likelihood`` and ``probabilities``
work on plain arrays of scores. Every sum is taken over fixed blocks, so results do
not depend on the thread count.
"""
from .nwgrad_ext import logistic as _ext

Step = _ext.Step
step = _ext.step
fit_alpha = _ext.fit_alpha
log_likelihood = _ext.log_likelihood
probabilities = _ext.probabilities

__all__ = ["Step", "step", "fit_alpha", "log_likelihood", "probabilities"]
