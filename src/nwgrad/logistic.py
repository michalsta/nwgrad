"""
A binary logistic link over per-pair alignment scores: P(y = 1) = expit(alpha + score).

This is the likelihood DiscrimAlign maximises. ``step(batch, labels, alpha0)`` does
one optimisation iteration's work after ``batch.score_and_grad()``: the fitted
intercept and the gradient ``sum_i (y_i - p_i) grad_i``. ``fit_alpha`` and
``probabilities`` work on plain arrays of scores. The log-likelihood itself is not
provided: evaluate it from the logits z = alpha + score, as
``sum(y*z - logaddexp(0, z))``, which needs no clipping. Every sum is taken over fixed
blocks, so results do not depend on the thread count.

Labels are not checked: each must be in [0, 1], a 0/1 class label or a soft label (a
target probability, giving the Bernoulli cross-entropy), and they must not be all 0 or
all 1. Otherwise the likelihood has no finite maximum, and ``fit_alpha`` and ``step``
return a meaningless intercept instead of raising.
"""
from .nwgrad_ext import logistic as _ext

Step = _ext.Step
step = _ext.step
fit_alpha = _ext.fit_alpha
probabilities = _ext.probabilities

__all__ = ["Step", "step", "fit_alpha", "probabilities"]
