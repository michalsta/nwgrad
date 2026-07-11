"""Exhaustive reference implementation of the alignment models.

Enumerates *every* alignment path and scores it directly from the definition,
with no dynamic programming.  Exponential, so only usable on very short
sequences, but it is a fully independent oracle: it shares no code with the
DP engine under test.

This exists mainly to pin down asymmetric gap penalties.  Biopython — the
oracle used by the rest of the suite — can only express a single gap cost for
both sequences, so it cannot check that the "a" penalties really drive gaps in
A and the "b" penalties really drive gaps in B.

Model (mirrors aligner.hpp):
  - step D consumes one char of each  → matrix.score(a_i, b_j)
  - step X consumes one char of A only → gap in B, charged with the b params
  - step Y consumes one char of B only → gap in A, charged with the a params
  - affine: each maximal run of k consecutive gap steps costs open + k * extend
    (the open is charged on every run, including leading and trailing ones, and
    an X run immediately followed by a Y run pays both opens)
  - linear: a run of k gap steps costs k * extend
"""

import itertools
import math

D, X, Y = "D", "X", "Y"


def all_paths(m, n):
    """Every monotone path from (0,0) to (m,n) as a string over {D,X,Y}."""
    if m == 0 and n == 0:
        yield ""
        return
    if m > 0 and n > 0:
        for rest in all_paths(m - 1, n - 1):
            yield D + rest
    if m > 0:
        for rest in all_paths(m - 1, n):
            yield X + rest
    if n > 0:
        for rest in all_paths(m, n - 1):
            yield Y + rest


def path_score(path, a, b, matrix, gap_open_a, gap_extend_a,
               gap_open_b, gap_extend_b, affine):
    """Score one path under the model above."""
    score = 0.0
    i = j = 0
    for step, run in itertools.groupby(path):
        k = len(list(run))
        if step == D:
            for _ in range(k):
                score += matrix.score(a[i], b[j])
                i += 1
                j += 1
        elif step == X:  # gap in B, consumes A, uses the b params
            score -= k * gap_extend_b + (gap_open_b if affine else 0.0)
            i += k
        else:            # gap in A, consumes B, uses the a params
            score -= k * gap_extend_a + (gap_open_a if affine else 0.0)
            j += k
    return score


def _scored_paths(a, b, params, affine):
    """(path, score) for every global alignment of a vs b."""
    d = params.to_dict()
    for path in all_paths(len(a), len(b)):
        yield path, path_score(path, a, b, params.matrix,
                               d["gap_open_a"], d["gap_extend_a"],
                               d["gap_open_b"], d["gap_extend_b"], affine)


def global_score(a, b, params, affine):
    """Needleman-Wunsch score by exhaustive enumeration."""
    return max(s for _, s in _scored_paths(a, b, params, affine))


def local_score(a, b, params, affine):
    """Smith-Waterman score: the best global score over every pair of substrings.

    Valid because all gap penalties are non-negative, so a local alignment never
    gains by starting or ending on a gap; the empty alignment scores 0.
    """
    best = 0.0
    for i0, i1 in itertools.combinations(range(len(a) + 1), 2):
        for j0, j1 in itertools.combinations(range(len(b) + 1), 2):
            best = max(best, global_score(a[i0:i1], b[j0:j1], params, affine))
    return best


def global_log_z(a, b, params, affine):
    """log-partition function: log sum(exp(score)) over every global alignment."""
    scores = [s for _, s in _scored_paths(a, b, params, affine)]
    hi = max(scores)
    return hi + math.log(sum(math.exp(s - hi) for s in scores))


def _counts(path, a, b, affine):
    """Feature counts of one path: matrix pair counts and gap counts.

    Returns (pairs, opens_a, extends_a, opens_b, extends_b) where `pairs` maps
    (char_a, char_b) -> multiplicity.  All five are plain non-negative counts.

    Note these are *counts*, not derivatives.  Because every parameter enters the
    score linearly, d(score)/d(param) is the parameter's multiplier: for a matrix
    entry that is +count, but for a gap penalty it is -count, since the score
    subtracts the penalties.  Callers comparing against the library's gradient
    must negate the four gap counts.
    """
    pairs = {}
    opens_a = extends_a = opens_b = extends_b = 0
    i = j = 0
    for step, run in itertools.groupby(path):
        k = len(list(run))
        if step == D:
            for _ in range(k):
                pairs[(a[i], b[j])] = pairs.get((a[i], b[j]), 0) + 1
                i += 1
                j += 1
        elif step == X:
            opens_b += 1 if affine else 0
            extends_b += k
            i += k
        else:
            opens_a += 1 if affine else 0
            extends_a += k
            j += k
    return pairs, opens_a, extends_a, opens_b, extends_b


def global_hard_counts(a, b, params, affine):
    """Feature counts along the single best global alignment.

    Only meaningful when the optimum is unique — ties make the subgradient
    implementation-defined, so callers must pick unambiguous inputs.
    """
    best_path, _ = max(_scored_paths(a, b, params, affine), key=lambda ps: ps[1])
    return _counts(best_path, a, b, affine)


def global_soft_counts(a, b, params, affine):
    """Expected feature counts over the Boltzmann ensemble of all alignments.

    This is the true gradient of the log-partition function, so it is what
    soft_grad must reproduce.
    """
    scored = list(_scored_paths(a, b, params, affine))
    hi = max(s for _, s in scored)
    weights = [math.exp(s - hi) for _, s in scored]
    z = sum(weights)

    pairs = {}
    opens_a = extends_a = opens_b = extends_b = 0.0
    for (path, _), w in zip(scored, weights):
        p, oa, ea, ob, eb = _counts(path, a, b, affine)
        for key, v in p.items():
            pairs[key] = pairs.get(key, 0.0) + v * w / z
        opens_a += oa * w / z
        extends_a += ea * w / z
        opens_b += ob * w / z
        extends_b += eb * w / z
    return pairs, opens_a, extends_a, opens_b, extends_b
