"""Exhaustive reference for TRACKED alignment: K parameter sets, chosen per row of A.

Enumerates every alignment path and scores it from the definition, with no dynamic
programming, so it shares nothing with the engine under test.  Exponential: tiny
sequences only.

Model (mirrors aligner.hpp, "Tracks").  DP row i (0..m) is the state after consuming
i residues of A.  With track[i] the set of residue a[i] and gap_track[i] the set of
row i:
  - D consumes a[i], b[j]          -> + matrices[track[i]][a[i], b[j]]
  - X consumes a[i] (gap in B)      -> - gaps[track[i]].extend_b, and
                                        - gaps[track[i]].open_b if it opens its run
  - Y consumes b[j] in row i         -> - gaps[gap_track[i]].extend_a, and
                                        - gaps[gap_track[i]].open_a if it opens its run
  A run opens when the previous step is of another kind (or there is none), so a
  vertical run pays the open of the row it starts in.  Linear: no opens.

Parameters are numpy arrays: mats (K, N, N) and gaps (K, 4) with columns
gap_open_a, gap_extend_a, gap_open_b, gap_extend_b; sequences are lists of alphabet
indices.  Gradients come back in the same shapes, with the gap columns negated
(d score / d penalty = -count), as the library reports them.
"""

import itertools
import math

import numpy as np

D, X, Y = 0, 1, 2
GO_A, GE_A, GO_B, GE_B = 0, 1, 2, 3


def default_gap_track(track):
    m = len(track)
    return [track[0] if m else 0] + list(track)


def all_paths(m, n):
    """Every monotone path from (0,0) to (m,n) as a tuple over {D,X,Y}."""
    if m == 0 and n == 0:
        yield ()
        return
    if m > 0 and n > 0:
        for rest in all_paths(m - 1, n - 1):
            yield (D,) + rest
    if m > 0:
        for rest in all_paths(m - 1, n):
            yield (X,) + rest
    if n > 0:
        for rest in all_paths(m, n - 1):
            yield (Y,) + rest


def path_eval(path, a, b, i0, j0, mats, gaps, track, gtrack, affine):
    """(score, grad_mats, grad_gaps) of one path starting at row i0, column j0."""
    K, N, _ = mats.shape
    gm = np.zeros_like(mats)
    gg = np.zeros_like(gaps)
    score = 0.0
    i, j = i0, j0
    prev = None
    for step in path:
        if step == D:
            s = track[i]
            score += mats[s, a[i], b[j]]
            gm[s, a[i], b[j]] += 1
            i += 1
            j += 1
        elif step == X:
            s = track[i]
            if affine and prev != X:
                score -= gaps[s, GO_B]
                gg[s, GO_B] -= 1
            score -= gaps[s, GE_B]
            gg[s, GE_B] -= 1
            i += 1
        else:
            g = gtrack[i]
            if affine and prev != Y:
                score -= gaps[g, GO_A]
                gg[g, GO_A] -= 1
            score -= gaps[g, GE_A]
            gg[g, GE_A] -= 1
            j += 1
        prev = step
    return score, gm, gg


def scored_paths(a, b, mats, gaps, track, gtrack, affine, local):
    """Every candidate alignment as (score, gm, gg, key).

    Global: every path over the whole of a and b.  Local: every path over every
    pair of non-empty substrings that starts and ends with a match, plus the empty
    alignment (score 0).  With positive gap penalties a local optimum never starts or
    ends in a gap, so that set holds every optimum the DP can return.
    """
    m, n = len(a), len(b)
    if not local:
        for p in all_paths(m, n):
            s, gm, gg = path_eval(p, a, b, 0, 0, mats, gaps, track, gtrack, affine)
            yield s, gm, gg, (0, 0, p)
        return
    yield 0.0, np.zeros_like(mats), np.zeros_like(gaps), None
    for i0, i1 in itertools.combinations(range(m + 1), 2):
        for j0, j1 in itertools.combinations(range(n + 1), 2):
            for p in all_paths(i1 - i0, j1 - j0):
                if p[0] != D or p[-1] != D:
                    continue
                s, gm, gg = path_eval(p, a, b, i0, j0, mats, gaps, track, gtrack, affine)
                yield s, gm, gg, (i0, j0, p)


def best(a, b, mats, gaps, track, gtrack, affine, local):
    """(score, gm, gg, margin): the optimum, its counts, and its lead over the runner-up
    with DIFFERENT counts (0 when the optimum's gradient is ambiguous)."""
    cands = sorted(scored_paths(a, b, mats, gaps, track, gtrack, affine, local),
                   key=lambda c: -c[0])
    top = cands[0]
    margin = math.inf
    for c in cands[1:]:
        if not (np.array_equal(c[1], top[1]) and np.array_equal(c[2], top[2])):
            margin = top[0] - c[0]
            break
    return top[0], top[1], top[2], margin


def soft(a, b, mats, gaps, track, gtrack, affine, local, temperature=1.0):
    """(T log Z(theta/T), expected counts) over the Boltzmann ensemble of GLOBAL paths.

    Local soft sums over the DP's own ensemble (free starts and ends, paths that may
    begin or end in a gap), which this oracle does not enumerate; the tests check
    tracked local soft against the untracked library and finite differences instead.
    """
    if local:
        raise NotImplementedError("local soft: compared against the untracked library instead")
    sc, gms, ggs = [], [], []
    for s, gm, gg, _ in scored_paths(a, b, mats, gaps, track, gtrack, affine, False):
        sc.append(s / temperature)
        gms.append(gm)
        ggs.append(gg)
    hi = max(sc)
    w = np.array([math.exp(s - hi) for s in sc])
    z = w.sum()
    logz = hi + math.log(z)
    p = w / z
    gm = sum(pi * g for pi, g in zip(p, gms))
    gg = sum(pi * g for pi, g in zip(p, ggs))
    return temperature * logz, gm, gg
