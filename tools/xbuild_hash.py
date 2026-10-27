#!/usr/bin/env python3
"""Cross-build / cross-level hasher: every hard path's results, one SHA-256 per configuration.

  python -I tools/xbuild_hash.py OUT.json [--seeds N]   (under each build, or each setting
                                                         such as NWGRAD_DENSE_CODES; diff the JSONs)

For every scalar type, every ISA level this CPU offers (plus scalar_fallback), Local and
Global, untracked and tracked (K=3): affine pointers / scores / hirschberg (small cutoffs,
so the base case runs) / hirschberg_pmax / score-only, linear pointers / scores /
hirschberg -- scores, gradients, guides, aligned strings, coordinates.  One SHA-256 per configuration, and one per (config, level) so a level that
disagrees with scalar_fallback inside one build shows too.
"""
import hashlib, json, sys, argparse, warnings
warnings.simplefilter("ignore")
import numpy as np
import nwgrad

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--seeds", type=int, default=6)
a = ap.parse_args()

CLS = {"double": nwgrad.SeqPairBatchDouble, "float": nwgrad.SeqPairBatch,
       "int16": nwgrad.SeqPairBatchInt16, "int32": nwgrad.SeqPairBatchInt32,
       "int64": nwgrad.SeqPairBatchInt64}
ALPHA = "ACDEFGHIKLMNPQRSTVWY"


def problem(rng, integral, K):
    n = int(rng.integers(5, 40))
    lens = lambda: int(rng.choice([0, 1, 2, rng.integers(3, 60), rng.integers(60, 200), rng.integers(300, 700)],
                                  p=[.03, .03, .04, .5, .3, .1]))
    A, B = [], []
    for _ in range(n):
        x = "".join(rng.choice(list(ALPHA), lens()))
        if rng.random() < 0.4 and len(x) > 4:   # a homologue: mutate, insert, delete
            y = list(x)
            for _ in range(len(y) // 6):
                k = int(rng.integers(len(y))); y[k] = str(rng.choice(list(ALPHA)))
            k = int(rng.integers(len(y))); y[k:k] = list(rng.choice(list(ALPHA), int(rng.integers(1, 9))))
            k = int(rng.integers(len(y))); del y[k:k + int(rng.integers(1, 9))]
            y = "".join(y)
        else:
            y = "".join(rng.choice(list(ALPHA), lens()))
        A.append(x); B.append(y)
    N = len(ALPHA)
    if integral:
        mats = rng.integers(-3, 5, size=(K, N, N)).astype(float)
        gaps = rng.integers(0, 4, size=(K, 4)).astype(float); gaps[:, 1] = np.maximum(gaps[:, 1], 1)
    else:
        mats = rng.normal(0, 1.5, size=(K, N, N)).round(int(rng.integers(0, 3)))
        gaps = rng.uniform(0.1, 3.0, size=(K, 4)).round(1)
    P = nwgrad.AlignParams.from_arrays(np.ascontiguousarray(mats), np.ascontiguousarray(gaps), ALPHA)
    tracks = [list(rng.integers(0, K, size=len(x))) for x in A] if K > 1 else None
    gtracks = [list(rng.integers(0, K, size=len(x) + 1)) for x in A] if K > 1 else None
    return A, B, P, tracks, gtracks


def run(prec, A, B, P, tracks, gtracks, mode, tb, cut, gm="affine", gd="hard"):
    kw = dict(n_threads=2, traceback=tb, gap_model=gm, mode=mode, grad_mode=gd)
    if tracks is not None: kw.update(tracks=tracks, gap_tracks=gtracks)
    b = CLS[prec](A, B, P, **kw)
    b.fill = "striped"
    if cut: b.hb_cutoff = cut
    h = hashlib.sha256()
    if gd == "none":   # score-only: the score kernels, no path
        s = b.score_and_grad()
        h.update(repr(float(s)).encode()); h.update(np.asarray(b.scores(), float).tobytes())
        return h.hexdigest()
    s = b.score_and_grad(keep_paths=True)
    h.update(repr(float(s)).encode())
    h.update(np.asarray(b.scores(), float).tobytes())
    for i in range(len(b)):
        p = b[i]
        g = p.grad.to_dict()
        for k in sorted(g):
            h.update(k.encode()); h.update(g[k].encode() if isinstance(g[k], str) else np.asarray(g[k], float).tobytes())
        h.update(repr(p.aligned()).encode())
        h.update(np.asarray(p.coordinates()).tobytes())
        if p.path_valid: h.update(np.asarray(p.guide_j).tobytes())
    return h.hexdigest()


levels = ["scalar_fallback"] + [l for l in nwgrad.available_isa_levels() if l != "scalar_fallback"]
out = {}
for seed in range(a.seeds):
    for prec in CLS:
        integral = prec.startswith("int") or seed % 2 == 0
        for K in (1, 3):
            rng = np.random.default_rng(1000 * seed + K)
            A, B, P, tr, gtr = problem(rng, integral, K)
            for mode in ("local", "global"):
                for tb, cut, gm, gd in (("pointers", 0, "affine", "hard"), ("hirschberg", 8, "affine", "hard"),
                                        ("hirschberg", 37, "affine", "hard"), ("hirschberg_pmax", 16, "affine", "hard"),
                                        ("scores", 0, "affine", "hard"), ("pointers", 0, "affine", "none"),
                                        ("pointers", 0, "linear", "hard"), ("scores", 0, "linear", "hard"),
                                        ("hirschberg", 8, "linear", "hard")):
                    if tb == "hirschberg_pmax" and not prec.startswith("int") and prec != "float":
                        pass
                    key = f"{seed}/{prec}/K{K}/{mode}/{tb}/{cut}/{gm}/{gd}"
                    per = {}
                    for lvl in levels:
                        nwgrad.set_isa_level(lvl)
                        try:
                            per[lvl] = run(prec, A, B, P, tr, gtr, mode, tb, cut, gm, gd)
                        except Exception as e:
                            per[lvl] = "ERR " + type(e).__name__ + ": " + str(e)[:80]
                    out[key] = per
    print(f"seed {seed} done", file=sys.stderr, flush=True)
json.dump({"levels": levels, "h": out}, open(a.out, "w"), indent=0)
# within-build: every level must equal scalar_fallback (hirschberg_pmax at double/float is
# not bit-exact across... it IS: scalar==simd is its contract)
bad = [(k, l) for k, per in out.items() for l, v in per.items() if v != per["scalar_fallback"]]
print(f"{len(out)} configs x {len(levels)} levels; {len(bad)} level/scalar mismatches", file=sys.stderr)
for k, l in bad[:20]: print("  ", k, l, out[k][l][:60], file=sys.stderr)
