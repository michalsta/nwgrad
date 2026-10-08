#!/usr/bin/env python3
"""A/B differential fuzz: the same random batches through two nwgrad builds.

    python tools/fuzz_ab.py --a /path/to/old/python --b /path/to/new/python 0 200

Each seed builds one homogeneous batch (one gap model / mode / grad mode — what 0.6
requires; segments with different params), runs score_and_grad(), set_params() +
banded_grad(), the stored-path alignment (keep_paths / the old align_full) and
weighted_grad(), and dumps every per-pair result.  Run under each interpreter (with
whichever API that build has) and compared: scores, per-pair gradients, guides,
alignments and weighted_grad bit for bit — both builds run the same kernels, so even
soft per-pair results must agree exactly.  Only the summed compute_grad() of soft pairs
may differ in the last bits (pre-0.6 merged per-thread sums in completion order).

    python tools/fuzz_ab.py --dump SEED OUT.pkl    # internal: one build, one seed
"""
import argparse, pickle, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np


def rparams(nwgrad, rng, alpha, gm):
    n = len(alpha)
    kind = rng.integers(3)
    if kind == 0:
        m = np.full((n, n), -1.0); np.fill_diagonal(m, 2.0)
    elif kind == 1:
        m = rng.integers(-4, 6, (n, n)).astype(float)
    else:
        m = rng.normal(size=(n, n))
    go = 0.0 if gm == "linear" else float(rng.choice([0.5, 2.5, 11.0, 1e-4]))
    ge = float(rng.choice([0.1, 0.7, 1.0]))
    return nwgrad.AlignParams(nwgrad.SubstMatrix(m, alphabet=alpha), go, ge, go * 1.3, ge * 0.8)


def rlen(rng):
    r = rng.random()
    if r < 0.05: return 0
    if r < 0.10: return 1
    if r < 0.85: return int(rng.integers(2, 60))
    if r < 0.97: return int(rng.integers(60, 300))
    return int(rng.integers(500, 700))


def dump(seed, out):
    import warnings
    warnings.simplefilter("ignore")
    import nwgrad
    rng = np.random.default_rng(seed)
    alpha = str(rng.choice(["ACGT", nwgrad.PROTEIN.symbols, "ACGTN"]))
    prec = str(rng.choice(["double", "float32"]))
    tb = str(rng.choice(["auto", "pointers", "scores", "hirschberg"]))
    gm = str(rng.choice(["affine", "linear"])); mode = str(rng.choice(["global", "local"]))
    gd = str(rng.choice(["hard", "hard", "soft", "none"]))
    fill = str(rng.choice(["interpair", "striped", "rowwise"]))
    soft_guide = str(rng.choice(["eager", "lazy", "posterior"]))
    threads = int(rng.integers(1, 5))
    hb = int(rng.choice([512, 8, 24]))
    bw = int(rng.integers(1, 10))
    segs = []
    for _ in range(int(rng.integers(1, 4))):
        k = int(rng.integers(1, 30)); lb = rlen(rng)
        A = ["".join(rng.choice(list(alpha), rlen(rng))) for _ in range(k)]
        B = ["".join(rng.choice(list(alpha), lb if rng.random() < 0.7 else rlen(rng)))
             for _ in range(k)]
        segs.append((A, B, rparams(nwgrad, rng, alpha, gm)))
    p2 = rparams(nwgrad, rng, alpha, gm)
    w = rng.normal(size=sum(len(s[0]) for s in segs))

    cls = nwgrad.SeqPairBatchDouble if prec == "double" else nwgrad.SeqPairBatch
    new_api = not hasattr(nwgrad, "BatchAligner")

    def make():
        if new_api:
            b = cls(threads, tb, gap_model=gm, mode=mode, grad_mode=gd)
        else:
            b = cls(threads, tb)
        b.fill = fill
        b.soft_guide = soft_guide
        b.hb_cutoff = hb
        for A, B, p in segs:
            if new_api:
                b.add_many(A, B, p)
            else:
                b.add_many(A, B, p, gap_model=gm, mode=mode, grad_mode=gd)
        return b

    def snap(b, paths=False):
        r = {"scores": b.scores()}
        if gd != "none" and not paths:   # the old align_full held its gradient back
            r["grads"] = b.grads()
        r["guides"] = [list(b[i].guide_j) if b[i].path_valid else None for i in range(len(b))]
        if paths:
            r["aligned"] = [b[i].aligned() for i in range(len(b))]
            r["coords"] = [b[i].coordinates().tolist() for i in range(len(b))]
        return r

    res = {"cfg": (alpha[:4], prec, tb, gm, mode, gd, fill, soft_guide, threads, hb, bw)}
    try:
        b = make()
        res["sum"] = b.score_and_grad()
        res["full"] = snap(b)
        if gd != "none":
            res["wg"] = b.weighted_grad(w).to_dict()
            res["cg"] = b.compute_grad().to_dict()
        b.set_params(p2)
        res["bsum"] = b.banded_grad(bw)
        res["banded"] = snap(b)
        b2 = make()
        if new_api:
            res["psum"] = b2.score_and_grad(keep_paths=True)
        else:
            b2.alloc_dp(); res["psum"] = b2.align_full()
        res["paths"] = snap(b2, paths=True)
    except Exception as e:   # must be the same exception under both builds
        res["error"] = (type(e).__name__, str(e)[:200])
    pickle.dump(res, open(out, "wb"))


def same(x, y, tol=False):
    if isinstance(x, dict):
        return x.keys() == y.keys() and all(same(x[k], y[k], tol) for k in x)
    if isinstance(x, (list, tuple)):
        return len(x) == len(y) and all(same(u, v, tol) for u, v in zip(x, y))
    if isinstance(x, np.ndarray) or isinstance(x, float):
        a, b = np.asarray(x, float), np.asarray(y, float)
        if a.shape != b.shape: return False
        if tol: return bool(np.allclose(a, b, rtol=1e-11, atol=1e-11, equal_nan=True))
        return bool(np.array_equal(a, b, equal_nan=True))
    return x == y


def compare(ra, rb):
    if "error" in ra or "error" in rb:
        if ra.get("error", (None,))[0] != rb.get("error", (None,))[0]:
            return f"error mismatch: {ra.get('error')} vs {rb.get('error')}"
        return None
    soft = ra["cfg"][5] == "soft"
    for k in ("sum", "full", "wg", "bsum", "banded"):
        if k in ra and not same(ra[k], rb.get(k)):
            return f"{k} differs"
    # The stored-path stage: the old align_full() ran every pair's own forward-backward,
    # keep_paths honours fill="interpair" (a shared soft pass): soft SCORES there agree
    # within tolerance only — the soft path's documented contract across fills.  The
    # paths themselves (Viterbi) and the guides must be exact.
    if not same(ra["psum"], rb["psum"], tol=soft):
        return "psum differs"
    pa, pb = ra["paths"], rb["paths"]
    if not same(pa["scores"], pb["scores"], tol=soft):
        return "path-stage scores differ"
    for k in ("aligned", "coords", "guides"):
        if not same(pa[k], pb[k]):
            return f"path-stage {k} differ"
    if "cg" in ra and not same(ra["cg"], rb["cg"], tol=soft):
        return "compute_grad differs"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lo", type=int, nargs="?"); ap.add_argument("hi", type=int, nargs="?")
    ap.add_argument("--a"); ap.add_argument("--b")
    ap.add_argument("--a-env", default=""); ap.add_argument("--b-env", default="")
    ap.add_argument("--dump", nargs=2)
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    if a.dump:
        dump(int(a.dump[0]), a.dump[1]); return
    import os, tempfile
    tmp = tempfile.mkdtemp()

    def run(py, env, seed, tag):
        out = f"{tmp}/{tag}{seed}.pkl"
        e = dict(os.environ)
        if env: e["PYTHONPATH"] = env
        p = subprocess.run([py, __file__, "--dump", str(seed), out], env=e,
                           capture_output=True, text=True)
        if p.returncode != 0:
            return {"error": ("CRASH", f"rc={p.returncode} {p.stderr[-300:]}")}
        return pickle.load(open(out, "rb"))

    def one(seed):
        ra, rb = run(a.a, a.a_env, seed, "a"), run(a.b, a.b_env, seed, "b")
        r = compare(ra, rb)
        return f"seed {seed} {ra.get('cfg')}: {r}" if r else None

    with ThreadPoolExecutor(a.jobs) as ex:
        bad = [r for r in ex.map(one, range(a.lo, a.hi)) if r]
    for r in bad: print(r)
    print(f"{a.hi - a.lo} seeds, {len(bad)} differ")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
