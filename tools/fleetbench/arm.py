#!/usr/bin/env python3
"""Time ONE arm on ONE cell; print one JSON line, write per-pair scores to --scores-out.

    arm.py --tool nwgrad --variant default --fixture F.tsv --kind protein --gap affine
           --mode global --out score --dtype float32 --threads 1 --reps 3 --deadline 12.5
           --scores-out /tmp/x.txt

Protocol, shared with the C/C++ drivers:
  - setup (reading, encoding, building objects) is untimed;
  - one warm-up pass, then --reps timed passes; the best pass is reported;
  - --deadline bounds one pass: per-pair tools stop at the first pair past it and report
    "timeout"; one-call batch tools cannot stop, the parent's wall-clock kill covers them;
  - status: ok | timeout | unsupported (no such operation in this tool) | error.
out: score (score only) | path (score + alignment) | hard (hard gradient) | soft.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from problems import problem, read_pairs, nwgrad_params  # noqa: E402


class Unsupported(Exception):
    pass


# ── nwgrad, Python API ───────────────────────────────────────────────────────────

NW_CLASS = {"float32": "SeqPairBatch", "float64": "SeqPairBatchDouble",
            "int16": "SeqPairBatchInt16", "int32": "SeqPairBatchInt32",
            "int64": "SeqPairBatchInt64"}
FN_SUFFIX = {"float32": "", "float64": "_double", "int16": "_int16", "int32": "_int32",
             "int64": "_int64"}


def parse_variant(v):
    """'default' or 'key=value[,key=value]'."""
    if v in ("default", ""): return {}
    return dict(kv.split("=", 1) for kv in v.split(","))


def nwgrad_batch(a, A, B, prob):
    import nwgrad
    P = nwgrad_params(nwgrad, prob)
    var = parse_variant(a.variant)
    if a.out == "soft" and a.dtype != "float64":
        raise Unsupported("nwgrad's soft path is float64")
    if a.out == "soft" and var.get("traceback"):
        raise Unsupported("soft has no traceback mode")
    info = {"isa": nwgrad.simd_isa(), "compiler": nwgrad.compiled_with()}
    if var.get("api") == "functions":
        return nwgrad_functions(a, A, B, P, info)
    gm = {"score": "none", "path": "none", "hard": "hard", "soft": "soft"}[a.out]
    kw = dict(n_threads=a.threads, gap_model=a.gap, mode=a.mode, grad_mode=gm)
    if "traceback" in var: kw["traceback"] = var["traceback"]
    if "kernel" in var: kw["kernel"] = var["kernel"]
    if "kernel" in var and var["kernel"] not in ("scalar_fallback",) + tuple(nwgrad.available_isa_levels()):
        raise Unsupported(f"kernel {var['kernel']} not on this CPU")
    b = getattr(nwgrad, NW_CLASS[a.dtype])(A, B, P, **kw)
    if "fill" in var: b.fill = var["fill"]
    if "schedule" in var: b.schedule = var["schedule"]
    if "soft_impl" in var: b.soft_impl = var["soft_impl"]
    keep = a.out == "path"
    grad = a.out in ("hard", "soft")
    info["n_threads"] = b.n_threads
    try:
        info["traceback_resolved"] = sorted({b[i].traceback for i in range(min(len(b), 50))})
    except Exception:
        pass

    def prep():
        b.set_params(P)                       # clears cached scores: every pass recomputes
    def run():
        b.score_and_grad(keep_paths=keep)
        if grad: b.compute_grad()             # the gradient as a training step consumes it
    def scores():
        return list(b.scores())
    return dict(kind="batch", prep=prep, run=run, scores=scores, info=info)


def nwgrad_functions(a, A, B, P, info):
    import nwgrad
    if a.threads != 1: raise Unsupported("single-pair functions run on the calling thread")
    pre = "nw" if a.mode == "global" else "sw"
    if a.out == "path": raise Unsupported("the functions return no path")
    if a.out == "soft":
        name = f"{pre}_affine_soft_grad" if a.gap == "affine" else f"{pre}_soft_grad"
        name += "_double"
    elif a.out == "score":
        name = f"{pre}_score_affine" if a.gap == "affine" else f"{pre}_score"
        name += FN_SUFFIX[a.dtype]
    else:
        name = f"{pre}_affine_grad" if a.gap == "affine" else f"{pre}_grad"
        name += FN_SUFFIX[a.dtype]
    f = getattr(nwgrad, name)
    if a.out == "score":
        return dict(kind="pair", f=lambda x, y: f(x, y, P), info=info)
    return dict(kind="pair", f=lambda x, y: f(x, y, P)[0], info=info)


# ── Biopython ────────────────────────────────────────────────────────────────────

def biopython(a, A, B, prob):
    if a.dtype != "float64": raise Unsupported("Biopython scores in float64")
    if a.out not in ("score", "path"): raise Unsupported(a.out)
    from Bio import Align
    from Bio.Align import substitution_matrices
    import numpy as np
    var = parse_variant(a.variant)
    al = Align.PairwiseAligner()
    if prob["named"] == "blosum62":
        al.substitution_matrix = substitution_matrices.load("BLOSUM62")
    else:
        m = substitution_matrices.Array(prob["alphabet"], 2,
                                        np.array(prob["matrix"], dtype=float))
        al.substitution_matrix = m
    if a.gap == "affine":
        al.open_gap_score, al.extend_gap_score = -(prob["go"] + prob["ge"]), -prob["ge"]
    else:
        al.open_gap_score = al.extend_gap_score = -prob["ge"]
    al.mode = var.get("mode", a.mode)
    if al.mode == "fogsaa" and a.mode != "global": raise Unsupported("FOGSAA is global only")
    info = {"algorithm": al.algorithm, "version": __import__("Bio").__version__}
    if a.out == "score":
        return dict(kind="pair", f=lambda x, y: al.score(x, y), info=info)
    def f(x, y):
        aln = al.align(x, y)[0]
        _ = aln.coordinates
        return aln.score
    return dict(kind="pair", f=f, info=info)


# ── scikit-bio ───────────────────────────────────────────────────────────────────

def skbio_arm(a, A, B, prob):
    if a.dtype != "float64": raise Unsupported("scikit-bio pair_align scores in float64")
    if a.out != "path": raise Unsupported("pair_align always traces back")
    import numpy as np
    import skbio
    from skbio.alignment import pair_align
    if prob["named"] == "blosum62":
        sub = "BLOSUM62"
    else:
        from skbio.sequence import SubstitutionMatrix
        sub = SubstitutionMatrix(prob["alphabet"], np.array(prob["matrix"], dtype=float))
    gc = (prob["go"], prob["ge"]) if a.gap == "affine" else prob["ge"]
    kw = dict(free_ends=False) if a.mode == "global" else {}
    def f(x, y):
        r = pair_align(x, y, mode=a.mode, sub_score=sub, gap_cost=gc, **kw)
        _ = r.paths[0]
        return r.score
    return dict(kind="pair", f=f, info={"version": skbio.__version__})


# ── pyopal (Opal) ────────────────────────────────────────────────────────────────

def pyopal_arm(a, A, B, prob):
    if a.dtype not in ("int16", "int32"):
        raise Unsupported("Opal: 8/16/32-bit saturating, filed under int16 or int32")
    if a.out not in ("score", "path"): raise Unsupported(a.out)
    import pyopal
    if prob["named"] == "blosum62":
        sm = "BLOSUM62"
    else:
        import scoring_matrices
        sm = scoring_matrices.ScoringMatrix(prob["matrix"], alphabet=prob["alphabet"])
    # Opal: a gap of length k costs open + (k-1)*extend
    go, ge = (prob["go"] + prob["ge"], prob["ge"]) if a.gap == "affine" else (prob["ge"], prob["ge"])
    alg = "nw" if a.mode == "global" else "sw"
    mode = "score" if a.out == "score" else "full"
    al = pyopal.Aligner(sm, gap_open=go, gap_extend=ge)
    alpha = al.alphabet
    def f(x, y):
        r = al.align(x, pyopal.Database([y], alphabet=alpha), mode=mode, algorithm=alg)[0]
        if mode == "full": _ = r.alignment
        return r.score
    # Opal reruns at 8 -> 16 -> 32 bits on overflow and does not say which width a pair
    # took; arm.py files the cell by the score range after the run (sat_filter).
    return dict(kind="pair", f=f, info={"version": pyopal.__version__, "sat": True},
                sat_filter=True)


# ── edlib ────────────────────────────────────────────────────────────────────────

def edlib_arm(a, A, B, prob):
    if not (prob["alphabet"] == "ACGT" and a.gap == "linear" and a.mode == "global"):
        raise Unsupported("edlib: unit-cost edit distance only (DNA, linear, global)")
    if a.dtype != "int32": raise Unsupported("edlib's distance is an int")
    if a.out not in ("score", "path"): raise Unsupported(a.out)
    import edlib
    task = "distance" if a.out == "score" else "path"
    def f(x, y):
        r = edlib.align(x, y, mode="NW", task=task)
        return -r["editDistance"]
    return dict(kind="pair", f=f, info={})


# ── DeepBLAST ────────────────────────────────────────────────────────────────────

def deepblast_arm(a, A, B, prob):
    if a.out != "soft" or a.mode != "global" or a.gap != "linear":
        raise Unsupported("DeepBLAST: global, single-state (linear) soft alignment only")
    if a.dtype != "float64": raise Unsupported("nwgrad's soft path is float64")
    import torch
    torch.set_num_threads(1)
    from deepblast.nw import NeedlemanWunschDecoder
    alpha = prob["alphabet"]
    M = torch.tensor(prob["matrix"], dtype=torch.float64, requires_grad=True)
    dec = NeedlemanWunschDecoder("softmax")
    idx = {c: i for i, c in enumerate(alpha)}
    enc = [(torch.tensor([idx[c] for c in x]), torch.tensor([idx[c] for c in y]))
           for x, y in zip(A, B)]
    cache = {}
    def f(x, y):
        ia, ib = cache.get((x, y)) or (torch.tensor([idx[c] for c in x]),
                                       torch.tensor([idx[c] for c in y]))
        theta = M[ia][:, ib].unsqueeze(0)
        gap = torch.full_like(theta, -float(prob["ge"]))
        v = dec(theta, gap)
        v.sum().backward()
        M.grad = None
        return float("nan")                  # a different model: no score to check
    del enc
    return dict(kind="pair", f=f, info={"model": "deepblast single-state",
                                       "unchecked": True})


TOOLS = {"nwgrad": nwgrad_batch, "biopython": biopython, "scikit-bio": skbio_arm,
         "pyopal": pyopal_arm, "edlib": edlib_arm, "deepblast": deepblast_arm}


# ── timing ───────────────────────────────────────────────────────────────────────

_POOL_F = None


def _pool_work(chunk):
    return [_POOL_F(x, y) for x, y in chunk]


def time_arm(a, A, B, arm):
    deadline = a.deadline if a.deadline > 0 else float("inf")
    # the warm-up pass also pays one-time costs (imports, JIT, plan building, first-touch
    # page faults), which do not scale with the work: it gets an absolute allowance on top
    warm_deadline = deadline + a.warmup_slack
    times = []
    status = "ok"
    sc = None
    if arm["kind"] == "batch":
        for rep in range(a.reps + 1):
            arm["prep"]()
            t0 = time.perf_counter(); arm["run"](); dt = time.perf_counter() - t0
            if rep: times.append(dt)
            if dt > (deadline if rep else warm_deadline): status = "timeout"; break
        sc = arm["scores"]()
    elif a.threads == 1:
        f = arm["f"]
        for rep in range(a.reps + 1):
            out = []
            t0 = time.perf_counter()
            lim = deadline if rep else warm_deadline
            for x, y in zip(A, B):
                out.append(f(x, y))
                if time.perf_counter() - t0 > lim:
                    status = "timeout"; break
            dt = time.perf_counter() - t0
            if status != "ok": times.append(dt); break
            if rep: times.append(dt)
            sc = out
    else:
        # harness-parallel: the tool has no threads of its own; split the pairs over
        # forked worker processes (pool started before timing)
        import multiprocessing as mp
        global _POOL_F
        _POOL_F = arm["f"]
        n = len(A); k = a.threads * 8
        chunks = [list(zip(A[i * n // k:(i + 1) * n // k], B[i * n // k:(i + 1) * n // k]))
                  for i in range(k)]
        with mp.get_context("fork").Pool(a.threads) as pool:
            pool.map(_pool_work, [[c[0]] for c in chunks if c])          # spin-up
            for rep in range(a.reps + 1):
                t0 = time.perf_counter()
                r = pool.map_async(_pool_work, chunks, chunksize=1)
                lim = deadline if rep else warm_deadline
                try:
                    res = r.get(timeout=lim if lim != float("inf") else None)
                except mp.TimeoutError:
                    status = "timeout"; times.append(time.perf_counter() - t0); break
                dt = time.perf_counter() - t0
                if rep: times.append(dt)
                sc = [s for part in res for s in part]
            if status != "ok": pool.terminate()
        arm["info"]["harness_parallel"] = True
    return status, times, sc


def main():
    ap = argparse.ArgumentParser()
    for k in ("tool", "variant", "fixture", "kind", "gap", "mode", "out", "dtype",
              "scores_out"):
        ap.add_argument("--" + k.replace("_", "-"), dest=k, default="default" if k == "variant" else None)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--deadline", type=float, default=0)
    ap.add_argument("--warmup-slack", dest="warmup_slack", type=float, default=3.0)
    a = ap.parse_args()
    res = {"status": "error"}
    try:
        A, B = read_pairs(a.fixture)
        prob = problem(a.kind, a.gap, a.mode)
        arm = TOOLS[a.tool](a, A, B, prob)
        status, times, sc = time_arm(a, A, B, arm)
        res = {"status": status, "times": times, "best_s": min(times) if times else None,
               "n_pairs": len(A), "cells": sum(len(x) * len(y) for x, y in zip(A, B)),
               "info": arm["info"]}
        if arm.get("sat_filter") and status == "ok" and sc:
            # a saturating cascade is filed under the narrowest width that held the cell
            # (every DP value is bounded by the score range here, so the final scores
            # decide it): int16 when every |score| fits 16 bits, int32 otherwise
            fits16 = max(abs(s) for s in sc) < 32767
            if (a.dtype == "int16") != fits16:
                raise Unsupported("saturating cascade filed under " + ("int16" if fits16 else "int32"))
        if sc is not None and a.scores_out and status == "ok":
            with open(a.scores_out, "w") as f:
                f.write("\n".join(repr(float(s)) for s in sc) + "\n")
    except Unsupported as e:
        res = {"status": "unsupported", "why": str(e)}
    except Exception as e:
        import traceback
        res = {"status": "error", "why": f"{type(e).__name__}: {e}",
               "trace": traceback.format_exc()[-1500:]}
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
