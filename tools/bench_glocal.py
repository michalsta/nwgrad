#!/usr/bin/env python3
"""Glocal on PosimiR's model: the native mode vs PosimiR's emulation, one training step.

    python tools/bench_glocal.py --data pairs.tsv [--n 200000] [--threads 1 6]
        [--rounds 5] [--precision float double] [--models linear affine]

PosimiR's shape: miRNA x target-site pairs (Manakov TSV: column 0 the miRNA, column 1
the site), one parameter set per miRNA position (L = the longest miRNA), plus two edge
sets: FREE (no charge for gaps in the miRNA) and DUMMY (charge 1).  Track 0..n-1, gap
track [edge, 0, ..., n-2, edge].  Arms, each in a fresh process, alternating, best of
--rounds; one step = set_params + score_and_grad + scores + weighted_grad:
  free        mode="global", edges FREE: glocal emulated by free rows 0 and n
              (what PosimiR's nwgrad backend runs), fill="interpair"
  striped     mode="glocal", edges DUMMY, fill="striped" (each pair its own fill)
  interpair   mode="glocal", edges DUMMY, fill="interpair" (the shared fill)
  control     interpair again (the noise floor)
Every arm's scores are checked equal to the free arm's (the optimum is the same; the
gap moves along rows 0 and n differ only on ties).
"""
import argparse, json, subprocess, sys

CHILD = r'''
import json, sys, time, warnings
warnings.simplefilter("ignore")
import numpy as np, nwgrad
arm, n, threads, data, prec, gm = sys.argv[1:7]
n, threads = int(n), int(threads)
rows = [l.rstrip("\n").split("\t") for l in open(data)][1:]
rng = np.random.default_rng(0)
idx = rng.choice(len(rows), size=min(n, len(rows)), replace=False)
A = [rows[i][0] for i in idx]; B = [rows[i][1] for i in idx]
L = max(map(len, A)); FREE, DUMMY = L, L + 1; K = L + 2
lens = np.array([len(a) for a in A])
tracks = np.concatenate([np.arange(k) for k in lens]).astype(np.uint16)
edge = FREE if arm == "free" else DUMMY
gap_tracks = np.concatenate([np.r_[edge, np.arange(k - 1), edge] for k in lens]).astype(np.uint16)
M0 = np.array([[0.53, -0.61, -0.60, -0.56], [-0.44, 0.58, -0.57, -0.48],
               [-0.46, -0.67, 0.58, -0.60], [-0.46, -0.51, -0.61, 0.51]])
prng = np.random.default_rng(1)
def params(k):
    mats = np.stack([M0 * k + prng.normal(0, 0.05, (4, 4)) for _ in range(K)])
    gaps = np.zeros((K, 4))
    gaps[:L, :] = np.abs(prng.normal(1.0, 0.2, (L, 4)))   # open_a, extend_a, open_b, extend_b
    gaps[FREE] = 0.0
    gaps[DUMMY] = [1.0, 1.0, 1.0, 1.0]
    return nwgrad.AlignParams.from_arrays(mats, gaps, "ATCG")
cls = nwgrad.SeqPairBatchDouble if prec == "double" else nwgrad.SeqPairBatch
mode = "global" if arm == "free" else "glocal"
p0 = params(1.0)
b = cls(A, B, p0, n_threads=threads, traceback="pointers", gap_model=gm, mode=mode,
        grad_mode="hard", tracks=tracks, gap_tracks=gap_tracks)
b.fill = "striped" if arm == "striped" else "interpair"
plan = b._inter_plan() if b.fill == "interpair" else None
b.score_and_grad()
w = rng.normal(size=len(A))
ps = [params(1.0 + 0.01 * (k + 1)) for k in range(5)]
ts = []
for p in ps:
    t = time.perf_counter()
    b.set_params(p); b.score_and_grad(); s = b.scores(); g = b.weighted_grad(w)
    ts.append(time.perf_counter() - t)
print(json.dumps({"s": min(ts), "plan": plan, "scores": [float(x) for x in s[:2000]]}))
'''


def run(py, arm, args, cfg):
    out = subprocess.run([py, "-c", CHILD, arm, str(args.n), str(cfg["t"]), args.data, cfg["prec"], cfg["gm"]],
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--threads", type=int, nargs="+", default=[1])
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--precision", nargs="+", default=["float", "double"])
    ap.add_argument("--models", nargs="+", default=["linear", "affine"])
    args = ap.parse_args()
    arms = ["free", "striped", "interpair", "control"]
    print("prec   model  thr | free s   | striped/free | interpair/free | control/interpair | inter plan [shared, own, groups]")
    for prec in args.precision:
        for gm in args.models:
            for t in args.threads:
                cfg = dict(prec=prec, gm=gm, t=t)
                best = {a: None for a in arms}
                for _ in range(args.rounds):
                    for a in arms:
                        r = run(args.python, "interpair" if a == "control" else a, args, cfg)
                        if best[a] is None or r["s"] < best[a]["s"]:
                            best[a] = r
                ref = best["free"]["scores"]
                tol = 1e-4 if prec == "float" else 1e-9
                bad = [a for a in ("striped", "interpair")
                       if max(abs(x - y) for x, y in zip(best[a]["scores"], ref)) > tol * 50]
                f = best["free"]["s"]
                print(f"{prec:6s} {gm:6s} {t:3d} | {f:8.4f} | {best['striped']['s'] / f:11.3f}x |"
                      f" {best['interpair']['s'] / f:13.3f}x | {best['control']['s'] / best['interpair']['s']:16.3f}x |"
                      f" {best['interpair']['plan']}" + (f"  SCORES DIFFER: {bad}" if bad else ""), flush=True)


if __name__ == "__main__":
    main()
