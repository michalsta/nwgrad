#!/usr/bin/env python3
"""What tracks cost: one training step, tracked vs untracked, in one nwgrad build.

    python tools/bench_tracks.py --data pairs.tsv [--n 100000] [--threads 1 12]
        [--rounds 5] [--sets 25] [--precision double float] [--models linear affine]
        [--modes local global] [--grad hard soft] [--storage paths|counts]

PosimiR's shape: miRNA x target-site pairs (Manakov TSV: column 0 the miRNA, column 1
the site), one parameter set per miRNA position (track = 0, 1, 2, ... clamped to K-1),
the default gap track.  Arms, each in a fresh process, alternating, best of --rounds:
  untracked   one set, no track (the code that existed before tracks)
  tracked1    one set, an all-zero track (the tracked code path, the same numbers)
  trackedK    K sets, one per position
Reported per configuration: seconds per step (set_params + score_and_grad + scores +
weighted_grad) and resident MB added by add_many, with tracked / untracked ratios.
A control arm (untracked twice) shows the noise floor.  --storage sets the tracked
arms' SeqPairBatch.track_storage (a build without it keeps its own store).
"""
import argparse, json, os, subprocess, sys

CHILD = r'''
import json, resource, sys, time, warnings
warnings.simplefilter("ignore")
import numpy as np, nwgrad
arm, n, threads, data, K, prec, gm, mode, gd, storage = sys.argv[1:11]
n, threads, K = int(n), int(threads), int(K)
rows = [l.rstrip("\n").split("\t") for l in open(data)][1:]
rng = np.random.default_rng(0)
idx = rng.choice(len(rows), size=min(n, len(rows)), replace=False)
A = [rows[i][0] for i in idx]; B = [rows[i][1] for i in idx]
M = np.array([[0.53, -0.61, -0.60, -0.56], [-0.44, 0.58, -0.57, -0.48],
              [-0.46, -0.67, 0.58, -0.60], [-0.46, -0.51, -0.61, 0.51]])
G = np.array([1.0, 0.5, 1.0, 0.5])
def params(k, sets):
    if sets == 1 and arm == "untracked":
        return nwgrad.AlignParams(M * k, "ACGT", *G)
    return nwgrad.AlignParams.from_arrays(np.stack([M * k] * sets), np.stack([G] * sets), "ACGT")
def rss_mb():
    for l in open("/proc/self/status"):
        if l.startswith("VmRSS:"): return int(l.split()[1]) / 1024
cls = nwgrad.SeqPairBatchDouble if prec == "double" else nwgrad.SeqPairBatch
b = cls(threads, gap_model=gm, mode=mode, grad_mode=gd)
if hasattr(b, "track_storage"):
    b.track_storage = storage
if gd == "soft":
    b.soft_guide = "lazy"; b.soft_temperature = 0.5; b.soft_impl = "scaled_or_log"
sets = K if arm == "trackedK" else 1
tracks = None
if arm == "tracked1": tracks = [[0] * len(a) for a in A]
if arm == "trackedK": tracks = [[min(p, K - 1) for p in range(len(a))] for a in A]
r0 = rss_mb()
b.add_many(A, B, params(1.0, sets), tracks=tracks) if tracks is not None else b.add_many(A, B, params(1.0, sets))
mb = rss_mb() - r0
b.score_and_grad()
w = rng.normal(size=len(A))
ts = []
for k in range(4):
    p = params(1.0 + 0.01 * (k + 1), sets)
    t = time.perf_counter()
    b.set_params(p); b.score_and_grad(); s = b.scores(); g = b.weighted_grad(w)
    ts.append(time.perf_counter() - t)
print(json.dumps({"s": min(ts), "mb": mb + 0.0}))
'''


def run(py, arm, args, cfg):
    out = subprocess.run([py, "-c", CHILD, arm, str(args.n), str(cfg["t"]), args.data, str(args.sets),
                          cfg["prec"], cfg["gm"], cfg["mode"], cfg["gd"], args.storage],
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--threads", type=int, nargs="+", default=[1])
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--sets", type=int, default=25)
    ap.add_argument("--precision", nargs="+", default=["double", "float"])
    ap.add_argument("--models", nargs="+", default=["linear", "affine"])
    ap.add_argument("--modes", nargs="+", default=["local"])
    ap.add_argument("--grad", nargs="+", default=["hard"])
    ap.add_argument("--storage", choices=["paths", "counts"], default="paths")
    args = ap.parse_args()
    arms = ["untracked", "control", "tracked1", "trackedK"]
    print("prec   model  mode   grad thr | untracked s  MB  | control | tracked1 | trackedK   MB")
    for prec in args.precision:
        for gm in args.models:
            for mode in args.modes:
                for gd in args.grad:
                    for t in args.threads:
                        cfg = dict(prec=prec, gm=gm, mode=mode, gd=gd, t=t)
                        best = {a: None for a in arms}
                        for _ in range(args.rounds):
                            for a in arms:
                                r = run(args.python, "untracked" if a == "control" else a, args, cfg)
                                if best[a] is None or r["s"] < best[a]["s"]:
                                    best[a] = r
                        u = best["untracked"]["s"]
                        print(f"{prec:6s} {gm:6s} {mode:6s} {gd:4s} {t:3d} | {u:9.4f} {best['untracked']['mb']:6.0f} |"
                              f" {best['control']['s'] / u:6.3f}x | {best['tracked1']['s'] / u:7.3f}x |"
                              f" {best['trackedK']['s'] / u:7.3f}x {best['trackedK']['mb']:6.0f}", flush=True)


if __name__ == "__main__":
    main()
