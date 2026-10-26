#!/usr/bin/env python3
"""The hard-gradient traceback modes against each other and against score-only, one build.

    python tools/bench_tracebacks.py [--lengths 300 1000 3000 10000] [--threads 1 12]
                                     [--rounds 5] [--modes ...] [--prec f32 f64]

Related protein pairs (BLOSUM62, gaps 11/1, 25 % substitutions, 6 % indels), the striped
per-pair fill, score_and_grad() after one warm-up call.  Every measurement in a fresh
process; the modes interleave within each round, best of --rounds.  Reports ns per DP cell
and each mode's ratio to score-only (grad_mode="none").
"""
import argparse, subprocess, sys

CHILD = r'''
import sys, time, warnings
warnings.simplefilter("ignore")
import numpy as np, nwgrad
from nwgrad.matrices import BLOSUM62
mode, L, prec, threads = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
rng = np.random.default_rng(L)
P = "ACDEFGHIKLMNPQRSTVWY"
npairs = max(1, int(round(12e6 / (L * L)))) * threads   # ~12 M cells per thread
A, B = [], []
for _ in range(npairs):
    a = "".join(rng.choice(list(P), L))
    b = []
    for c in a:
        r = rng.random()
        if r < 0.03: continue
        if r < 0.06: b.append(P[int(rng.integers(20))])
        b.append(c if rng.random() > 0.25 else P[int(rng.integers(20))])
    A.append(a); B.append("".join(b))
p = nwgrad.AlignParams(BLOSUM62, 11.0, 1.0, 11.0, 1.0)
cls = nwgrad.SeqPairBatch if prec == "f32" else nwgrad.SeqPairBatchDouble
kw = dict(grad_mode="none") if mode == "none" else dict(grad_mode="hard", traceback=mode)
b = cls(A, B, p, n_threads=threads, **kw)
b.fill = "striped"
b.score_and_grad()
t = time.perf_counter(); b.score_and_grad(); dt = time.perf_counter() - t
print(dt / sum(len(x) * len(y) for x, y in zip(A, B)) * 1e9)
'''

MODES = ["none", "pointers", "hirschberg", "hirschberg_pmax", "grid"]


def run(mode, L, prec, thr):
    out = subprocess.run([sys.executable, "-c", CHILD, mode, str(L), prec, str(thr)],
                         capture_output=True, text=True)
    if out.returncode:
        raise RuntimeError(out.stderr[-2000:])
    return float(out.stdout.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=int, nargs="+", default=[300, 1000, 3000, 10000])
    ap.add_argument("--threads", type=int, nargs="+", default=[1])
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--modes", nargs="+", default=MODES)
    ap.add_argument("--prec", nargs="+", default=["f32", "f64"])
    args = ap.parse_args()
    modes = ["none"] + [m for m in args.modes if m != "none"]
    print(f"{'prec':4s} {'thr':>3s} {'L':>6s}  " + "  ".join(f"{m:>18s}" for m in modes))
    for thr in args.threads:
        for prec in args.prec:
            for L in args.lengths:
                best = {m: float("inf") for m in modes}
                for _ in range(args.rounds):
                    for m in modes:
                        best[m] = min(best[m], run(m, L, prec, thr))
                base = best["none"]
                print(f"{prec:4s} {thr:3d} {L:6d}  " +
                      "  ".join(f"{best[m]:7.3f} ns {best[m] / base:5.2f}x" for m in modes), flush=True)


if __name__ == "__main__":
    main()
