#!/usr/bin/env python3
"""The three alignment modes side by side on the paths that are not plain hard Full fills:
soft (per pair and inter-pair), banded (per pair and inter-pair), and the tracebacks on
long pairs (pointers / hirschberg / hirschberg_pmax).  A mode-specific gap shows up as a
column out of line with the other two.

    python tools/bench_modes.py [--n 20000] [--threads 1 6] [--reps 5]
        [--parts soft banded hb] [--long 2000]

Pairs: miRNA-like A (18-25) against 50-long B, DNA, a fixed 4x4 matrix with ties broken.
Each cell: the best of --reps calls (seconds), after one warm-up call.
"""
import argparse
import random
import time

import numpy as np

import nwgrad

M = np.array([[0.53, -0.61, -0.60, -0.56], [-0.44, 0.58, -0.57, -0.48],
              [-0.46, -0.67, 0.58, -0.60], [-0.46, -0.51, -0.61, 0.51]])
MODES = ("global", "local", "glocal")


def best(f, reps):
    f()
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
    return min(ts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--threads", type=int, nargs="+", default=[1])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--parts", nargs="+", default=["soft", "banded", "hb"])
    ap.add_argument("--long", type=int, default=2000, help="A's length for the traceback part")
    ap.add_argument("--band", type=int, default=8)
    args = ap.parse_args()
    rng = random.Random(1)
    A = ["".join(rng.choice("ACGT") for _ in range(rng.randint(18, 25))) for _ in range(args.n)]
    B = ["".join(rng.choice("ACGT") for _ in range(50)) for _ in range(args.n)]
    p = nwgrad.AlignParams(nwgrad.SubstMatrix(M, alphabet="ACGT"), gap_open_a=1.0, gap_extend_a=0.8,
                           gap_open_b=1.2, gap_extend_b=0.9)
    print(f"nwgrad {nwgrad.get_isa_level()}, {args.n} pairs; seconds, best of {args.reps}")
    print(f"{'part':28s} {'thr':>3s} " + " ".join(f"{m:>9s}" for m in MODES) + "   glocal/global")
    for prec, cls in (("double", nwgrad.SeqPairBatchDouble), ("float", nwgrad.SeqPairBatch)):
        for t in args.threads:
            rows = []
            for gm in ("linear", "affine"):
                if "soft" in args.parts:
                    for fill in ("striped", "interpair"):
                        r = []
                        for mode in MODES:
                            b = cls(A, B, p, gap_model=gm, mode=mode, grad_mode="soft", n_threads=t)
                            b.fill = fill
                            b.soft_guide = "lazy"
                            r.append(best(b.score_and_grad, args.reps))
                        rows.append((f"{prec} {gm} soft {fill}", r))
                if "banded" in args.parts:
                    for fill in ("striped", "interpair"):
                        r = []
                        for mode in MODES:
                            b = cls(t, gap_model=gm, mode=mode, grad_mode="hard")
                            b.fill = fill
                            r.append(best(lambda: b.align(A, B, p, args.band, [], [], "auto"), args.reps))
                        rows.append((f"{prec} {gm} band{args.band} {fill}", r))
            for name, r in rows:
                print(f"{name:28s} {t:3d} " + " ".join(f"{x:9.4f}" for x in r) + f"   {r[2] / r[0]:6.3f}", flush=True)
    if "hb" in args.parts:
        a = "".join(rng.choice("ACGT") for _ in range(args.long))
        b = ("".join(rng.choice("ACGT") for _ in range(args.long)) + a[args.long // 10: -args.long // 10]
             + "".join(rng.choice("ACGT") for _ in range(args.long)))
        print(f"\ntracebacks, one {len(a)} x {len(b)} pair (score_and_grad), seconds")
        for prec, cls in (("double", nwgrad.SeqPairDouble), ("float", nwgrad.SeqPair)):
            for gm in ("linear", "affine"):
                for tb in ("pointers", "hirschberg", "hirschberg_pmax"):
                    if gm == "linear" and tb == "hirschberg_pmax":
                        continue
                    r = []
                    for mode in MODES:
                        sp = cls(a, b, p, gap_model=gm, mode=mode, traceback=tb)
                        r.append(best(sp.score_and_grad, args.reps))
                    print(f"{prec + ' ' + gm + ' ' + tb:28s}     " + " ".join(f"{x:9.4f}" for x in r)
                          + f"   {r[2] / r[0]:6.3f}", flush=True)


if __name__ == "__main__":
    main()
