"""Soft (forward-backward) path vs hard interpair, on miRNA x target-site pairs.

python tools/bench_soft.py --data .../manakov_fit_rc.tsv --n 100000 --threads 1 12

Per (mode, gap model): SeqPairBatchDouble.score_and_grad() with grad_mode="soft"
(each soft_impl given) against grad_mode="hard"; both with traceback="pointers",
fill="interpair" (soft pairs share only the guide Viterbi fill).  Reports us/pair, best of --reps, and soft/hard.  The matrix is
DiscrimAlign's fitted local-linear one (entries ~ +-0.6); gaps as below.
"""
import argparse, csv, time

import numpy as np
import nwgrad

M = [[0.5317913251634095, -0.6093661897398082, -0.6015643682535813, -0.5647387919107214],
     [-0.4403381513087269, 0.5844189960899253, -0.573571888213131, -0.4814330380001284],
     [-0.45585729543303277, -0.6690642220229696, 0.5820651816141649, -0.5996365659434386],
     [-0.45719211693446804, -0.5103276839824311, -0.6113086433867316, 0.5136569234033567]]

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--n", type=int, default=100000)
ap.add_argument("--threads", type=int, nargs="+", default=[1])
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--impl", nargs="+", default=None,
                help="soft_impl values to time (default: the build's default only)")
ap.add_argument("--soft-guide", default="eager", choices=["eager", "lazy"],
                help="SeqPairBatch.soft_guide for the soft arm")
ap.add_argument("--prec", default="double", choices=["double", "float32"],
                help="batch precision (float32: SeqPairBatch; the soft pass is double either way)")
ap.add_argument("--configs", nargs="+",
                default=["local-affine", "global-affine", "local-linear", "global-linear"])
args = ap.parse_args()

with open(args.data) as f:
    rows = list(csv.reader(f, delimiter="\t"))[1:]
idx = np.random.default_rng(0).choice(len(rows), size=args.n, replace=False)
A = [rows[i][0] for i in idx]
B = [rows[i][1] for i in idx]

params = nwgrad.AlignParams(nwgrad.SubstMatrix(np.array(M), "ACGT"), 1.0, 0.5, 1.0, 0.5)
lin = nwgrad.AlignParams(nwgrad.SubstMatrix(np.array(M), "ACGT"), 0.0, 1.2147, 0.0, 1.2147)


def run(threads, mode, gap, grad_mode, tb="auto", fill=None, impl=None):
    cls = nwgrad.SeqPairBatch if args.prec == "float32" else nwgrad.SeqPairBatchDouble
    b = cls(n_threads=threads, traceback=tb)
    if fill:
        b.fill = fill
    if impl is not None:
        b.soft_impl = impl
    if grad_mode == "soft":
        b.soft_guide = args.soft_guide
    b.add_many(A, B, params if gap == "affine" else lin, gap_model=gap, mode=mode,
               grad_mode=grad_mode)
    b.score_and_grad()  # warm: allocation, plan
    best = 1e30
    for _ in range(args.reps):
        t = time.perf_counter()
        b.score_and_grad()
        best = min(best, time.perf_counter() - t)
    return best / len(A) * 1e6


print(f"isa={nwgrad.simd_isa()} n={args.n} soft_guide={args.soft_guide} prec={args.prec}")
for threads in args.threads:
    for cfg in args.configs:
        mode, gap = cfg.split("-")
        hard = run(threads, mode, gap, "hard", tb="pointers", fill="interpair")
        line = f"t={threads:3d} {cfg:14s} hard_interpair {hard:8.3f} us/pair"
        for impl in (args.impl or [None]):
            soft = run(threads, mode, gap, "soft", tb="pointers", fill="interpair", impl=impl)
            line += f" | soft[{impl or 'default'}] {soft:8.3f} ({soft / hard:5.1f}x)"
        print(line, flush=True)
