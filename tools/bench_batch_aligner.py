"""Streaming align (BatchAligner.align, SeqPairBatch.align from 0.6) per fill, on miRNA x target-site pairs.

python tools/bench_batch_aligner.py --data .../manakov_fit_rc.tsv --n 100000 --threads 1 12

Per (precision, grad mode, mode, gap model): fill="striped" vs fill="interpair",
us/pair, best of --reps.  Same pairs and matrix as tools/bench_fill.py.
"""
import argparse, csv, time

import numpy as np
import nwgrad
from _stream import Stream

M = [[0.5317913251634095, -0.6093661897398082, -0.6015643682535813, -0.5647387919107214],
     [-0.4403381513087269, 0.5844189960899253, -0.573571888213131, -0.4814330380001284],
     [-0.45585729543303277, -0.6690642220229696, 0.5820651816141649, -0.5996365659434386],
     [-0.45719211693446804, -0.5103276839824311, -0.6113086433867316, 0.5136569234033567]]

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--n", type=int, default=100000)
ap.add_argument("--threads", type=int, nargs="+", default=[1])
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--prec", nargs="+", default=["float32", "double"])
ap.add_argument("--grad", nargs="+", default=["hard", "soft"])
ap.add_argument("--band", type=int, default=0)
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


def run(prec, threads, mode, gap, grad, fill):
    ba = Stream(params if gap == "affine" else lin, band=args.band, gap_model=gap,
                mode=mode, grad_mode=grad, n_threads=threads, double=prec == "double")
    ba.fill = fill
    ba.align(A, B)
    best = 1e30
    for _ in range(args.reps):
        t = time.perf_counter()
        ba.align(A, B)
        best = min(best, time.perf_counter() - t)
    return best / len(A) * 1e6


print(f"isa={nwgrad.simd_isa()} n={len(A)} band={args.band}")
print(f"{'prec':8s} {'thr':>3s} {'grad':5s} {'config':14s} {'striped':>8s} {'interpair':>9s} {'ratio':>6s}")
for prec in args.prec:
    for t in args.threads:
        for grad in args.grad:
            for cfg in args.configs:
                mode, gap = cfg.split("-")
                s = run(prec, t, mode, gap, grad, "striped")
                i = run(prec, t, mode, gap, grad, "interpair")
                print(f"{prec:8s} {t:3d} {grad:5s} {cfg:14s} {s:8.3f} {i:9.3f} {i / s:6.2f}",
                      flush=True)
