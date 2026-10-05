"""Hard score_and_grad() per fill and precision, on miRNA x target-site pairs.

python tools/bench_fill.py --data .../manakov_fit_rc.tsv --n 100000 --threads 1 12

Per (precision, mode, gap model): SeqPairBatch[Double] with fill="striped" vs
fill="interpair", traceback="pointers"; plus the float32 default traceback ("auto")
for both fills, which is what a bare SeqPairBatch() runs.  Reports us/pair, best of
--reps, and interpair/striped.  Same pairs and matrix as tools/bench_soft.py.
"""
import argparse, csv, time

import numpy as np
import nwgrad

M = [[0.5317913251634095, -0.6093661897398082, -0.6015643682535813, -0.5647387919107214],
     [-0.4403381513087269, 0.5844189960899253, -0.573571888213131, -0.4814330380001284],
     [-0.45585729543303277, -0.6690642220229696, 0.5820651816141649, -0.5996365659434386],
     [-0.45719211693446804, -0.5103276839824311, -0.6113086433867316, 0.5136569234033567]]

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=None)
ap.add_argument("--n", type=int, default=100000)
ap.add_argument("--threads", type=int, nargs="+", default=[1])
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--prec", nargs="+", default=["float32", "double"])
ap.add_argument("--tb", nargs="+", default=["pointers", "auto"])
ap.add_argument("--band", type=int, nargs="*", default=[],
                help="also time banded_grad(bw) after one score_and_grad, per fill")
ap.add_argument("--vary-b", type=int, nargs=2, metavar=("LO", "HI"), default=None,
                help="replace each B by a random DNA sequence of length in [LO, HI] (seed 1): "
                     "the ragged-B case, many B lengths each rare")
ap.add_argument("--protein", type=int, nargs=4, metavar=("ALO", "AHI", "BLO", "BHI"),
                default=None, help="random canonical-20 protein pairs, len A in [ALO, AHI], "
                "len B in [BLO, BHI] (--n of them, seed 2), random N(0,1) matrix; --data ignored")
ap.add_argument("--grad", default="hard", choices=["hard", "soft"])
ap.add_argument("--configs", nargs="+",
                default=["local-affine", "global-affine", "local-linear", "global-linear"])
args = ap.parse_args()

if args.protein:
    AA = "ARNDCQEGHILKMFPSTWYV"
    rng = np.random.default_rng(2)
    a0, a1, b0, b1 = args.protein
    A = ["".join(rng.choice(list(AA), int(k))) for k in rng.integers(a0, a1 + 1, args.n)]
    B = ["".join(rng.choice(list(AA), int(k))) for k in rng.integers(b0, b1 + 1, args.n)]
else:
    with open(args.data) as f:
        rows = list(csv.reader(f, delimiter="\t"))[1:]
    idx = np.random.default_rng(0).choice(len(rows), size=args.n, replace=False)
    A = [rows[i][0] for i in idx]
    B = [rows[i][1] for i in idx]
if args.vary_b:
    rng = np.random.default_rng(1)
    B = ["".join(rng.choice(list("ACGT"), int(k)))
         for k in rng.integers(args.vary_b[0], args.vary_b[1] + 1, len(B))]

if args.protein:
    PM = np.random.default_rng(3).normal(size=(20, 20))
    params = nwgrad.AlignParams(nwgrad.SubstMatrix(PM, AA), 3.0, 1.0, 3.0, 1.0)
    lin = nwgrad.AlignParams(nwgrad.SubstMatrix(PM, AA), 0.0, 1.5, 0.0, 1.5)
else:
    params = nwgrad.AlignParams(nwgrad.SubstMatrix(np.array(M), "ACGT"), 1.0, 0.5, 1.0, 0.5)
    lin = nwgrad.AlignParams(nwgrad.SubstMatrix(np.array(M), "ACGT"), 0.0, 1.2147, 0.0, 1.2147)
CLS = {"float32": nwgrad.SeqPairBatch, "double": nwgrad.SeqPairBatchDouble}


def run(prec, threads, mode, gap, tb, fill, bw=0):
    b = CLS[prec](n_threads=threads, traceback=tb)
    b.fill = fill
    b.add_many(A, B, params if gap == "affine" else lin, gap_model=gap, mode=mode,
               grad_mode=args.grad)
    b.score_and_grad()  # warm: allocation, plan (and the guides banded_grad needs)
    step = (lambda: b.banded_grad(bw)) if bw else b.score_and_grad
    if bw:
        step()
    best = 1e30
    for _ in range(args.reps):
        t = time.perf_counter()
        step()
        best = min(best, time.perf_counter() - t)
    return best / len(A) * 1e6


print(f"isa={nwgrad.simd_isa()} n={len(A)} grad={args.grad}")
print(f"{'prec':8s} {'thr':>3s} {'config':14s} {'tb':8s} {'striped':>8s} {'interpair':>9s} {'ratio':>6s}")
for prec in args.prec:
    for t in args.threads:
        for cfg in args.configs:
            mode, gap = cfg.split("-")
            for tb in args.tb:
                if tb == "auto" and prec == "double" and gap == "affine":
                    continue   # double affine auto = exact hirschberg; pointers row covers it
                for bw in [0] + args.band:
                    s = run(prec, t, mode, gap, tb, "striped", bw)
                    i = run(prec, t, mode, gap, tb, "interpair", bw)
                    lab = tb if not bw else f"band{bw}"
                    print(f"{prec:8s} {t:3d} {cfg:14s} {lab:8s} {s:8.3f} {i:9.3f} {i / s:6.2f}",
                          flush=True)
