"""Per-pair soft path on protein-sized pairs (alphabet > 8, so no inter-pair pass).

python tools/bench_soft_protein.py --len 300 --n 400 --threads 1

Random sequences over the NCBI protein alphabet, BLOSUM62 scaled by --scale (the
soft path at scale 1 is a ~1/T = 1 ensemble), affine gaps 11/1.  Soft (scaled, eager
and lazy guide) against hard with traceback="pointers"; best of --reps, us/pair and
ns/cell.
"""
import argparse, time

import numpy as np
import nwgrad
from nwgrad.matrices import BLOSUM62

ap = argparse.ArgumentParser()
ap.add_argument("--len", type=int, nargs="+", default=[300])
ap.add_argument("--n", type=int, default=400)
ap.add_argument("--threads", type=int, nargs="+", default=[1])
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--scale", type=float, default=0.3)
ap.add_argument("--modes", nargs="+", default=["local", "global"])
args = ap.parse_args()

alpha = BLOSUM62.alphabet
M = np.asarray(BLOSUM62.to_matrix(), dtype=float)
params = nwgrad.AlignParams(nwgrad.SubstMatrix(M * args.scale, alpha),
                            11 * args.scale, 1 * args.scale, 11 * args.scale, 1 * args.scale)
canon = "ARNDCQEGHILKMFPSTWYV"


def run(threads, L, mode, grad_mode, guide="eager"):
    rng = np.random.default_rng(L)
    A = ["".join(rng.choice(list(canon), L)) for _ in range(args.n)]
    B = ["".join(rng.choice(list(canon), L)) for _ in range(args.n)]
    b = nwgrad.SeqPairBatchDouble(n_threads=threads, traceback="pointers")
    if grad_mode == "soft":
        b.soft_guide = guide
    b.add_many(A, B, params, gap_model="affine", mode=mode, grad_mode=grad_mode)
    b.score_and_grad()
    best = 1e30
    for _ in range(args.reps):
        t = time.perf_counter()
        b.score_and_grad()
        best = min(best, time.perf_counter() - t)
    return best / args.n * 1e6


print(f"isa={nwgrad.simd_isa()} n={args.n} scale={args.scale}")
for threads in args.threads:
    for L in args.len:
        for mode in args.modes:
            h = run(threads, L, mode, "hard")
            se = run(threads, L, mode, "soft", "eager")
            sl = run(threads, L, mode, "soft", "lazy")
            cells = L * L
            print(f"t={threads:3d} len={L:5d} {mode:6s} hard {h:9.1f} us | soft eager {se:9.1f} "
                  f"({se / h:4.1f}x) | soft lazy {sl:9.1f} ({sl / h:4.1f}x, {sl * 1e3 / cells:5.2f} ns/cell)",
                  flush=True)
