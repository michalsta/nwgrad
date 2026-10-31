#!/usr/bin/env python3
"""The float32 soft path against the double one, in ONE process and binary.

    python tools/bench_soft_f32.py                      # DNA 22x50 and protein L100/L300, 1 thread
    python tools/bench_soft_f32.py --threads 1 12 --isa avx2 sse4 --lens 100 300 1000

`soft_precision` is a runtime switch of SeqPairBatch, so every arm here is the same
code selected at runtime (no code-placement noise: see bench_simd.py).  Per (fixture,
isa, threads, gap model, mode, fill): a soft score_and_grad() with soft_precision
"double" (A), "float32" (B), and the double class SeqPairBatchDouble (C, the Viterbi
in double too); a CONTROL arm (A against a second "double" batch) first — if it is not
~1.00x, the numbers are noise.  Reports us/pair and B/A, C/A, best of --repeats.
The float32 results are checked against double's within --rel (log Z and the summed
gradient), or the run fails.
"""
from __future__ import annotations
import argparse
import os
import sys
import time

import numpy as np


def fixture(nwgrad, name, n, length, seed):
    rng = np.random.default_rng(seed)
    if name == "dna":
        alpha = "ACGT"
        M = [[0.53, -0.61, -0.60, -0.56], [-0.44, 0.58, -0.57, -0.48],
             [-0.46, -0.67, 0.58, -0.60], [-0.46, -0.51, -0.61, 0.51]]
        p = nwgrad.AlignParams(nwgrad.SubstMatrix(np.array(M), alphabet=alpha), 1.0, 0.5, 1.0, 0.5)
        la, lb = 22, 50
        syms = np.array(list(alpha))
        A, B = [], []
        for _ in range(n):
            a = syms[rng.integers(0, 4, la)]
            b = syms[rng.integers(0, 4, lb)]
            off = int(rng.integers(0, lb - la + 1))
            mut = rng.random(la) < 0.2
            b[off:off + la] = np.where(mut, b[off:off + la], a)
            A.append("".join(a)); B.append("".join(b))
        return p, A, B
    from nwgrad import matrices
    p = nwgrad.AlignParams(matrices.BLOSUM62 * 0.3, 11.0 * 0.3, 1.0 * 0.3, 11.0 * 0.3, 1.0 * 0.3)
    alpha = p.matrix.alphabet
    syms = np.array(list(alpha.symbols if hasattr(alpha, "symbols") else alpha))
    A, B = [], []
    for _ in range(n):
        a = syms[rng.integers(0, len(syms), length)]
        b = a.copy()
        mut = rng.random(length) < 0.3
        b[mut] = syms[rng.integers(0, len(syms), int(mut.sum()))]
        A.append("".join(a)); B.append("".join(b))
    return p, A, B


def timeit(fn, min_time):
    t0 = time.perf_counter(); reps = 0
    while True:
        fn(); reps += 1
        dt = time.perf_counter() - t0
        if dt >= min_time: break
    return dt / reps


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--fixture", nargs="+", default=["dna", "protein"], choices=["dna", "protein"])
    ap.add_argument("--n", type=int, default=20000, help="pairs per batch (dna)")
    ap.add_argument("--n-protein", type=int, default=400)
    ap.add_argument("--lens", type=int, nargs="+", default=[100, 300], help="protein lengths")
    ap.add_argument("--threads", type=int, nargs="+", default=[1])
    ap.add_argument("--isa", nargs="+", default=None)
    ap.add_argument("--fills", nargs="+", default=["interpair", "striped"])
    ap.add_argument("--configs", nargs="+", default=["local-affine", "global-affine", "local-linear", "global-linear"])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--min-time", type=float, default=0.5)
    ap.add_argument("--rel", type=float, default=1e-3)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--json", metavar="FILE")
    args = ap.parse_args()
    if args.quick: args.repeats, args.min_time = 2, 0.2
    os.environ.setdefault("LC_ALL", "C")
    import nwgrad
    isas = args.isa or [l for l in nwgrad.available_isa_levels() if l != "scalar_fallback"]
    print(f"nwgrad {nwgrad.__version__}  {nwgrad.compiled_with()}  levels {nwgrad.available_isa_levels()}")
    out = {"compiler": nwgrad.compiled_with(), "arms": {}}
    cases = []
    for fx in args.fixture:
        if fx == "dna": cases.append(("dna", 0))
        else: cases += [("protein", L) for L in args.lens]
    for fx, L in cases:
        n = args.n if fx == "dna" else max(32, args.n_protein * 100 // max(L, 100))
        p, A, B = fixture(nwgrad, fx, n, L, args.seed)
        tag = fx if fx == "dna" else f"protein L{L}"
        print(f"\n== {tag}: {n} pairs")
        for isa in isas:
            for th in args.threads:
                print(f"-- {isa}, {th} thread(s): us/pair; B/A = float32/double (same class), C/A = SeqPairBatchDouble/double")
                print(f"   {'config':14s} {'fill':10s} {'ctrl':>5s} {'A double':>10s} {'B float32':>10s} {'B/A':>6s} {'C dbl cls':>10s} {'C/A':>6s}")
                for cfg in args.configs:
                    mode, gm = cfg.split("-")
                    for fill in args.fills:
                        kw = dict(gap_model=gm, mode=mode, n_threads=th, kernel=isa, grad_mode="soft", fill=fill)
                        arms = {}
                        for name, cls, prec in [("ctrl", nwgrad.SeqPairBatch, "double"),
                                                ("A", nwgrad.SeqPairBatch, "double"),
                                                ("B", nwgrad.SeqPairBatch, "float32"),
                                                ("C", nwgrad.SeqPairBatchDouble, None)]:
                            b = cls(A, B, p, **kw)
                            b.soft_impl = "scaled_or_log"
                            if prec: b.soft_precision = prec
                            b.score_and_grad()
                            arms[name] = b
                        # correctness: float32 within --rel of double
                        ra, rb = arms["A"], arms["B"]
                        sa, sb = np.asarray(ra.scores()), np.asarray(rb.scores())
                        ga, gb = ra.compute_grad(), rb.compute_grad()
                        for x, y, what in [(sa, sb, "scores"), (np.asarray(ga.matrices), np.asarray(gb.matrices), "matrices"),
                                           (np.asarray(ga.gaps), np.asarray(gb.gaps), "gaps")]:
                            err = np.max(np.abs(x - y) / (np.abs(x) + 1e-6 + 1e-3 * np.max(np.abs(x))))
                            if not err <= args.rel:
                                raise SystemExit(f"{tag} {isa} {cfg} {fill}: float32 {what} differ from double by {err:.2e}")
                        best = {k: float("inf") for k in arms}
                        for r in range(args.repeats):
                            order = list(arms) if r % 2 == 0 else list(arms)[::-1]
                            for k in order:
                                best[k] = min(best[k], timeit(arms[k].score_and_grad, args.min_time))
                        us = {k: v / n * 1e6 for k, v in best.items()}
                        print(f"   {cfg:14s} {fill:10s} {us['ctrl'] / us['A']:5.2f} {us['A']:10.3f} {us['B']:10.3f} "
                              f"{us['B'] / us['A']:6.2f} {us['C']:10.3f} {us['C'] / us['A']:6.2f}")
                        for k, v in us.items():
                            out["arms"][f"{tag}/{isa}/{th}/{cfg}/{fill}/{k}"] = v
    if args.json:
        import json
        with open(args.json, "w") as f: json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
