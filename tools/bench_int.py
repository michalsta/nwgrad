#!/usr/bin/env python3
"""Benchmark the scalar types against each other: float32 (the default), double and
the integer types int16 / int32 / int64, on the same integral parameters.

    python tools/bench_int.py                          # DNA 22x50 and protein, 1 thread
    python tools/bench_int.py --threads 1 12 --isa avx2 sse4
    python tools/bench_int.py --fixture protein --len 300 --quick

One process, one binary: every arm is a runtime choice (the class, the `kernel=`
backend), so codegen is identical across arms (see bench_simd.py for why).  A CONTROL
arm (float32 vs a second float32 batch) is reported first; if it is not ~1.00x, the
numbers are noise.  Results are checked: every type's scores and summed gradient must
equal double's bit for bit (integral parameters make double exact), or the run fails.

Arms per (fixture, isa, threads): a hard training step (score_and_grad on the batch),
a score-only batch (grad_mode="none") and streaming align().  Reported in us/pair and
as time / float32's time (below 1.00 = faster than the float32 default).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

TYPES = ["float32", "double", "int16", "int32", "int64"]


def batch_class(nwgrad, t):
    return {"float32": nwgrad.SeqPairBatch, "double": nwgrad.SeqPairBatchDouble,
            "int16": nwgrad.SeqPairBatchInt16, "int32": nwgrad.SeqPairBatchInt32,
            "int64": nwgrad.SeqPairBatchInt64}[t]


def make_fixture(nwgrad, name, n, length, seed):
    rng = np.random.default_rng(seed)
    if name == "dna":
        alpha = "ACGT"
        m = np.where(np.eye(4, dtype=bool), 2.0, -3.0)
        p = nwgrad.AlignParams(nwgrad.SubstMatrix(m, alphabet=alpha), 5.0, 2.0, 5.0, 2.0)
        la, lb = 22, 50
        # related pairs: B contains a mutated copy of A at a random offset
        seqs_a, seqs_b = [], []
        syms = np.array(list(alpha))
        for _ in range(n):
            a = syms[rng.integers(0, 4, la)]
            b = syms[rng.integers(0, 4, lb)]
            off = int(rng.integers(0, lb - la + 1))
            mut = rng.random(la) < 0.2
            b[off:off + la] = np.where(mut, b[off:off + la], a)
            seqs_a.append("".join(a)); seqs_b.append("".join(b))
        return p, seqs_a, seqs_b
    if name == "protein":
        from nwgrad import matrices
        p = nwgrad.AlignParams(matrices.BLOSUM62, 11.0, 1.0, 11.0, 1.0)
        alpha = p.matrix.alphabet
        syms = np.array(list(alpha.symbols if hasattr(alpha, "symbols") else alpha))
        seqs_a, seqs_b = [], []
        for _ in range(n):
            a = syms[rng.integers(0, len(syms), length)]
            b = a.copy()
            mut = rng.random(length) < 0.3
            b[mut] = syms[rng.integers(0, len(syms), int(mut.sum()))]
            seqs_a.append("".join(a)); seqs_b.append("".join(b))
        return p, seqs_a, seqs_b
    raise SystemExit(f"unknown fixture {name}")


def timeit(fn, repeats, min_time):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        reps = 0
        while True:
            fn(); reps += 1
            dt = time.perf_counter() - t0
            if dt >= min_time: break
        best = min(best, dt / reps)
    return best


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--fixture", nargs="+", default=["dna", "protein"], choices=["dna", "protein"])
    ap.add_argument("--n", type=int, default=20000, help="pairs per batch (dna)")
    ap.add_argument("--n-protein", type=int, default=2000, help="pairs per batch (protein)")
    ap.add_argument("--len", type=int, default=200, help="protein length")
    ap.add_argument("--threads", type=int, nargs="+", default=[1])
    ap.add_argument("--isa", nargs="+", default=None, help="backends (default: every level this CPU runs)")
    ap.add_argument("--types", nargs="+", default=TYPES)
    ap.add_argument("--mode", default="local", choices=["global", "local"])
    ap.add_argument("--gap-model", default="affine", choices=["affine", "linear"])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--min-time", type=float, default=1.0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--json", metavar="FILE", help="also write every arm's best time (s/pair) here")
    args = ap.parse_args()
    out = {"arms": {}}
    if args.quick:
        args.repeats, args.min_time = 2, 0.3
    os.environ.setdefault("LC_ALL", "C")
    import nwgrad
    isas = args.isa or [l for l in nwgrad.available_isa_levels() if l != "scalar_fallback"]
    print(f"nwgrad {nwgrad.__version__}  {nwgrad.compiled_with()}  levels {nwgrad.available_isa_levels()}")
    out["compiler"] = nwgrad.compiled_with()

    for fx in args.fixture:
        n = args.n if fx == "dna" else args.n_protein
        p, sa, sb = make_fixture(nwgrad, fx, n, args.len, args.seed)
        print(f"\n== {fx}: {n} pairs, |A|~{len(sa[0])} |B|~{len(sb[0])}, {args.gap_model} {args.mode}")
        for isa in isas:
            for th in args.threads:
                kw = dict(gap_model=args.gap_model, mode=args.mode, n_threads=th, kernel=isa)
                # the arms, built up front (interleaved timing below)
                arms = {}
                # (the double arm is always built: it is the oracle for the check below)
                for t in ["control"] + args.types + (["double"] if "double" not in args.types else []):
                    cls = batch_class(nwgrad, "float32" if t == "control" else t)
                    hard = cls(sa, sb, p, **kw)
                    none = cls(sa, sb, p, grad_mode="none", **kw)
                    stream = cls(n_threads=th, gap_model=args.gap_model, mode=args.mode)
                    arms[t] = (hard, none, stream)
                # correctness: every type == double
                ref = None
                for t, (hard, none, _) in arms.items():
                    hard.score_and_grad(); none.score_and_grad()
                    got = (hard.scores(), hard.compute_grad().matrices, hard.compute_grad().gaps, none.scores())
                    if t == "double": ref = got
                for t, (hard, none, _) in arms.items():
                    if t in ("control", "float32"): continue
                    got = (hard.scores(), hard.compute_grad().matrices, hard.compute_grad().gaps, none.scores())
                    for g, r in zip(got, ref):
                        if not np.array_equal(g, r):
                            raise SystemExit(f"{fx} {isa} {t}: results differ from double")
                timed = ["control"] + args.types
                res = {t: [0.0, 0.0, 0.0] for t in timed}
                for r in range(args.repeats):
                    for t in timed:
                        hard, none, stream = arms[t]
                        for k, fn in enumerate([lambda: hard.score_and_grad(), lambda: none.score_and_grad(),
                                                lambda: stream.align(sa, sb, p, kernel=isa)]):
                            dt = timeit(fn, 1, args.min_time)
                            res[t][k] = dt if r == 0 else min(res[t][k], dt)
                base = res["float32"]
                print(f"-- {isa}, {th} thread(s): us/pair (ratio to float32)")
                print(f"   {'type':8s} {'hard step':>22s} {'score only':>22s} {'stream align':>22s}")
                for t in ["control"] + args.types:
                    cells = []
                    for k in range(3):
                        us = res[t][k] / n * 1e6
                        cells.append(f"{us:10.3f} ({res[t][k] / base[k]:5.2f}x)")
                        out["arms"][f"{fx}/{isa}/{th}/{t}/{['hard', 'score', 'stream'][k]}"] = res[t][k] / n
                    print(f"   {t:8s} " + " ".join(f"{c:>22s}" for c in cells))
    if args.json:
        import json
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
