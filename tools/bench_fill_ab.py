#!/usr/bin/env python3
"""A/B benchmark of the hard-gradient fills between two nwgrad builds.

    python tools/bench_fill_ab.py --a OLD_PYTHON --b NEW_PYTHON [--threads 1 12] \\
        [--rounds 5] [--sections ...] [--control]

Every measurement in a fresh process, the builds alternating, best of --rounds per arm;
--control runs A against itself first (read every ratio against it).  Each section is one
traceback mode's fill, a SeqPairBatch(..., fill="striped") score_and_grad() over 60 protein
pairs of 300-1500 aa (planted homology, 25 % substitutions; pairs over hb_cutoff = 512 take
the Hirschberg recursion), after one untimed warm-up call:
  ptr_g_f32 / ptr_g_f64      traceback="pointers", affine Global
  ptr_l_f64                  traceback="pointers", affine Local
  scores_g_f64               traceback="scores" (the striped Scores fill), affine Global
  hb_g_f32 / hb_g_f64        traceback="hirschberg" (exact sweeps + base case), Global
  pmax_g_f32                 traceback="hirschberg_pmax" (prefix-max sweeps, exact base case)
  lin_ptr_f64                traceback="pointers", linear Global (the striped linear kernel)
  lin_hb_f64                 traceback="hirschberg", linear Global (the linear sweep)
"""
import argparse, json, subprocess

CHILD = r'''
import sys, time, warnings
warnings.simplefilter("ignore")
import numpy as np, nwgrad
sec, threads = sys.argv[1], int(sys.argv[2])
rng = np.random.default_rng(0)
P = "ACDEFGHIKLMNPQRSTVWY"
pm = rng.normal(-0.7, 0.6, (20, 20)); np.fill_diagonal(pm, rng.normal(2.0, 0.3, 20))
pp = nwgrad.AlignParams(nwgrad.SubstMatrix(pm, alphabet=P), 3.3, 0.3, 3.3, 0.3)
A, B = [], []
for _ in range(60):
    a = "".join(rng.choice(list(P), int(rng.integers(300, 1500))))
    b = []
    for c in a:
        r = rng.random()
        if r < 0.02: continue
        if r < 0.04: b.append(P[int(rng.integers(20))])
        b.append(c if rng.random() > 0.25 else P[int(rng.integers(20))])
    A.append(a); B.append("".join(b))
tb, gm, prec = {
    "ptr_g_f32": ("pointers", "affine", 32), "ptr_g_f64": ("pointers", "affine", 64),
    "ptr_l_f64": ("pointers", "affine", 64), "scores_g_f64": ("scores", "affine", 64),
    "hb_g_f32": ("hirschberg", "affine", 32), "hb_g_f64": ("hirschberg", "affine", 64),
    "pmax_g_f32": ("hirschberg_pmax", "affine", 32),
    "lin_ptr_f64": ("pointers", "linear", 64), "lin_hb_f64": ("hirschberg", "linear", 64),
}[sec]
mode = "local" if "_l_" in sec else "global"
cls = nwgrad.SeqPairBatch if prec == 32 else nwgrad.SeqPairBatchDouble
b = cls(A, B, pp, n_threads=threads, traceback=tb, gap_model=gm, mode=mode, grad_mode="hard")
b.fill = "striped"
b.score_and_grad()
t = time.perf_counter(); b.score_and_grad(); print(time.perf_counter() - t)
'''

SECTIONS = ["ptr_g_f32", "ptr_g_f64", "ptr_l_f64", "scores_g_f64", "hb_g_f32", "hb_g_f64",
            "pmax_g_f32", "lin_ptr_f64", "lin_hb_f64"]


def run(py, sec, threads):
    out = subprocess.run([py, "-c", CHILD, sec, str(threads)], capture_output=True, text=True)
    if out.returncode:
        raise RuntimeError(out.stderr[-2000:])
    return float(out.stdout.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--threads", type=int, nargs="+", default=[1])
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--sections", nargs="+", default=SECTIONS)
    ap.add_argument("--control", action="store_true")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    rows = []
    print(f"{'section':14s} {'thr':>3s} {'A':>9s} {'A_':>9s} {'B':>9s}   B/A  A'/A  (s)")
    for thr in args.threads:
        for sec in args.sections:
            t = {"a": [], "a2": [], "b": []}
            for _ in range(args.rounds):
                t["a"].append(run(args.a, sec, thr))
                if args.control: t["a2"].append(run(args.a, sec, thr))
                t["b"].append(run(args.b, sec, thr))
            a, b = min(t["a"]), min(t["b"])
            a2 = min(t["a2"]) if t["a2"] else float("nan")
            rows.append(dict(section=sec, threads=thr, a=a, a2=a2, b=b))
            print(f"{sec:14s} {thr:3d} {a:9.4f} {a2:9.4f} {b:9.4f}  {b / a:5.2f} {a2 / a:5.2f}",
                  flush=True)
    if args.json:
        json.dump(rows, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
