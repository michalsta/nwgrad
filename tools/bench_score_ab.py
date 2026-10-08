#!/usr/bin/env python3
"""A/B benchmark of the score-only paths between two nwgrad builds.

    python tools/bench_score_ab.py --a OLD_PYTHON --b NEW_PYTHON --data manakov.tsv \\
        [--threads 1 12] [--rounds 5] [--control]

Every measurement in a fresh process, the builds alternating, best of --rounds per arm;
--control runs A against itself first (read every ratio against it).  Sections:
  mk_align      align() of 100k Manakov pairs (22 x 50, DNA), grad_mode none, local affine
  mk_step       set_params + score_and_grad() of the same, grad_mode none (a batch)
  mk_align_f32  mk_align at float32 (SeqPairBatch)
  prot_align    align() of 400 protein pairs, 300-1000 aa, global affine, float32
  prot_align64  the same at double
  conv          4000 nw_score_affine() calls, protein, 100-400 aa (per-call overhead)
"""
import argparse, json, subprocess, sys

CHILD = r'''
import sys, time, warnings
warnings.simplefilter("ignore")
import numpy as np, nwgrad
sec, threads, data = sys.argv[1], int(sys.argv[2]), sys.argv[3]
rng = np.random.default_rng(0)
def manakov(n):
    rows = [l.rstrip("\n").split("\t") for l in open(data)][1:]
    idx = rng.choice(len(rows), size=min(n, len(rows)), replace=False)
    return [rows[i][0] for i in idx], [rows[i][1] for i in idx]
M = np.array([[0.53, -0.61, -0.60, -0.56], [-0.44, 0.58, -0.57, -0.48],
              [-0.46, -0.67, 0.58, -0.60], [-0.46, -0.51, -0.61, 0.51]])
pd = nwgrad.AlignParams(nwgrad.SubstMatrix(M, alphabet="ACGT"), 1.0, 0.5, 1.0, 0.5)
P = "ACDEFGHIKLMNPQRSTVWY"
pm = rng.normal(-0.7, 0.6, (20, 20)); np.fill_diagonal(pm, rng.normal(2.0, 0.3, 20))
pp = nwgrad.AlignParams(nwgrad.SubstMatrix(pm, alphabet=P), 3.3, 0.3, 3.3, 0.3)
def prot(n, lo, hi):
    A, B = [], []
    for _ in range(n):
        a = "".join(rng.choice(list(P), int(rng.integers(lo, hi))))
        b = list(a)
        for k in range(len(b)):
            if rng.random() < 0.25: b[k] = P[int(rng.integers(20))]
        A.append(a); B.append("".join(b))
    return A, B
def timed(f):
    t = time.perf_counter(); f(); return time.perf_counter() - t
if sec in ("mk_align", "mk_align_f32"):
    A, B = manakov(100000)
    cls = nwgrad.SeqPairBatch if sec.endswith("f32") else nwgrad.SeqPairBatchDouble
    b = cls(threads, gap_model="affine", mode="local", grad_mode="none")
    b.align(A[:1000], B[:1000], pd)
    print(timed(lambda: b.align(A, B, pd)))
elif sec == "mk_step":
    A, B = manakov(100000)
    b = nwgrad.SeqPairBatchDouble(threads, gap_model="affine", mode="local", grad_mode="none")
    b.add_many(A, B, pd)
    b.score_and_grad()
    def step():
        b.set_params(pd); b.score_and_grad(); b.scores()
    print(timed(step))
elif sec in ("prot_align", "prot_align64"):
    A, B = prot(400, 300, 1000)
    cls = nwgrad.SeqPairBatchDouble if sec.endswith("64") else nwgrad.SeqPairBatch
    b = cls(threads, gap_model="affine", mode="global", grad_mode="none")
    print(timed(lambda: b.align(A, B, pp)))
elif sec == "conv":
    A, B = prot(4000, 100, 400)
    def run():
        for a, x in zip(A, B): nwgrad.nw_score_affine(a, x, pp)
    print(timed(run))
'''

SECTIONS = ["mk_align", "mk_step", "mk_align_f32", "prot_align", "prot_align64", "conv"]


def run(py, sec, threads, data):
    out = subprocess.run([py, "-c", CHILD, sec, str(threads), data], capture_output=True, text=True)
    if out.returncode:
        raise RuntimeError(out.stderr[-2000:])
    return float(out.stdout.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--data", required=True)
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
                t["a"].append(run(args.a, sec, thr, args.data))
                if args.control: t["a2"].append(run(args.a, sec, thr, args.data))
                t["b"].append(run(args.b, sec, thr, args.data))
            a, b = min(t["a"]), min(t["b"])
            a2 = min(t["a2"]) if t["a2"] else float("nan")
            rows.append(dict(section=sec, threads=thr, a=a, a2=a2, b=b))
            print(f"{sec:14s} {thr:3d} {a:9.4f} {a2:9.4f} {b:9.4f}  {b / a:5.2f} {a2 / a:5.2f}",
                  flush=True)
    if args.json:
        json.dump(rows, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
