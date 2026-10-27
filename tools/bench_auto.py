#!/usr/bin/env python3
"""Which traceback / hb_cutoff should `auto` pick?  One build, several arms.

  python tools/bench_auto.py --py PYTHON --arms pointers pointers@one hirschberg:512 hirschberg_pmax:512 ...
                --lens 1000 2000 --threads 1 6 12 --precs double float int32 --modes global local
                --fixtures unrel homol --rounds 2

An arm is traceback[:hb_cutoff][@one|@dense] (@: NWGRAD_DENSE_CODES=0 / 1 in that process).
Each (case, arm) in a fresh process, arms alternating (order reversed every round); best
over rounds.  Reports wall ns per cell per thread and each arm's time / the first arm's.
Fixtures: unrel = random proteome pairs within +-10 % of L; homol = a proteome sequence
and a copy with 30 % substitutions plus a few indels; proteins concatenated past the
file's lengths.  BLOSUM62, 11/1, hard gradient.
"""
import argparse, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = "GCF_000001405.40_GRCh38.p14_protein.faa"   # the human proteome, as tools/bench_local_hb.py

CHILD = r'''
import json, os, sys, time, warnings
warnings.simplefilter("ignore")
if sys.argv[8].endswith("@one"): os.environ["NWGRAD_DENSE_CODES"] = "0"   # one-byte codes forced
if sys.argv[8].endswith("@dense"): os.environ["NWGRAD_DENSE_CODES"] = "1"
import numpy as np, nwgrad
from nwgrad.matrices import BLOSUM62
prec, mode, L, th, gcells, reps, data, arm, fix = sys.argv[1:10]
L, th, gcells, reps = int(L), int(th), float(gcells), int(reps)
tb, _, cut = arm.split("@")[0].partition(":")
def _load_seqs(path):
    # one sequence a line, or FASTA (e.g. GCF_000001405.40_GRCh38.p14_protein.faa); the
    # BLOSUM62 alphabet, U -> C, O -> X, anything else -> X; 60-3000 residues, distinct
    ok = set("ARNDCQEGHILKMFPSTWYVBZX*")
    out, seen, cur, fasta = [], set(), [], False
    def add(s):
        s = s.upper().replace("U", "C").replace("O", "X")
        s = "".join(c if c in ok else "X" for c in s)
        if 60 <= len(s) <= 3000 and s not in seen: seen.add(s); out.append(s)
    for line in open(path):
        if line.startswith(">"):
            fasta = True
            if cur: add("".join(cur)); cur = []
        elif fasta: cur.append(line.strip())
        elif line.strip(): add(line.strip())
    if cur: add("".join(cur))
    return out
seqs = _load_seqs(data)
pool = [s for s in seqs if int(0.9 * L) <= len(s) <= int(1.1 * L) + 1]
rng = np.random.default_rng(L)
if len(pool) < 50:   # longer than the proteome file holds: real proteins, concatenated
    pool = []
    while len(pool) < 200:
        x = ""
        while len(x) < 0.9 * L: x += seqs[rng.integers(len(seqs))]
        pool.append(x[:int(rng.integers(int(0.9 * L), int(1.1 * L) + 1))])
npairs = max(2 * th, int(gcells * 1e9 / (L * L)))
AL = "ACDEFGHIKLMNPQRSTVWY"
A, B = [], []
for _ in range(npairs):
    a = pool[rng.integers(len(pool))]
    if fix == "unrel":
        b = pool[rng.integers(len(pool))]
    else:
        y = list(a)
        for k in rng.choice(len(y), int(0.3 * len(y)), replace=False): y[k] = AL[rng.integers(20)]
        for _ in range(max(1, len(y) // 200)):
            k = int(rng.integers(len(y)))
            if rng.random() < 0.5: y[k:k] = [AL[i] for i in rng.integers(0, 20, int(rng.integers(1, 12)))]
            else: del y[k:k + int(rng.integers(1, 12))]
        b = "".join(y)
    A.append(a); B.append(b)
cells = float(sum((len(a) + 1) * (len(b) + 1) for a, b in zip(A, B)))
cls = {"double": nwgrad.SeqPairBatchDouble, "float": nwgrad.SeqPairBatch,
       "int32": nwgrad.SeqPairBatchInt32, "int64": nwgrad.SeqPairBatchInt64}[prec]
p = nwgrad.AlignParams(BLOSUM62, 11.0, 1.0, 11.0, 1.0)
b = cls(A, B, p, n_threads=th, traceback=tb, gap_model="affine", mode=mode, grad_mode="hard")
if cut: b.hb_cutoff = int(cut)
b.score_and_grad()
best = 1e30
for _ in range(reps):
    t = time.perf_counter(); s = b.score_and_grad(); best = min(best, time.perf_counter() - t)
print(json.dumps({"ns": best / cells * 1e9 * th, "sum": float(s), "isa": nwgrad.simd_isa()}))
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--py", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--lens", type=int, nargs="+", default=[1000, 2000])
    ap.add_argument("--threads", type=int, nargs="+", default=[1])
    ap.add_argument("--precs", nargs="+", default=["double"])
    ap.add_argument("--modes", nargs="+", default=["local"])
    ap.add_argument("--fixtures", nargs="+", default=["unrel"])
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--gcells", type=float, default=0.3)
    ap.add_argument("--json", default=None)
    ap.add_argument("--data", default=DATA, help="proteins: FASTA or one a line")
    a = ap.parse_args()
    out = []
    print("thread-ns/cell; (x) = arm / first arm; * = fastest", flush=True)
    for fix in a.fixtures:
        for prec in a.precs:
            for mode in a.modes:
                for th in a.threads:
                    for L in a.lens:
                        res = {x: [] for x in a.arms}
                        for r in range(a.rounds):
                            for arm in (a.arms if r % 2 == 0 else a.arms[::-1]):
                                cmd = [a.py, "-I", "-c", CHILD, prec, mode, str(L), str(th),
                                       str(a.gcells * th), str(a.reps), os.path.abspath(a.data), arm, fix]
                                o = subprocess.run(cmd, capture_output=True, text=True, cwd="/tmp")
                                if o.returncode:
                                    res[arm].append(float("nan")); print(o.stderr[-300:], file=sys.stderr); continue
                                res[arm].append(json.loads(o.stdout.strip().splitlines()[-1])["ns"])
                        best = {k: min(v) for k, v in res.items()}
                        b0 = best[a.arms[0]]; fastest = min(best, key=best.get)
                        cells = "  ".join(f"{k} {v:.3f}" + ("" if k == a.arms[0] else f" ({v / b0:.2f})")
                                          + ("*" if k == fastest else "") for k, v in best.items())
                        print(f"{fix:5s} {prec:6s} {mode:6s} t{th:<3d} L{L:<5d} {cells}", flush=True)
                        out.append(dict(fix=fix, prec=prec, mode=mode, th=th, L=L, res=res))
    if a.json:
        json.dump(out, open(a.json, "w"))


if __name__ == "__main__":
    main()
