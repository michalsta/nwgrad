#!/usr/bin/env python3
"""A/B any fill path between nwgrad builds (or settings), alternating processes.

  python tools/bench_paths.py --py A=/path/to/python B=/path/to/python [ctl=/path/to/python]
         --lens 500 2000 --threads 1 6 12 --precs double float int32 --modes local global
         [--tb pointers|scores|hirschberg|hirschberg_pmax] [--gm affine|linear]
         [--gd hard|none] [--cut ROWS] [--exact] [--data proteins.faa]

Each (arm, case) runs in a fresh process, the arms alternating (reversed every round);
best ns/cell over rounds, and every arm's ratio to the first.  An arm is a python
executable, or a wrapper script that sets an environment (e.g. NWGRAD_DENSE_CODES=0/1)
and execs one, so one build can A/B its own settings; repeat the first arm as a control.
Pairs: random proteome pairs within +-10 % of L (proteins concatenated past the file's
lengths; --exact: exactly L x L), BLOSUM62 11/1 (integral: every scalar type runs),
fill="striped".  ns/cell is thread-seconds per cell.
"""
import argparse, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = "GCF_000001405.40_GRCh38.p14_protein.faa"   # the human proteome, as tools/bench_local_hb.py

CHILD = r'''
import json, sys, time, os, warnings
warnings.simplefilter("ignore")
import numpy as np, nwgrad
from nwgrad.matrices import BLOSUM62
prec, mode, L, th, gcells, reps, data, isa = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6]), sys.argv[7], sys.argv[8]
exact = sys.argv[9] == "exact"
tb, gm, gd, cut = sys.argv[10], sys.argv[11], sys.argv[12], int(sys.argv[13])
if isa != "-": nwgrad.set_isa_level(isa)
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
lo, hi = int(0.9 * L), int(1.1 * L) + 1
pool = [s for s in seqs if lo <= len(s) <= hi]
rng = np.random.default_rng(L)
if len(pool) < 50:   # longer than the proteome file holds: real proteins, concatenated
    pool = []
    while len(pool) < 200:
        x = ""
        while len(x) < lo: x += seqs[rng.integers(len(seqs))]
        pool.append(x[:int(rng.integers(lo, hi))])
npairs = max(4 * th, int(gcells * 1e9 / (L * L)))
ia = rng.integers(0, len(pool), npairs); ib = rng.integers(0, len(pool), npairs)
A = [pool[i] for i in ia]; B = [pool[i] for i in ib]
if exact:   # exactly L x L: real residues, cut / tiled to length L
    A = [(a * (L // len(a) + 1))[:L] for a in A]; B = [(b * (L // len(b) + 1))[:L] for b in B]
cells = float(sum((len(a) + 1) * (len(b) + 1) for a, b in zip(A, B)))
cls = {"double": nwgrad.SeqPairBatchDouble, "float": nwgrad.SeqPairBatch,
       "int32": nwgrad.SeqPairBatchInt32, "int16": nwgrad.SeqPairBatchInt16,
       "int64": nwgrad.SeqPairBatchInt64}[prec]
p = nwgrad.AlignParams(BLOSUM62, 11.0, 1.0, 11.0, 1.0)
if gm == "linear": p = nwgrad.AlignParams(BLOSUM62, 0.0, 1.0, 0.0, 1.0)
b = cls(A, B, p, n_threads=th, traceback=tb, gap_model=gm, mode=mode, grad_mode=gd)
b.fill = "striped"
if cut: b.hb_cutoff = cut
b.score_and_grad()
best = 1e30
for _ in range(reps):
    t = time.perf_counter(); s = b.score_and_grad(); dt = time.perf_counter() - t
    best = min(best, dt)
print(json.dumps({"ns": best / cells * 1e9 * th, "wall": best, "sum": float(s), "n": npairs,
                  "isa": nwgrad.simd_isa(), "cw": str(nwgrad.compiled_with()), "file": nwgrad.__file__}))
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--py", nargs="+", required=True, help="NAME=python")
    ap.add_argument("--lens", type=int, nargs="+", default=[100, 200, 500, 1000, 1500, 2000])
    ap.add_argument("--threads", type=int, nargs="+", default=[1])
    ap.add_argument("--precs", nargs="+", default=["double"])
    ap.add_argument("--modes", nargs="+", default=["local"])
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--gcells", type=float, default=0.5, help="Gcells per pass (per thread x this)")
    ap.add_argument("--isa", default="-")
    ap.add_argument("--json", default=None)
    ap.add_argument("--data", default=DATA, help="proteins: FASTA or one a line")
    ap.add_argument("--exact", action="store_true", help="every pair exactly L x L")
    ap.add_argument("--tb", default="pointers"); ap.add_argument("--gm", default="affine")
    ap.add_argument("--gd", default="hard"); ap.add_argument("--cut", type=int, default=0)
    a = ap.parse_args()
    arms = [x.split("=", 1) for x in a.py]
    out = []
    print("ns/cell is thread-seconds per cell (wall*threads/cells); ratio = arm/first arm", flush=True)
    for prec in a.precs:
        for mode in a.modes:
            for th in a.threads:
                for L in a.lens:
                    res = {n: [] for n, _ in arms}
                    sums = {}
                    for r in range(a.rounds):
                        order = arms if r % 2 == 0 else arms[::-1]
                        for name, py in order:
                            cmd = [py, "-I", "-c", CHILD, prec, mode, str(L), str(th),
                                   str(a.gcells * th), str(a.reps), os.path.abspath(a.data), a.isa, "exact" if a.exact else "-", a.tb, a.gm, a.gd, str(a.cut)]
                            o = subprocess.run(cmd, capture_output=True, text=True, cwd="/tmp")
                            if o.returncode:
                                print(o.stderr, file=sys.stderr); sys.exit(1)
                            d = json.loads(o.stdout.strip().splitlines()[-1])
                            res[name].append(d["ns"]); sums[name] = d["sum"]
                    if len(set(sums.values())) != 1:
                        print("  *** score sums differ:", sums)
                    base = min(res[arms[0][0]])
                    cells = "  ".join(f"{n} {min(v):.4f}" + ("" if n == arms[0][0] else f" ({min(v)/base:.3f})")
                                      for n, v in res.items())
                    print(f"{a.tb}/{a.gm}/{a.gd}{'/c'+str(a.cut) if a.cut else ''} {prec:6s} {mode:6s} t{th:<3d} L{L:<5d} {cells}   [{d['isa']}]", flush=True)
                    out.append(dict(prec=prec, mode=mode, th=th, L=L, res=res))
    if a.json:
        json.dump(out, open(a.json, "w"))


if __name__ == "__main__":
    main()
