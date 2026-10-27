#!/usr/bin/env python3
"""Summarise ab_pr.py runs: results/ab_<host>.jsonl -> markdown on stdout.

Per host: geometric means of PR/main for each compiler, the claim-specific cell groups,
the largest gains and every regression beyond 5 %, and any arm with wrong scores.
"""
import argparse
import glob
import json
import math
import os
from collections import defaultdict


def gmean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


GROUPS = [
    ("integer score (gcc folding fix)", lambda s, m, g, o, d: o == "score" and d.startswith("int")),
    ("L100 path / hard (inter-pair hoisting)", lambda s, m, g, o, d: s in ("protein_L100", "dna_L100") and o in ("path", "hard")),
    ("int64 local linear (x86 clang cmov)", lambda s, m, g, o, d: d == "int64" and m == "local" and g == "linear"),
    ("float32 / float64", lambda s, m, g, o, d: d.startswith("float")),
    ("miRNA", lambda s, m, g, o, d: s == "mirna"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    a = ap.parse_args()
    out = []
    for path in sorted(glob.glob(os.path.join(a.results, "ab_*.jsonl"))):
        host = os.path.basename(path)[3:-6]
        best = defaultdict(dict)
        wrong = defaultdict(list)
        for line in open(path):
            r = json.loads(line)
            if r["status"] == "ok":
                best[r["cell"]][r["build"]] = max(best[r["cell"]].get(r["build"], 0), r["mcells_s"])
            elif r["status"] == "wrong":
                wrong[r["build"]].append(r["cell"])
        full = {c: v for c, v in best.items() if len(v) == 4}
        out.append(f"\n#### {host} ({len(full)} cells)\n")
        out.append("| cells | n | PR gcc / main gcc | PR clang / main clang | main clang / main gcc | PR clang / PR gcc |")
        out.append("|---|---:|---:|---:|---:|---:|")
        rows = [("all", lambda *k: True)] + GROUPS
        for name, pred in rows:
            cs = [c for c in full if pred(*c.split("|"))]
            if not cs: continue
            f = lambda a_, b_: f"{gmean([full[c][a_] / full[c][b_] for c in cs]):.3f}"
            out.append(f"| {name} | {len(cs)} | {f('pr-gcc', 'main-gcc')} | {f('pr-clang', 'main-clang')} | "
                       f"{f('main-clang', 'main-gcc')} | {f('pr-clang', 'pr-gcc')} |")
        for comp in ("gcc", "clang"):
            r = sorted(((full[c][f"pr-{comp}"] / full[c][f"main-{comp}"], c) for c in full))
            top = ", ".join(f"{c.replace('|', ' ')} {x:.2f}" for x, c in r[-4:][::-1])
            reg = [f"{c.replace('|', ' ')} {x:.2f}" for x, c in r if x < 0.95]
            out.append(f"\n- **{comp}**, largest gains: {top}")
            out.append(f"- **{comp}**, regressions beyond 5 %: " + ("; ".join(reg) if reg else "none"))
        out.append("- wrong scores: " + (", ".join(f"{b}: {len(v)}" for b, v in wrong.items()) if wrong else "none"))
    print("\n".join(out))


if __name__ == "__main__":
    main()
