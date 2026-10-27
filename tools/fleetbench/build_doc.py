#!/usr/bin/env python3
"""Fill benchmarks_skeleton.md's <!-- GEN:name --> markers -> docs/benchmarks.md.

    build_doc.py --results results/ --inshort inshort.md --commit 23ca19b --out docs/benchmarks.md

The hand-written conclusions (--inshort) and the skeleton's prose stay hand-written;
every number in a table comes from the results.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import doc_tables as T  # noqa: E402
from report import load  # noqa: E402


def hosts_table(H):
    out = ["| host | CPU, ISA nwgrad dispatched | OS | compilers | passes | threads (mt) |",
           "|---|---|---|---|---:|---|"]
    os_of = {"nighthaven": "Gentoo", "oldsolace": "Gentoo (laptop)", "solace": "Ubuntu 26.04 under WSL2 (laptop)",
             "spot": "Fedora Asahi (laptop)", "wloczykij": "Gentoo"}
    for h in T.hosts_in(H):
        p = H[h]["plan"] or {}
        mts = sorted({r["tlabel"] for r in H[h]["recs"].values() if r["phase"] == "mt"})
        out.append(f"| **{h}** | {T.HOST_DESC.get(h, '')}; `{p.get('auto_level', '?')}` | {os_of.get(h, '')} | "
                   f"{p.get('compiler', '?')} (+ clang build) | {p.get('reps', '?')} | {', '.join(mts) or '-'} |")
    return out


def ab_section(results):
    import subprocess
    out = subprocess.run([sys.executable, os.path.join(HERE, "ab_report.py"), "--results", results],
                         capture_output=True, text=True).stdout
    return out.strip() or "(no A/B results)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--inshort", required=True)
    ap.add_argument("--commit", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    H = load(a.results)
    H.pop("test", None)
    gen = {
        "date": time.strftime("%Y-%m-%d"),
        "commit": a.commit,
        "inshort": open(a.inshort).read().strip(),
        "hosts": "\n".join(hosts_table(H)),
        "summary": "\n".join(T.summary(H)),
        "variants": "\n".join(T.variants(H)),
        "compilers": "\n".join(T.compilers(H)),
        "mt": "\n".join(T.mt(H)),
        "detail": "\n".join(T.detail(H)),
        "ab": ab_section(a.results),
    }
    s = open(os.path.join(HERE, "benchmarks_skeleton.md")).read()
    for k, v in gen.items():
        s = s.replace(f"<!-- GEN:{k} -->", v)
    if "<!-- GEN:" in s:
        sys.exit("unfilled marker left: " + s[s.index("<!-- GEN:"):][:40])
    with open(a.out, "w") as f:
        f.write(s)


if __name__ == "__main__":
    main()
