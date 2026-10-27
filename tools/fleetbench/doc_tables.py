#!/usr/bin/env python3
"""results/*.jsonl -> the generated tables of docs/benchmarks.md (printed as markdown).

    doc_tables.py --results results/ [--section summary|variants|compilers|mt|detail]

summary    per host and use-case: each competitor's geometric-mean speed ratio
           (nwgrad default / tool) over every single-thread cell both completed
variants   nwgrad's non-default variants against the default, per host
compilers  clang vs gcc build and C++ vs Python API, per host and dtype
mt         the multithread phase: Mcell/s per tool, native threads vs harness-parallel
detail     every cell, one collapsible block per host and fixture set
"""
import argparse
import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report import load, cell_entry  # noqa: E402

HOST_ORDER = ["nighthaven", "oldsolace", "solace", "spot", "wloczykij"]
HOST_DESC = {
    "nighthaven": "i5-12500 (Alder Lake), AVX2 — x86-64-v3, the primary target",
    "oldsolace": "i7-7700HQ (Kaby Lake), AVX2 — x86-64-v3",
    "solace": "i7-11850H (Tiger Lake), AVX-512 — under WSL2",
    "spot": "Apple M1, NEON — Fedora Asahi",
    "wloczykij": "4× Opteron 6380 (Piledriver), SSE4 — 64 cores, 8 NUMA nodes",
}
COMPETITORS = ["parasail", "seqan3", "seqan2", "ssw", "ksw2", "wfa2", "edlib", "biopython",
               "scikit-bio", "pyopal", "deepblast"]
OUTS = [("score", "Score only"), ("path", "Score + alignment"),
        ("hard", "Hard gradient"), ("soft", "Soft gradient")]
DTYPES = ["int16", "int32", "int64", "float32", "float64"]


def gmean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def cells(recs):
    by = defaultdict(list)
    for r in recs.values():
        by[(r["phase"], r["set"], r["tlabel"], r["mode"], r["gap"], r["out"], r["dtype"])].append(r)
    return by


def default_of(rs):
    d = [r for r in rs if r["tool"] == "nwgrad" and r["variant"] == "default"]
    return d[0] if d and d[0]["status"] == "ok" and d[0].get("mcells_s") else None


def best_ok(rs):
    ok = [r for r in rs if r["status"] == "ok" and r.get("mcells_s")]
    return max(ok, key=lambda r: r["mcells_s"]) if ok else None


def hosts_in(H):
    return [h for h in HOST_ORDER if h in H] + [h for h in H if h not in HOST_ORDER and h != "test"]


def summary(H):
    out = []
    for o, title in OUTS:
        rows = []
        tools_seen = set()
        for host in hosts_in(H):
            by = cells(H[host]["recs"])
            per = defaultdict(lambda: defaultdict(list))   # tool -> dtype -> ratios
            slow = defaultdict(lambda: defaultdict(int))
            for ck, rs in by.items():
                if ck[0] != "st" or ck[5] != o: continue
                d = default_of(rs)
                if not d: continue
                for t in COMPETITORS:
                    trs = [r for r in rs if r["tool"] == t]
                    if not trs: continue
                    b = best_ok(trs)
                    if b:
                        per[t][ck[6]].append(d["mcells_s"] / b["mcells_s"]); tools_seen.add(t)
                    elif any(r["status"] == "timeout" for r in trs):
                        slow[t][ck[6]] += 1; tools_seen.add(t)
            rows.append((host, per, slow))
        tools = [t for t in COMPETITORS if t in tools_seen]
        if not tools: continue
        out.append(f"\n#### {title}\n")
        out.append("Geometric mean over the single-thread cells (fixture sets × global/local × "
                   "affine/linear) that both nwgrad and the tool completed with correct scores; "
                   "above 1, nwgrad is faster. `+k` = k more cells where the tool was stopped at "
                   "10× nwgrad's time. `n` = cells in the mean.\n")
        out.append("| host | dtype | " + " | ".join(tools) + " |")
        out.append("|---|---|" + "---|" * len(tools))
        for host, per, slow in rows:
            for dt in DTYPES:
                vals = []
                any_ = False
                for t in tools:
                    xs = per[t].get(dt, [])
                    s = slow[t].get(dt, 0)
                    if xs or s: any_ = True
                    if not xs and not s: vals.append("–"); continue
                    g = gmean(xs)
                    cell = f"{g:.2f}x (n={len(xs)})" if g else ""
                    if s: cell += f" +{s}"
                    vals.append(cell.strip())
                if any_:
                    out.append(f"| {host} | {dt} | " + " | ".join(vals) + " |")
    return out


def variants(H):
    out = ["| host | variant | dtype | cells | variant / default, geometric mean | range | faster in |",
           "|---|---|---|---:|---:|---|---:|"]
    for host in hosts_in(H):
        by = cells(H[host]["recs"])
        acc = defaultdict(list)
        for ck, rs in by.items():
            if ck[0] != "st": continue
            d = default_of(rs)
            if not d: continue
            for r in rs:
                if r["tool"] != "nwgrad" or r["variant"] == "default": continue
                if r["status"] == "ok" and r.get("mcells_s"):
                    acc[(r["variant"], ck[6])].append(r["mcells_s"] / d["mcells_s"])
                elif r["status"] == "timeout":
                    acc[(r["variant"], ck[6])].append(0.1)          # at most 1/10
        for (v, dt) in sorted(acc):
            xs = acc[(v, dt)]
            wins = sum(1 for x in xs if x > 1.05)
            out.append(f"| {host} | `{v}` | {dt} | {len(xs)} | {gmean(xs):.2f} | "
                       f"{min(xs):.2f}–{max(xs):.2f} | {wins} |")
    return out


def compilers(H):
    out = ["| host | dtype | clang / gcc (Python) | C++ API / Python (gcc) | C++ clang / C++ gcc | cells |",
           "|---|---|---:|---:|---:|---:|"]
    for host in hosts_in(H):
        by = cells(H[host]["recs"])
        acc = defaultdict(lambda: defaultdict(list))
        for ck, rs in by.items():
            if ck[0] != "st": continue
            d = default_of(rs)
            if not d: continue
            get = {r["tool"]: r for r in rs if r["variant"] == "default" and r["status"] == "ok" and r.get("mcells_s")}
            if "nwgrad-clang" in get: acc[ck[6]]["clang"].append(get["nwgrad-clang"]["mcells_s"] / d["mcells_s"])
            if "nwgrad-cpp" in get: acc[ck[6]]["cpp"].append(get["nwgrad-cpp"]["mcells_s"] / d["mcells_s"])
            if "nwgrad-cpp" in get and "nwgrad-cpp-clang" in get:
                acc[ck[6]]["cppc"].append(get["nwgrad-cpp-clang"]["mcells_s"] / get["nwgrad-cpp"]["mcells_s"])
        for dt in DTYPES:
            a = acc.get(dt)
            if not a: continue
            f = lambda k: f"{gmean(a[k]):.2f}" if a[k] else "–"
            out.append(f"| {host} | {dt} | {f('clang')} | {f('cpp')} | {f('cppc')} | {len(a['clang'])} |")
    return out


def mt(H):
    """Per host and thread count: nwgrad's speed-up over its own single thread, and each
    competitor's geometric-mean ratio (nwgrad / tool) over the multithread cells."""
    out = ["nwgrad's speed-up is its multithread Mcell/s over its single-thread Mcell/s on the "
           "same fixture (the multithread sets are 10x larger). Ratios as above (nwgrad / tool); "
           "‡ = harness-parallel (the tool has no threads of its own; the pairs were split over "
           "threads or processes by the harness), otherwise the tool's own threading.", "",
           "| host | threads | dtype | nwgrad speed-up | " + " | ".join(COMPETITORS[:-1]) + " |",
           "|---|---|---|---|" + "---|" * (len(COMPETITORS) - 1)]
    for host in hosts_in(H):
        by = cells(H[host]["recs"])
        for tl in sorted({ck[2] for ck in by if ck[0] == "mt"}, key=lambda x: -int(x.split(":")[0])):
            for dt in ["int32", "float32", "float64"]:
                sp = []
                per = defaultdict(list); slow = defaultdict(int); hp = defaultdict(bool)
                for ck, rs in by.items():
                    if ck[0] != "mt" or ck[2] != tl or ck[6] != dt: continue
                    d = default_of(rs)
                    if not d: continue
                    st = by.get(("st", ck[1], "1") + ck[3:])
                    sd = default_of(st) if st else None
                    if sd: sp.append(d["mcells_s"] / sd["mcells_s"])
                    for t in COMPETITORS:
                        trs = [r for r in rs if r["tool"] == t]
                        b = best_ok(trs)
                        if b:
                            per[t].append(d["mcells_s"] / b["mcells_s"])
                            hp[t] |= (b.get("info") or {}).get("harness_parallel") in (True, "true")
                        elif any(r["status"] == "timeout" for r in trs):
                            slow[t] += 1
                if not sp: continue
                vals = []
                for t in COMPETITORS[:-1]:
                    if not per[t] and not slow[t]: vals.append("–"); continue
                    g = gmean(per[t])
                    v = (f"{g:.2f}x" if g else "") + (" ‡" if hp[t] else "") + (f" +{slow[t]}" if slow[t] else "")
                    vals.append(v.strip())
                out.append(f"| {host} | {tl} | {dt} | {gmean(sp):.1f}x ({min(sp):.1f}–{max(sp):.1f}) | " + " | ".join(vals) + " |")
    return out


def detail(H):
    from report import host_tables
    out = []
    for host in hosts_in(H):
        out.append(f"\n<details><summary><b>{host}</b> — {HOST_DESC.get(host, '')}: every cell</summary>\n")
        out += host_tables(host, H[host])
        out.append("\n</details>")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--section", default="summary",
                    choices=["summary", "variants", "compilers", "mt", "detail"])
    a = ap.parse_args()
    H = load(a.results)
    H.pop("test", None)
    print("\n".join({"summary": summary, "variants": variants, "compilers": compilers,
                     "mt": mt, "detail": detail}[a.section](H)))


if __name__ == "__main__":
    main()
