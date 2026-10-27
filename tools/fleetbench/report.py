#!/usr/bin/env python3
"""results/*.jsonl -> preliminary-results.md: progress, ETA, and per-host tables.

    report.py --results results/ --out preliminary-results.md [--state JSON]

Mcell/s = DP cells / best pass.  In a table, each tool shows its best arm that ran to the
end with every score correct; (x) = nwgrad default's speed / the tool's, so above 1 nwgrad
is faster.  Markers: '-' no implementation for that cell, '>10x' stopped at 10x nwgrad's
time, 'wrong' scores disagreed with the reference, 'err' crashed or failed.
"""
import argparse
import glob
import json
import os
import time
from collections import defaultdict

TOOLS = ["parasail", "seqan3", "seqan2", "ssw", "ksw2", "wfa2", "edlib", "biopython",
         "scikit-bio", "pyopal", "deepblast"]
NW_FAMILY = ["nwgrad", "nwgrad-clang", "nwgrad-cpp", "nwgrad-cpp-clang"]


def load(results):
    hosts = {}
    for path in sorted(glob.glob(os.path.join(results, "*.jsonl"))):
        host = os.path.basename(path)[:-6]
        if host.startswith("ab_"):
            continue                       # A/B runs (ab_pr.py) have their own format
        recs = {}
        for line in open(path):
            try:
                r = json.loads(line)
                recs[r["key"]] = r
            except Exception:
                pass
        plan = None
        pp = os.path.join(results, host + ".plan.json")
        if os.path.exists(pp):
            plan = json.load(open(pp))
        hosts[host] = dict(recs=recs, plan=plan,
                           done=os.path.exists(os.path.join(results, host + ".done")))
    return hosts


def eta(h, hosts=None):
    """Seconds left: each remaining arm costs what the SAME arm (same cell, same tool and
    variant) took on the other hosts, scaled by this host's speed relative to them (median
    wall ratio over the arms both have run).  Falls back to this host's recent mean."""
    plan, recs = h["plan"], h["recs"]
    if not plan or "plan" not in plan:
        return None, 0, 0
    others = [o for o in (hosts or {}).values() if o is not h]
    def strip(k):                      # key without the thread label (differs per host)
        p = k.split("|")
        return "|".join(p[:6] + ([p[6] if p[0] == "st" else "mt"]) + p[7:])
    other_wall = defaultdict(list)
    for o in others:
        for k, r in o["recs"].items():
            if r.get("wall"): other_wall[strip(k)].append(r["wall"])
    ratios = []
    for k, r in recs.items():
        ow = other_wall.get(strip(k))
        if ow and r.get("wall") and r["wall"] > 0.05:
            ratios.append(r["wall"] / (sorted(ow)[len(ow) // 2]))
    factor = sorted(ratios)[len(ratios) // 2] if len(ratios) > 50 else 1.0
    recent = [r.get("wall") or 0 for r in sorted(recs.values(), key=lambda r: r.get("t_end", 0))[-400:]]
    fallback = sum(recent) / len(recent) if recent else 5.0
    left = 0.0
    done_n = total = 0
    for ck, arms in plan["plan"]:
        d0 = recs.get(f"{ck}|{arms[0]}")
        if d0 is not None and d0["status"] != "ok":
            done_n += 1; total += 1
            continue
        for arm in arms:
            total += 1
            k = f"{ck}|{arm}"
            if k in recs:
                done_n += 1
                continue
            ow = other_wall.get(strip(k))
            left += factor * (sorted(ow)[len(ow) // 2]) if ow else fallback
    return left, done_n, total


def fmt_dur(s):
    if s is None: return "?"
    h, m = divmod(int(s) // 60, 60)
    return f"{h} h {m:02d} min" if h else f"{m} min"


def cell_entry(rs, nw):
    """best ok arm among records rs of one tool; else the most telling failure"""
    ok = [r for r in rs if r["status"] == "ok" and r.get("mcells_s")]
    if ok:
        b = max(ok, key=lambda r: r["mcells_s"])
        v = b["mcells_s"]
        sat = " sat" if (b.get("info") or {}).get("sat") in (True, "true") and b["tool"] in ("parasail", "pyopal", "ssw") else ""
        ratio = f" ({nw / v:.2f}x)" if nw else ""
        return f"{v:.0f}{ratio}{sat}"
    st = {r["status"] for r in rs}
    if "timeout" in st: return ">10x"
    if "wrong" in st: return "wrong"
    if "error" in st: return "err"
    return "-"


def host_tables(host, h):
    recs = h["recs"]
    by_cell = defaultdict(list)
    for r in recs.values():
        ck = (r["phase"], r["set"], r["tlabel"], r["mode"], r["gap"], r["out"], r["dtype"])
        by_cell[ck].append(r)
    out = []
    sets = defaultdict(list)
    for ck in by_cell:
        sets[(ck[0], ck[1], ck[2])].append(ck)
    order = {"st": 0, "mt": 1}
    for (phase, s, tl) in sorted(sets, key=lambda k: (order[k[0]], k[1], k[2])):
        cks = sorted(sets[(phase, s, tl)], key=lambda k: (k[3], k[4], ["score", "path", "hard", "soft"].index(k[5]),
                                                           ["int16", "int32", "int64", "float32", "float64"].index(k[6])))
        tools_here = [t for t in TOOLS if any(r["tool"] == t for ck in cks for r in by_cell[ck])]
        out.append(f"\n#### {host} · {s} · {'1 thread' if tl == '1' else tl + ' threads'}\n")
        head = ["cell", "**nwgrad (default)**", "best nwgrad variant", "clang", "C++"] + tools_here
        out.append("| " + " | ".join(head) + " |")
        out.append("|" + "---|" * len(head))
        for ck in cks:
            rs = by_cell[ck]
            d = [r for r in rs if r["tool"] == "nwgrad" and r["variant"] == "default"]
            if not d: continue
            d = d[0]
            label = f"{ck[3]} {ck[4]} {ck[5]} {ck[6]}"
            if d["status"] != "ok":
                why = (d.get("why") or d["status"]).split("\n")[0][:60]
                out.append(f"| {label} | n/a: {why} |" + " |" * (len(head) - 2))
                continue
            nw = d.get("mcells_s")
            var = [r for r in rs if r["tool"] == "nwgrad" and r["variant"] != "default" and r["status"] == "ok" and r.get("mcells_s")]
            bv = max(var, key=lambda r: r["mcells_s"]) if var else None
            bvs = f"{bv['variant']}: {bv['mcells_s']:.0f}" if bv else "-"
            row = [label, f"**{nw:.0f}**", bvs,
                   cell_entry([r for r in rs if r["tool"] == "nwgrad-clang"], nw),
                   cell_entry([r for r in rs if r["tool"] == "nwgrad-cpp"], nw)]
            for t in tools_here:
                row.append(cell_entry([r for r in rs if r["tool"] == t], nw))
            noisy = any((r.get("load_after") or 0) > (r.get("threads") or 1) + 1.5 for r in rs)   # the runner itself adds ~1
            if noisy: row[0] += " (load!)"
            out.append("| " + " | ".join(row) + " |")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--state", default="{}")
    a = ap.parse_args()
    state = json.loads(a.state)
    hosts = load(a.results)
    L = ["# nwgrad fleet benchmark: preliminary results", "",
         f"Updated {time.strftime('%Y-%m-%d %H:%M %Z')}. Not committed; regenerated by "
         "`bench/report.py` on every coordinator pass.", "",
         "Mcell/s; each tool's best arm whose scores all matched nwgrad's float64 reference. "
         "(x) = nwgrad default / tool: above 1, nwgrad is faster. `-` no implementation, "
         "`>10x` stopped at 10x nwgrad's time, `wrong` scores disagreed, `err` failed. "
         "`(load!)` = load average above the threads in use + 1.5 (the runner itself adds about 1) during some arm.", "",
         "## Progress", "", "| host | state | arms done | remaining (est.) | last result |",
         "|---|---|---|---|---|"]
    etas = []
    for host in sorted(set(hosts) | set(state)):
        h = hosts.get(host)
        if not h:
            L.append(f"| {host} | {state.get(host, '?')} | - | - | - |")
            continue
        rem, done_n, total = eta(h, hosts)
        last = max((r.get("t_end", 0) for r in h["recs"].values()), default=0)
        st = "finished" if h["done"] else state.get(host, "?")
        if not h["done"] and rem is not None: etas.append(rem)
        L.append(f"| {host} | {st} | {done_n} / {total} | {'0' if h['done'] else fmt_dur(rem)} | "
                 f"{time.strftime('%H:%M', time.localtime(last)) if last else '-'} |")
    if etas:
        L += ["", f"**Estimated time remaining: {fmt_dur(max(etas))}** (slowest host; each "
              "remaining arm priced at what the same arm took elsewhere, scaled by this "
              "host's measured speed -- arms no host has run yet use the host's recent mean)."]
    for host in sorted(hosts):
        p = hosts[host]["plan"] or {}
        L += ["", f"## {host}", "",
              f"nwgrad {p.get('compiler', '?')}, dispatching {p.get('auto_level', '?')} "
              f"(levels {', '.join(p.get('levels', []))}); best of {p.get('reps', '?')} passes."]
        L += host_tables(host, hosts[host])
    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        f.write("\n".join(L) + "\n")
    os.replace(tmp, a.out)


if __name__ == "__main__":
    main()
