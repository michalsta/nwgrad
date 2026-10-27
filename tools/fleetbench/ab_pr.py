#!/usr/bin/env python3
"""A/B: nwgrad builds against each other on the same cells, interleaved.

    ab_pr.py --root ~/nwbench --host spot --builds main-gcc=venv-gcc main-clang=venv-clang \
             pr-gcc=venv-pr12-gcc pr-clang=venv-pr12-clang --rounds 3 --reps 3

Every cell runs every build once per round, in an order rotated each round, so a drift in
the machine's state hits all builds alike; a build's figure is its best pass over all rounds.
Each run's per-pair scores are checked against the float64 reference like the main run's.
Appends to results/ab_<host>.jsonl and resumes from it.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from runner import reference, read_scores, check, loadavg  # noqa: E402
from problems import kind_of  # noqa: E402

SETS = ["protein_L100", "protein_L300", "dna_L100", "mirna"]
DTYPES = ["int16", "int32", "int64", "float32", "float64"]


def cells():
    for s in SETS:
        for mode in ("global", "local"):
            for gap in ("affine", "linear"):
                for out in ("score", "path", "hard", "soft"):
                    for dt in DTYPES:
                        if out == "soft" and dt != "float64":
                            continue
                        yield dict(phase="st", set=s, kind=kind_of(s), mode=mode, gap=gap, out=out, dtype=dt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--host", required=True)
    ap.add_argument("--builds", nargs="+", required=True, help="label=venvdir")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    root = os.path.abspath(os.path.expanduser(a.root))
    builds = [b.split("=", 1) for b in a.builds]
    env = dict(work=f"{root}/work", py_gcc=f"{root}/venv-gcc/bin/python")
    out = f"{root}/results/ab_{a.host}.jsonl"
    done = set()
    if os.path.exists(out):
        for l in open(out):
            try: done.add(json.loads(l)["key"])
            except Exception: pass
    f = open(out, "a", buffering=1)
    sc_path = f"{root}/work/ab_scores.txt"
    for c in cells():
        ck = f"{c['set']}|{c['mode']}|{c['gap']}|{c['out']}|{c['dtype']}"
        fixture = f"{root}/fixtures/{c['set']}_st.tsv"
        ref = None if c["out"] == "soft" else reference(env, fixture, c)
        for rnd in range(a.rounds):
            order = builds[rnd % len(builds):] + builds[:rnd % len(builds)]
            for label, venv in order:
                key = f"{ck}|{label}|{rnd}"
                if key in done: continue
                if os.path.exists(sc_path): os.remove(sc_path)
                cmd = [f"{root}/{venv}/bin/python", f"{HERE}/arm.py", "--tool", "nwgrad",
                       "--fixture", fixture, "--kind", c["kind"], "--gap", c["gap"],
                       "--mode", c["mode"], "--out", c["out"], "--dtype", c["dtype"],
                       "--threads", "1", "--reps", str(a.reps), "--scores-out", sc_path]
                l0 = loadavg(); t0 = time.time()
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
                line = [x for x in p.stdout.splitlines() if x.startswith("{")]
                res = json.loads(line[-1]) if line else {"status": "error", "why": p.stderr[-400:]}
                rec = dict(key=key, cell=ck, build=label, round=rnd, status=res.get("status"),
                           best_s=res.get("best_s"), times=res.get("times"), cells=res.get("cells"),
                           info=res.get("info"), why=res.get("why"), load_before=l0,
                           load_after=loadavg(), t_start=t0, t_end=time.time())
                if rec["status"] == "ok":
                    rec["mcells_s"] = rec["cells"] / rec["best_s"] / 1e6
                    if ref is not None:
                        ok, detail = check(read_scores(sc_path), ref, c["dtype"])
                        if not ok: rec["status"] = "wrong"; rec["why"] = detail
                f.write(json.dumps(rec) + "\n")
    open(f"{root}/results/ab_{a.host}.done", "w").write(str(time.time()))


if __name__ == "__main__":
    main()
