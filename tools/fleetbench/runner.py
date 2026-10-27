#!/usr/bin/env python3
"""Run the fleet benchmark on THIS host: plan every cell, resume from results.jsonl.

    runner.py --root ~/nwbench --host nighthaven --reps 3 --mt 6
    runner.py --root ~/nwbench --host wloczykij --reps 3 --mt 64 --mt 8:numa0
    runner.py --root ~/nwbench --plan          # print the plan, run nothing

A cell is (fixture set, mode, gap model, output, dtype, threads).  In each cell nwgrad's
DEFAULT arm runs first; its best pass T0 sets every other arm's budget: a pass longer
than 10*T0 ends the arm ("timeout").  A cell nwgrad cannot run (no such dtype for this
output, an integer overflow bound) is skipped whole.  Every arm's per-pair scores are
checked against nwgrad's float64 score-only reference; an arm that disagrees is recorded
as "wrong" and its time is not used.

One JSON object per finished arm is appended to results.jsonl, so a restart (a laptop
powered off mid-run) resumes after the last finished arm.
"""
import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from problems import problem, kind_of, write_problem_txt  # noqa: E402

DTYPES = ["int16", "int32", "int64", "float32", "float64"]
ST_SETS = ["protein_L100", "protein_L300", "protein_L1000", "protein_L3000", "protein_L10000",
           "dna_L100", "dna_L1000", "dna_L10000", "mirna"]
MT_SETS = ["protein_L100", "protein_L300", "protein_L1000", "protein_L3000",
           "dna_L100", "dna_L1000", "mirna"]
MT_DTYPES = ["int32", "float32", "float64"]
TIMEOUT_X = 10.0
WARMUP_SLACK = 3.0      # seconds on top of 10*T0 for the warm-up pass (one-time costs)


# ── the plan ─────────────────────────────────────────────────────────────────────

def nwgrad_variants(out, dtype, gap, levels, auto_level):
    """(variant, why) pairs of nwgrad's Python batch beyond the default, single thread."""
    v = []
    if dtype not in ("float32", "float64"):
        return v
    if out in ("path", "hard") and gap == "affine":
        for tb in ("pointers", "scores", "hirschberg", "hirschberg_pmax", "grid"):
            v.append(f"traceback={tb}")
    if out in ("path", "hard") and gap == "linear":
        for tb in ("pointers", "scores", "hirschberg"):
            v.append(f"traceback={tb}")
    v.append("fill=striped")
    if gap == "affine" and dtype == "float64":
        v.append("fill=rowwise")
    if out != "soft":
        for lv in ["scalar_fallback"] + [l for l in levels if l != auto_level]:
            v.append(f"kernel={lv}")
    if out == "soft":
        v.append("soft_impl=log")
    if out != "path":
        v.append("api=functions")
    return v


def plan(env):
    """Yield (cell, arms); arms[0] is nwgrad's default."""
    for phase, sets, dtypes, threads_list in (
            ("st", ST_SETS, DTYPES, [("1", 1, None)]),
            ("mt", MT_SETS, MT_DTYPES, env["mt"])):
        for s in sets:
            kind = kind_of(s)
            for mode in ("global", "local"):
                for gap in ("affine", "linear"):
                    for out in ("score", "path", "hard", "soft"):
                        for dtype in dtypes:
                            if out == "soft" and dtype != "float64":
                                continue            # nwgrad's soft path is float64 only
                            for tlabel, nthr, numa in threads_list:
                                cell = dict(phase=phase, set=s, kind=kind, mode=mode, gap=gap,
                                            out=out, dtype=dtype, threads=nthr, tlabel=tlabel,
                                            numa=numa)
                                yield cell, arms_for(cell, env)


def arms_for(c, env):
    st = c["phase"] == "st"
    out, dtype, gap, mode, kind = c["out"], c["dtype"], c["gap"], c["mode"], c["kind"]
    A = [("nwgrad", "default")]                     # always first: T0
    if st:
        A += [("nwgrad", v) for v in nwgrad_variants(out, dtype, gap, env["levels"], env["auto_level"])]
    else:
        A += [("nwgrad", "fill=striped"), ("nwgrad", "schedule=sorted")]
    if env["have"].get("clang_py"): A.append(("nwgrad-clang", "default"))
    if env["have"].get("cpp_gcc"): A.append(("nwgrad-cpp", "default"))
    if env["have"].get("cpp_clang"): A.append(("nwgrad-cpp-clang", "default"))
    if out in ("score", "path"):
        if dtype.startswith("int") and env["have"].get("parasail"):
            impls = ("striped", "scan", "diag") if st else ("striped", "scan")
            widths = {"int16": ["16", "sat"], "int32": ["32", "sat"], "int64": ["64"]}[dtype]
            for w in widths:
                for i in impls:
                    A.append(("parasail", f"impl={i},width={w}"))
        alph = "dna" if kind == "dna" else "protein"
        if env["have"].get(f"seqan3_{alph}_{dtype}_vec"): A.append(("seqan3", "vectorised"))
        if st and env["have"].get(f"seqan3_{alph}_{dtype}_scalar"): A.append(("seqan3", "scalar"))
        if env["have"].get(f"seqan2_{alph}_{dtype}"):
            A.append(("seqan2", "exec=vectorial"))
            if st: A.append(("seqan2", "exec=scalar"))
        if dtype == "int16" and mode == "local" and env["have"].get("ssw"): A.append(("ssw", "default"))
        if dtype == "int32" and mode == "global" and env["have"].get("ksw2"): A.append(("ksw2", "default"))
        if dtype == "int32" and mode == "global" and kind == "dna" and env["have"].get("wfa2"):
            mems = ("high", "med", "low", "ultralow") if st else ("high", "ultralow")
            for m in mems: A.append(("wfa2", f"memory={m}"))
            if gap == "linear": A.append(("wfa2", "metric=edit"))
        if dtype == "float64" and env["have"].get("biopython"): A.append(("biopython", "default"))
        if dtype == "float64" and out == "path" and env["have"].get("scikit-bio"): A.append(("scikit-bio", "default"))
        if dtype in ("int16", "int32") and env["have"].get("pyopal"): A.append(("pyopal", "default"))
        if dtype == "int32" and kind == "dna" and gap == "linear" and mode == "global" and env["have"].get("edlib"):
            A.append(("edlib", "default"))
    # DeepBLAST keeps whole float64 autograd tables (several GB at L=10000): not there
    if out == "soft" and mode == "global" and gap == "linear" and st and env["have"].get("deepblast") \
            and not c["set"].endswith("L10000"):
        A.append(("deepblast", "default"))
    return A


# ── running one arm ──────────────────────────────────────────────────────────────

def arm_cmd(tool, variant, c, env, fixture, prob_txt, scores_out, deadline):
    common = ["--fixture", fixture, "--gap", c["gap"], "--mode", c["mode"], "--out", c["out"],
              "--dtype", c["dtype"], "--threads", str(c["threads"]), "--reps", str(env["reps"]),
              "--deadline", f"{deadline:.6f}", "--warmup-slack", str(WARMUP_SLACK),
              "--scores-out", scores_out, "--variant", variant]
    py = [env["py_gcc"], os.path.join(HERE, "arm.py"), "--kind", c["kind"]]
    b = env["bin"]
    alph = "dna" if c["kind"] == "dna" else "protein"
    if tool == "nwgrad": cmd = py + ["--tool", "nwgrad"] + common
    elif tool == "nwgrad-clang": cmd = [env["py_clang"], os.path.join(HERE, "arm.py"), "--kind", c["kind"], "--tool", "nwgrad"] + common
    elif tool == "nwgrad-cpp": cmd = [f"{b}/nwgrad_cpp_gcc", "--problem", prob_txt] + common
    elif tool == "nwgrad-cpp-clang": cmd = [f"{b}/nwgrad_cpp_clang", "--problem", prob_txt] + common
    elif tool == "parasail": cmd = [f"{b}/parasail_drv", "--problem", prob_txt] + common
    elif tool == "seqan3":
        cmd = [f"{b}/seqan3_{alph}_{c['dtype']}_{'vec' if variant == 'vectorised' else 'scalar'}", "--problem", prob_txt] + common
    elif tool == "seqan2": cmd = [f"{b}/seqan2_{alph}_{c['dtype']}", "--problem", prob_txt] + common
    elif tool in ("ssw", "ksw2", "wfa2"): cmd = [f"{b}/{tool}_drv", "--problem", prob_txt] + common
    else: cmd = py + ["--tool", tool] + common
    if c.get("numa") == "numa0":
        cmd = ["numactl", "--cpunodebind=0", "--membind=0"] + cmd
    return cmd


def run_cmd(cmd, wall):
    t0 = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                         start_new_session=True)
    try:
        out, err = p.communicate(timeout=wall)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        p.communicate()
        return {"status": "timeout", "why": f"killed after {wall:.0f} s wall"}, time.time() - t0
    lines = [l for l in out.splitlines() if l.startswith("{")]
    if not lines:
        return {"status": "error", "why": f"exit {p.returncode}: {err[-600:]}"}, time.time() - t0
    return json.loads(lines[-1]), time.time() - t0


def read_scores(path):
    with open(path) as f:
        return [float(l) for l in f if l.strip()]


def check(scores, ref, dtype):
    if ref is None or len(scores) != len(ref):
        return False, f"{len(scores)} scores for {len(ref) if ref else '?'} pairs"
    worst = 0.0
    for s, r in zip(scores, ref):
        if s != s:
            return False, "nan"
        d = abs(s - r)
        tol = (0.5 + 1e-4 * abs(r)) if dtype == "float32" else 1e-6 * max(1.0, abs(r))
        if d > tol:
            return False, f"score {s} where the reference is {r}"
        worst = max(worst, d)
    return True, worst


def loadavg():
    try:
        return float(open("/proc/loadavg").read().split()[0])
    except Exception:
        return None


# ── references ───────────────────────────────────────────────────────────────────

def reference(env, set_file, c):
    """nwgrad float64 score-only per-pair scores (exact: every parameter is integral)."""
    key = f"{os.path.basename(set_file)[:-4]}_{c['gap']}_{c['mode']}"
    path = os.path.join(env["work"], "refs", key + ".txt")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        code = (
            "import sys; sys.path.insert(0, %r)\n"
            "import nwgrad\nfrom problems import problem, read_pairs, nwgrad_params\n"
            "A, B = read_pairs(%r)\n"
            "P = nwgrad_params(nwgrad, problem(%r, %r, %r))\n"
            "b = nwgrad.SeqPairBatchDouble(A, B, P, n_threads=0, gap_model=%r, mode=%r, grad_mode='none', traceback='pointers')\n"
            "b.score_and_grad()\n"
            "open(%r, 'w').write('\\n'.join(repr(float(s)) for s in b.scores()) + '\\n')\n"
        ) % (HERE, set_file, c["kind"], c["gap"], c["mode"], c["gap"], c["mode"], path + ".tmp")
        subprocess.run([env["py_gcc"], "-I", "-c", code], check=True)
        os.replace(path + ".tmp", path)
    return read_scores(path)


# ── main loop ────────────────────────────────────────────────────────────────────

def detect(env):
    b = env["bin"]
    have = {}
    for name in os.listdir(b) if os.path.isdir(b) else []:
        if name.startswith("seqan3_") or name.startswith("seqan2_"):
            have[name] = True
    have["cpp_gcc"] = os.path.exists(f"{b}/nwgrad_cpp_gcc")
    have["cpp_clang"] = os.path.exists(f"{b}/nwgrad_cpp_clang")
    have["clang_py"] = os.path.exists(env["py_clang"])
    for t in ("parasail", "ssw", "ksw2", "wfa2"):
        have[t] = os.path.exists(f"{b}/{t}_drv")
    probe = ("import importlib.util as u, json; print(json.dumps({m: u.find_spec(m) is not None "
             "for m in ['Bio', 'skbio', 'pyopal', 'edlib', 'deepblast', 'torch']}))")
    mods = json.loads(subprocess.run([env["py_gcc"], "-I", "-c", probe], capture_output=True, text=True).stdout)
    have["biopython"] = mods["Bio"]; have["scikit-bio"] = mods["skbio"]
    have["pyopal"] = mods["pyopal"]; have["edlib"] = mods["edlib"]
    have["deepblast"] = mods["deepblast"] and mods["torch"]
    info = json.loads(subprocess.run([env["py_gcc"], "-I", "-c",
        "import nwgrad, json; print(json.dumps([nwgrad.simd_isa(), nwgrad.available_isa_levels(), nwgrad.compiled_with()]))"],
        capture_output=True, text=True).stdout)
    env["auto_level"], env["levels"], env["compiler"] = info[0], info[1], info[2]
    env["have"] = have


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--host", default=socket.gethostname())
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--mt", action="append", default=[],
                    help="threads for the multithread phase: N or N:numa0 (repeatable)")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--only-phase", choices=["st", "mt"])
    ap.add_argument("--sets", nargs="*", help="restrict to these fixture sets (testing)")
    a = ap.parse_args()
    root = os.path.abspath(os.path.expanduser(a.root))
    env = dict(root=root, bin=f"{root}/bin", work=f"{root}/work", fixtures=f"{root}/fixtures",
               py_gcc=f"{root}/venv-gcc/bin/python", py_clang=f"{root}/venv-clang/bin/python",
               reps=a.reps)
    mt = []
    for spec in a.mt:
        n, _, numa = spec.partition(":")
        mt.append((spec, int(n), numa or None))
    env["mt"] = mt
    detect(env)
    os.makedirs(env["work"], exist_ok=True)
    for kind in ("protein", "dna"):
        for gap in ("affine", "linear"):
            for mode in ("global", "local"):
                write_problem_txt(f"{env['work']}/{kind}_{gap}_{mode}.prob", problem(kind, gap, mode))
    results = f"{root}/results/{a.host}.jsonl"
    os.makedirs(os.path.dirname(results), exist_ok=True)
    done = {}
    if os.path.exists(results):
        for line in open(results):
            try:
                r = json.loads(line)
                done[r["key"]] = r
            except Exception:
                pass
    cells = [(c, arms) for c, arms in plan(env)
             if (not a.only_phase or c["phase"] == a.only_phase) and (not a.sets or c["set"] in a.sets)]
    total_arms = sum(len(arms) for _, arms in cells)
    if a.plan:
        print(json.dumps({"cells": len(cells), "arms": total_arms, "have": env["have"],
                          "levels": env["levels"], "auto": env["auto_level"]}))
        for c, arms in cells[:5]:
            print(c, arms)
        return
    plan_keys = []
    for c, arms in cells:
        ck = f"{c['phase']}|{c['set']}|{c['mode']}|{c['gap']}|{c['out']}|{c['dtype']}|{c['tlabel']}"
        plan_keys.append([ck, [f"{t}:{v}" for t, v in arms]])
    with open(f"{root}/results/{a.host}.plan.json", "w") as f:
        json.dump({"cells": len(cells), "arms": total_arms, "started": time.time(),
                   "compiler": env["compiler"], "auto_level": env["auto_level"],
                   "levels": env["levels"], "have": env["have"], "reps": env["reps"],
                   "plan": plan_keys}, f)
    scores_out = f"{env['work']}/scores.txt"
    fout = open(results, "a", buffering=1)
    for c, arms in cells:
        ckey = f"{c['phase']}|{c['set']}|{c['mode']}|{c['gap']}|{c['out']}|{c['dtype']}|{c['tlabel']}"
        keys = [f"{ckey}|{t}:{v}" for t, v in arms]
        d0 = done.get(keys[0])
        if d0 is not None and d0["status"] != "ok":
            continue                                   # cell skipped (nwgrad cannot run it)
        if d0 is not None and all(k in done for k in keys):
            continue
        set_file = f"{env['fixtures']}/{c['set']}_{c['phase']}.tsv"
        prob_txt = f"{env['work']}/{c['kind']}_{c['gap']}_{c['mode']}.prob"
        ref = None
        T0 = d0["best_s"] if d0 else None
        soft_ref = None
        for (tool, variant), key in zip(arms, keys):
            if key in done:
                if tool == "nwgrad" and variant == "default" and c["out"] == "soft":
                    soft_ref = done[key].get("_scores")
                continue
            if ref is None and c["out"] != "soft":
                ref = reference(env, set_file, c)
            default = tool == "nwgrad" and variant == "default"
            deadline = 0 if default else TIMEOUT_X * T0
            wall = 3600 if default else 120 + WARMUP_SLACK + (env["reps"] + 2) * TIMEOUT_X * T0 * 1.3
            paused = 0.0
            while os.path.exists(f"{root}/PAUSE"):        # the coordinator saw the host busy
                time.sleep(30); paused += 30
            if os.path.exists(scores_out): os.remove(scores_out)
            cmd = arm_cmd(tool, variant, c, env, set_file, prob_txt, scores_out, deadline)
            l0 = loadavg(); t_start = time.time()
            res, wall_used = run_cmd(cmd, wall)
            l1 = loadavg()
            rec = dict(key=key, host=a.host, **{k: c[k] for k in ("phase", "set", "kind", "mode", "gap", "out", "dtype", "threads", "tlabel")},
                       tool=tool, variant=variant, status=res.get("status"), best_s=res.get("best_s"),
                       times=res.get("times"), cells=res.get("cells"), n_pairs=res.get("n_pairs"),
                       info=res.get("info"), why=res.get("why") or (res.get("info") or {}).get("why"), T0=T0, load_before=l0, load_after=l1,
                       t_start=t_start, t_end=time.time(), wall=wall_used, paused_before=paused)
            if rec["status"] == "ok":
                if rec["best_s"] and rec["cells"]:
                    rec["mcells_s"] = rec["cells"] / rec["best_s"] / 1e6
                sc = read_scores(scores_out) if os.path.exists(scores_out) else None
                if c["out"] == "soft":
                    if default:
                        soft_ref = sc
                        rec["_scores"] = sc if sc and len(sc) <= 4000 else None
                    elif (rec.get("info") or {}).get("unchecked"):
                        rec["check"] = "unchecked (a different model)"
                    elif soft_ref is not None and sc is not None:
                        ok = len(sc) == len(soft_ref) and all(abs(x - y) <= 1e-6 * max(1, abs(y)) for x, y in zip(sc, soft_ref))
                        if not ok: rec["status"] = "wrong"; rec["why"] = "soft score differs from the default arm's"
                elif sc is None:
                    rec["status"] = "error"; rec["why"] = "no scores written"
                else:
                    ok, detail = check(sc, ref, c["dtype"])
                    if ok: rec["max_err"] = detail
                    else: rec["status"] = "wrong"; rec["why"] = detail
            if default:
                if rec["status"] != "ok":
                    fout.write(json.dumps(rec) + "\n"); done[key] = rec
                    break                                # skip the cell
                T0 = rec["best_s"]
            fout.write(json.dumps(rec) + "\n")
            done[key] = rec
    fout.close()
    open(f"{root}/results/{a.host}.done", "w").write(str(time.time()))


if __name__ == "__main__":
    main()
