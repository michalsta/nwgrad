#!/usr/bin/env python3
"""A/B benchmark of the batch layer between two nwgrad builds (e.g. 0.5.x and 0.6).

    python tools/bench_ab.py --a OLD_PYTHON --b NEW_PYTHON --data pairs.tsv \\
        [--n 100000] [--threads 1 8] [--rounds 5] [--control]

Every measurement runs in a fresh process, the two builds ALTERNATING (a, b, a, b, ...),
best of --rounds per arm: separately compiled builds differ in code layout, and a
shared process would let one arm warm the other's caches.  --control runs the A build
against itself first; read every ratio relative to it (if the control is not ~1.00x,
the host is too noisy for the numbers below it).

Sections (the DiscrimAlign shape: miRNA x target-site pairs, local affine, double):
  construct      add_many() of N pairs, seconds and resident MB added
  iter_hard      set_params + score_and_grad + scores + weighted_grad  (one training step)
  iter_banded    set_params + banded_grad(4) + scores + weighted_grad
  iter_soft      the same, grad_mode soft, soft_guide lazy, T = 0.5
  stream         streaming align() of the N pairs (BatchAligner before 0.6), hard
  seqpair        2000 standalone SeqPair(...).score_and_grad() calls (per-pair overhead)
"""
import argparse, json, os, subprocess, sys, time

CHILD = r'''
import json, os, resource, sys, time, warnings
warnings.simplefilter("ignore")
import numpy as np, nwgrad
sec, n, threads, data = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
rows = [l.rstrip("\n").split("\t") for l in open(data)][1:]
rng = np.random.default_rng(0)
idx = rng.choice(len(rows), size=min(n, len(rows)), replace=False)
A = [rows[i][0] for i in idx]; B = [rows[i][1] for i in idx]
M = np.array([[0.53, -0.61, -0.60, -0.56], [-0.44, 0.58, -0.57, -0.48],
              [-0.46, -0.67, 0.58, -0.60], [-0.46, -0.51, -0.61, 0.51]])
def params(k):
    return nwgrad.AlignParams(nwgrad.SubstMatrix(M * k, alphabet="ACGT"), 1.0, 0.5, 1.0, 0.5)
new = not hasattr(nwgrad, "BatchAligner")
def batch(gd, **kw):
    if new:
        b = nwgrad.SeqPairBatchDouble(threads, "pointers", gap_model="affine", mode="local",
                                      grad_mode=gd)
    else:
        b = nwgrad.SeqPairBatchDouble(threads, "pointers")
    for k, v in kw.items(): setattr(b, k, v)
    if new: b.add_many(A, B, params(1.0))
    else:   b.add_many(A, B, params(1.0), gap_model="affine", mode="local", grad_mode=gd)
    return b
def rss_mb():
    try:
        for l in open("/proc/self/status"):
            if l.startswith("VmRSS:"): return int(l.split()[1]) / 1024
    except OSError:
        pass
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 * 1024 if sys.platform == "darwin" else 1024)
w = rng.normal(size=len(A))
out = {}
if sec == "construct":
    r0 = rss_mb(); t = time.perf_counter(); b = batch("hard")
    out["s"] = time.perf_counter() - t; out["mb"] = rss_mb() - r0
elif sec in ("iter_hard", "iter_banded", "iter_soft"):
    gd = "soft" if sec == "iter_soft" else "hard"
    kw = {"soft_guide": "lazy", "soft_temperature": 0.5, "soft_impl": "scaled_or_log"} if gd == "soft" else {}
    b = batch(gd, **kw)
    b.score_and_grad()
    ts = []
    for k in range(4):
        p = params(1.0 + 0.01 * (k + 1))
        t = time.perf_counter()
        b.set_params(p)
        if sec == "iter_banded": b.banded_grad(4)
        else: b.score_and_grad()
        s = b.scores(); g = b.weighted_grad(w)
        ts.append(time.perf_counter() - t)
    out["s"] = min(ts)
elif sec == "stream":
    p = params(1.0)
    if new:
        b = nwgrad.SeqPairBatchDouble(threads, gap_model="affine", mode="local", grad_mode="hard")
        f = lambda: b.align(A, B, p)
    else:
        ba = nwgrad.BatchAlignerDouble(p, gap_model="affine", mode="local", grad_mode="hard",
                                       n_threads=threads)
        f = lambda: ba.align(A, B)
    f(); ts = []
    for _ in range(3):
        t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
    out["s"] = min(ts)
elif sec == "seqpair":
    p = params(1.0); m = min(2000, len(A))
    t = time.perf_counter()
    for i in range(m):
        nwgrad.SeqPairDouble(A[i], B[i], p, gap_model="affine", mode="local").score_and_grad()
    out["s"] = (time.perf_counter() - t) / m
print(json.dumps(out))
'''


def measure(py, env, sec, n, threads, data):
    e = dict(os.environ)
    if env: e["PYTHONPATH"] = env
    p = subprocess.run([py, "-c", CHILD, sec, str(n), str(threads), data], env=e,
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f"{sec} failed under {py}:\n{p.stderr[-2000:]}")
    return json.loads(p.stdout.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--a-env", default=""); ap.add_argument("--b-env", default="")
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--threads", type=int, nargs="+", default=[1])
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--sections", nargs="+",
                    default=["construct", "iter_hard", "iter_banded", "iter_soft", "stream", "seqpair"])
    ap.add_argument("--control", action="store_true")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    arms = [("A", a.a, a.a_env), ("B", a.b, a.b_env)]
    if a.control:
        arms = [("A", a.a, a.a_env), ("A'", a.a, a.a_env)] + [("B", a.b, a.b_env)]
    rows = []
    print(f"host={os.uname().nodename} n={a.n} rounds={a.rounds}")
    print(f"{'section':12s} {'thr':>3s} " + " ".join(f"{n:>10s}" for n, _, _ in arms) +
          "   B/A" + ("  A'/A" if a.control else ""))
    for thr in a.threads:
        for sec in a.sections:
            if sec == "seqpair" and thr != a.threads[0]:
                continue
            best = {n: None for n, _, _ in arms}
            for _ in range(a.rounds):
                for n, py, env in arms:
                    r = measure(py, env, sec, a.n, thr, a.data)
                    if best[n] is None or r["s"] < best[n]["s"]:
                        best[n] = r
            ratio = best["B"]["s"] / best["A"]["s"]
            scale, unit = (1e6, "us") if sec == "seqpair" else (1.0, "s")
            line = (f"{sec:12s} {thr:3d} " +
                    " ".join(f"{best[n]['s'] * scale:10.4f}" for n, _, _ in arms) +
                    f"  {ratio:5.2f}")
            if a.control:
                ctrl = best["A'"]["s"] / best["A"]["s"]
                line += f"  {ctrl:5.2f}"
            line += f"  ({unit})"
            if sec == "construct":
                line += "   MB: " + " ".join(f"{best[n]['mb']:.0f}" for n, _, _ in arms)
            print(line, flush=True)
            rows.append({"section": sec, "threads": thr, **{n: best[n] for n, _, _ in arms}})
    if a.json:
        json.dump({"host": os.uname().nodename, "n": a.n, "rows": rows}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
