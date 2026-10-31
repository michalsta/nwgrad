#!/usr/bin/env python3
"""Sweep the fill gates per ISA level: where does each crossover lie on THIS CPU?

    python tools/gate_sweep.py --out gates_host.jsonl                 # every level, 1 + all threads
    python tools/gate_sweep.py --levels avx2 --threads 1 --sections hard-affine score-affine
    python tools/gate_sweep.py --summary gates_*.jsonl                # the tables, from the logs

Every gate (nwgrad._tuning) picks between fills that are bit-identical, so for each
configuration (level, threads, section, fixture, type, gap model, mode, lengths) three
arms run in ONE process, same binary, interleaved, best of --reps:
  inter     every inter-pair cap off: any eligible pair joins a shared group
  own       inter_group_bytes = 0 (and inter_soft_bytes): every pair its own fill
  default   the compiled gates
and for the striped-vs-scalar linear gate (section striped-min), own fill only:
  striped   linear_striped_min_n = 0 (the striped kernel from any length)
  scalar    linear_striped_min_n = 10**6 (the scalar fill always)
  default
The summary prints, per configuration and length, inter/own (or striped/scalar) and
default/best: a default above 1.05 at some length is a gate set wrongly for this CPU.
Each arm's first call after a gate change regroups the plan; it is warmed up untimed.
Scores are checked equal across arms once per configuration (they must be).
"""
from __future__ import annotations
import argparse
import json
import os
import platform
import socket
import sys
import time

import numpy as np

TYPES = ["float32", "double", "int16", "int32", "int64"]
SECTIONS = ["hard-affine", "hard-linear", "score-affine", "score-linear",
            "soft-affine", "soft-linear", "striped-min", "dna-hard-affine",
            "dna-hard-linear", "dna-score-affine", "dna-striped-min"]
LENS_AFF = [40, 60, 80, 100, 120, 140, 160, 200, 250, 300]
LENS_LIN = [40, 60, 80, 100, 120, 160, 200, 250, 300, 400]
LENS_SCORE = [40, 60, 80, 100, 120, 150, 200, 250]
LENS_SOFT = [20, 30, 40, 60, 80, 100, 120, 160]
LENS_MINN = [16, 24, 32, 40, 48, 56, 64, 80, 96, 128]
LENS_DNA = [30, 50, 80, 120, 200]


def cls_of(nwgrad, t):
    return {"float32": nwgrad.SeqPairBatch, "double": nwgrad.SeqPairBatchDouble,
            "int16": nwgrad.SeqPairBatchInt16, "int32": nwgrad.SeqPairBatchInt32,
            "int64": nwgrad.SeqPairBatchInt64}[t]


def fixture(nwgrad, name, n, la, lb, seed):
    """Related pairs: B is A with 30 % substitutions (protein) or carries a mutated copy
    of A at a random offset (dna, |A| = la < |B| = lb).  Integral parameters, so the
    integer types run the same problem."""
    rng = np.random.default_rng(seed)
    if name == "protein":
        from nwgrad import matrices
        p = nwgrad.AlignParams(matrices.BLOSUM62, 11.0, 1.0, 11.0, 1.0)
        alpha = p.matrix.alphabet
        syms = np.array(list(alpha.symbols if hasattr(alpha, "symbols") else alpha))
        A, B = [], []
        for _ in range(n):
            a = syms[rng.integers(0, len(syms), la)]
            b = syms[rng.integers(0, len(syms), lb)]
            k = min(la, lb)
            mut = rng.random(k) < 0.3
            b[:k] = np.where(mut, b[:k], a[:k])
            A.append("".join(a)); B.append("".join(b))
        return p, A, B
    if name == "dna":
        alpha = "ACGT"
        m = np.where(np.eye(4, dtype=bool), 2.0, -3.0)
        p = nwgrad.AlignParams(nwgrad.SubstMatrix(m, alphabet=alpha), 5.0, 2.0, 5.0, 2.0)
        syms = np.array(list(alpha))
        A, B = [], []
        for _ in range(n):
            a = syms[rng.integers(0, 4, la)]
            b = syms[rng.integers(0, 4, lb)]
            if lb >= la:
                off = int(rng.integers(0, lb - la + 1))
                mut = rng.random(la) < 0.2
                b[off:off + la] = np.where(mut, b[off:off + la], a)
            A.append("".join(a)); B.append("".join(b))
        return p, A, B
    raise SystemExit(f"unknown fixture {name}")


def n_pairs(la, lb, threads_eff, cells_target):
    n = int(cells_target * max(1.0, threads_eff) ** 0.7 / (la * lb))
    n = max(128, min(n, 60000))
    return (n + 63) // 64 * 64


def timeit(fn, min_time):
    t0 = time.perf_counter(); reps = 0
    while True:
        fn(); reps += 1
        dt = time.perf_counter() - t0
        if dt >= min_time: break
    return dt / reps


INTER_OFF = dict(inter_group_bytes=None, inter_soft_bytes=None, inter_score_gather_cells=None,
                 inter_linear_int_gather_cells=None, inter_linear_f32_gather_cells=None,
                 inter_linear_f64_gather_cells=None, inter_global_linear_int_cells=None,
                 inter_global_linear_f32_cells=None)
OWN = dict(inter_group_bytes=0, inter_soft_bytes=0)
ARMS_INTER = {"inter": INTER_OFF, "own": OWN, "default": {}}
ARMS_MINN = {"striped": dict(OWN, linear_striped_min_n=None),
             "scalar": dict(OWN, linear_striped_min_n=10 ** 6),
             "default": dict(OWN)}


def section_plan(sec):
    """(fixture, gap_model, output, types, arms, lengths) of a section."""
    dna = sec.startswith("dna-")
    s = sec[4:] if dna else sec
    out, gap = s.split("-") if "-" in s else (s, None)
    if out == "striped":
        return (("dna" if dna else "protein"), "linear", "hard", TYPES, ARMS_MINN,
                [(22, n) for n in LENS_MINN] if dna else [(40, n) for n in LENS_MINN])
    if dna:
        lens = [(22, n) for n in LENS_DNA]
    elif out == "soft":
        lens = [(L, L) for L in LENS_SOFT]
    elif out == "score":
        lens = [(L, L) for L in LENS_SCORE]
    else:
        lens = [(L, L) for L in (LENS_AFF if gap == "affine" else LENS_LIN)]
    types = ["float32", "double"] if out == "soft" else TYPES
    return ("dna" if dna else "protein"), gap, out, types, ARMS_INTER, lens


def run(args):
    os.environ.setdefault("LC_ALL", "C")
    import nwgrad
    from nwgrad import _tuning
    host = args.host or socket.gethostname()
    levels = args.levels or [l for l in nwgrad.available_isa_levels() if l != "scalar_fallback"]
    ncpu = os.cpu_count() or 1
    head = dict(host=host, machine=platform.machine(), compiler=nwgrad.compiled_with(),
                version=nwgrad.__version__, levels=levels, ncpu=ncpu, defaults=_tuning.defaults())
    print(json.dumps(dict(header=head)), flush=True)
    out = open(args.out, "a") if args.out else None

    def emit(rec):
        line = json.dumps(rec)
        print(line, flush=True)
        if out: out.write(line + "\n"); out.flush()

    for sec in args.sections:
        fx, gap, output, types, arms, lens = section_plan(sec)
        if args.types: types = [t for t in types if t in args.types]
        for level in levels:
            for th in args.threads:
                th_eff = ncpu if th == 0 else th
                for la, lb in lens:
                    n = n_pairs(la, lb, th_eff, args.cells)
                    p, A, B = fixture(nwgrad, fx, n, la, lb, args.seed)
                    for mode in args.modes:
                        for t in types:
                            kw = dict(gap_model=gap, mode=mode, n_threads=th, kernel=level)
                            if output == "score": kw["grad_mode"] = "none"
                            if output == "soft": kw["grad_mode"] = "soft"
                            try:
                                batch = cls_of(nwgrad, t)(A, B, p, **kw)
                                if output == "soft": batch.soft_impl = "scaled_or_log"
                            except (ValueError, TypeError, RuntimeError) as e:
                                emit(dict(level=level, threads=th, section=sec, fixture=fx, type=t,
                                          gap=gap, mode=mode, output=output, la=la, lb=lb, n=n,
                                          skipped=str(e)[:120]))
                                continue
                            call = batch.score_and_grad
                            best = {a: float("inf") for a in arms}
                            ref = None
                            for r in range(args.reps):
                                order = list(arms) if r % 2 == 0 else list(arms)[::-1]
                                for a in order:
                                    with _tuning.override(**arms[a]):
                                        call()                     # regroup + warm-up, untimed
                                        if r == 0:
                                            sc = np.asarray(batch.scores())
                                            if ref is None: ref = sc
                                            elif not np.array_equal(sc, ref):
                                                raise SystemExit(f"{sec} {level} {t} {mode} {la}x{lb}: "
                                                                 f"arm {a} scores differ")
                                        dt = timeit(call, args.min_time)
                                    best[a] = min(best[a], dt)
                            rec = dict(level=level, threads=th, section=sec, fixture=fx, type=t,
                                       gap=gap, mode=mode, output=output, la=la, lb=lb, n=n,
                                       us={a: best[a] / n * 1e6 for a in arms})
                            emit(rec)
    if out: out.close()


def summary(paths, thresh):
    rows = []
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                r = json.loads(line)
                if "header" in r:
                    print(f"# {r['header']['host']} {r['header']['machine']} {r['header']['compiler']} "
                          f"levels {r['header']['levels']} ncpu {r['header']['ncpu']}")
                    continue
                if "skipped" in r: continue
                r["host"] = os.path.basename(path)
                rows.append(r)
    key = lambda r: (r["host"], r["section"], r["level"], r["threads"], r["gap"] or "", r["mode"],
                     r["type"])
    rows.sort(key=lambda r: key(r) + (r["la"], r["lb"]))
    cur = None
    losses = []
    for r in rows:
        k = key(r)
        if k != cur:
            cur = k
            print(f"\n== {r['host']} {r['section']} {r['level']} thr={r['threads']} {r['gap']} "
                  f"{r['mode']} {r['type']}  (us/pair; ratio = first/second arm; dflt = default/best)")
        us = r["us"]
        arms = list(us)
        a, b, d = us[arms[0]], us[arms[1]], us["default"]
        best = min(a, b)
        flag = "  <-- default loses" if d / best > thresh else ""
        print(f"   {r['la']:4d}x{r['lb']:<4d} n={r['n']:<6d} {arms[0]} {a:9.3f}  {arms[1]} {b:9.3f}  "
              f"ratio {a / b:5.2f}  default {d:9.3f}  dflt {d / best:5.2f}{flag}")
        if d / best > thresh:
            losses.append((d / best, k, r["la"], r["lb"]))
    if losses:
        print(f"\n## defaults losing more than {thresh:.2f}x, worst first")
        for loss, k, la, lb in sorted(losses, reverse=True):
            print(f"   {loss:5.2f}  {' '.join(map(str, k))} {la}x{lb}")


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--sections", nargs="+", default=SECTIONS, choices=SECTIONS)
    ap.add_argument("--levels", nargs="+", default=None)
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 0], help="0 = the default (all)")
    ap.add_argument("--modes", nargs="+", default=["local", "global"])
    ap.add_argument("--types", nargs="+", default=None)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--min-time", type=float, default=0.2)
    ap.add_argument("--cells", type=float, default=1.0e8, help="DP cells per batch call at 1 thread")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--host", default=None)
    ap.add_argument("--out", default=None, help="append JSON lines here")
    ap.add_argument("--summary", nargs="+", metavar="JSONL", help="print the tables from logs and exit")
    ap.add_argument("--thresh", type=float, default=1.05)
    args = ap.parse_args()
    if args.summary:
        summary(args.summary, args.thresh)
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
