#!/usr/bin/env python3
"""nwgrad against the pairwise aligners people actually use, on the same problems.

    python tools/bench_competitors.py                        # build fixture, run all arms
    python tools/bench_competitors.py --lengths 100 300 1000 --quick
    python tools/bench_competitors.py --pairs-out pairs.json # freeze the fixture ...
    python tools/bench_competitors.py --pairs-in pairs.json  # ... and replay it on another host

Competitors (each optional; a missing package is reported and its arms skipped):
  parasail (striped SIMD, the de-facto fast aligner), Biopython PairwiseAligner,
  pyopal (Opal SIMD database search), scikit-bio pair_align (0.7+; it dropped SSW), and
  DeepBLAST's NeedlemanWunschDecoder (PyTorch/numba, the differentiable comparison).

SAME PROBLEM, OR NO NUMBER.  Every arm aligns BLOSUM62 with nwgrad's affine gaps
open=11, extend=1 (a gap of length k costs 11 + k).  The other tools spell that
differently -- determined empirically, by exact score agreement, not from their docs:
  parasail / pyopal  open=12, extend=1      (open + (k-1)*extend)
  Biopython          open=-12, extend=-1
  scikit-bio         gap_cost=(11, 1), free_ends=False (its DEFAULT is semi-global!)
Before anything is timed, each score-reporting arm must reproduce nwgrad's float64
score on every pair of the fixture, or the run aborts.  DeepBLAST is the exception: its
recurrence adds the match score at every cell whatever the move and leaves end gaps
free, so it solves a different model and is compared on cost per cell only.

Each (arm, length) runs in its own process (arms sharing one contaminate each other's
allocator and caches), single-threaded, best of 3 passes over the pair set.  The
fixture is real proteins when a FASTA is available (--fasta, default: the human
proteome in the repo root if present), each paired with a mutated copy: 20 %
substitutions, ~3 % deletions, ~3 % insertion runs of 1-5 -- homologs, not noise.
"""

import argparse
import json
import os
import subprocess
import sys
import time

GO, GE = 11, 1                      # nwgrad convention: gap of length k costs GO + GE*k
AA = "ACDEFGHIKLMNPQRSTVWY"
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FASTA = os.path.join(HERE, "..", "GCF_000001405.40_GRCh38.p14_protein.faa")


# ── fixture ──────────────────────────────────────────────────────────────────────

def read_fasta(path):
    seqs, cur = [], []
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if cur: seqs.append("".join(cur))
                cur = []
            else:
                cur.append(line.strip())
    if cur: seqs.append("".join(cur))
    return [s for s in seqs if set(s) <= set(AA)]


def build_pairs(lengths, n_for, fasta, seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    pool = read_fasta(fasta) if fasta and os.path.exists(fasta) else None
    def mutate(s):
        out = []
        for c in s:
            if rng.random() < 0.03: continue
            out.append(c if rng.random() > 0.2 else str(rng.choice(list(AA))))
            if rng.random() < 0.03: out.extend(rng.choice(list(AA), int(rng.integers(1, 6))))
        return "".join(out)
    fixture = {"source": "proteome" if pool else "synthetic", "pairs": {}}
    seen = set()
    for L in lengths:
        n = n_for(L)
        if pool:
            cand = [s for s in pool if 0.9 * L <= len(s) <= 1.1 * L and s not in seen]
            idx = rng.choice(len(cand), size=min(n, len(cand)), replace=False)
            base = [cand[i] for i in idx]
            seen.update(base)
        else:
            base = ["".join(rng.choice(list(AA), L)) for _ in range(n)]
        fixture["pairs"][str(L)] = [(a, mutate(a)) for a in base]
    return fixture


# ── arms: name -> (task, factory); factory() returns f(a, b) ─────────────────────

def nwgrad_params(nwgrad):
    from nwgrad.matrices import BLOSUM62
    return nwgrad.AlignParams(BLOSUM62, gap_open_a=GO, gap_extend_a=GE,
                              gap_open_b=GO, gap_extend_b=GE)


def arm_factories():
    A = {}

    def nw_score():
        import nwgrad; p = nwgrad_params(nwgrad)
        return lambda a, b: nwgrad.nw_score_affine(a, b, p)
    def nw_align():
        import nwgrad; p = nwgrad_params(nwgrad)
        return lambda a, b: nwgrad.nw_affine_grad(a, b, p)[0]      # score + path + gradient
    def sw_align():
        import nwgrad; p = nwgrad_params(nwgrad)
        return lambda a, b: nwgrad.sw_affine_grad(a, b, p)[0]
    def nw_soft():
        import nwgrad; p = nwgrad_params(nwgrad)
        return lambda a, b: nwgrad.nw_affine_soft_grad(a, b, p)[0]
    A["nwgrad"] = {"score_global": nw_score, "align_global": nw_align,
                   "align_local": sw_align, "soft_grad": nw_soft}
    # DeepBLAST's model is single-state (one softmax over three moves per cell), so the
    # like-for-like nwgrad arm is the LINEAR soft gradient; the affine one runs three
    # coupled states forward and backward and does several times the work per cell.
    def nw_soft_linear():
        import nwgrad; p = nwgrad_params(nwgrad)
        return lambda a, b: nwgrad.nw_soft_grad(a, b, p)[0]
    A["nwgrad-linear"] = {"soft_grad": nw_soft_linear}

    def ps(fn, trace):
        def make():
            import parasail
            f = getattr(parasail, fn)
            if trace:
                return lambda a, b: (lambda r: (r.cigar.seq, r.score)[1])(f(a, b, GO + 1, GE, parasail.blosum62))
            return lambda a, b: f(a, b, GO + 1, GE, parasail.blosum62).score
        return make
    A["parasail"] = {"score_global": ps("nw_striped_sat", False),
                     "align_global": ps("nw_trace_striped_sat", True),
                     "align_local": ps("sw_trace_striped_sat", True)}

    def bio(mode, full):
        def make():
            from Bio import Align
            from Bio.Align import substitution_matrices
            al = Align.PairwiseAligner()
            al.substitution_matrix = substitution_matrices.load("BLOSUM62")
            al.open_gap_score, al.extend_gap_score = -(GO + GE), -GE
            al.mode = mode
            if full:
                return lambda a, b: (lambda x: (x.coordinates, x.score)[1])(al.align(a, b)[0])
            return lambda a, b: al.score(a, b)
        return make
    A["biopython"] = {"score_global": bio("global", False), "align_global": bio("global", True),
                      "align_local": bio("local", True)}

    def opal(alg, mode):
        def make():
            import pyopal
            def f(a, b):
                r = list(pyopal.align(a, [b], "BLOSUM62", gap_open=GO + 1, gap_extend=GE,
                                      algorithm=alg, mode=mode, threads=1))[0]
                if mode == "full": _ = r.alignment
                return r.score
            return f
        return make
    A["pyopal"] = {"score_global": opal("nw", "score"), "align_global": opal("nw", "full"),
                   "align_local": opal("sw", "full")}

    def skb(mode):
        def make():
            import skbio.alignment as sa
            kw = dict(free_ends=False) if mode == "global" else {}
            return lambda a, b: sa.pair_align(a, b, mode=mode, sub_score="BLOSUM62",
                                              gap_cost=(GO, GE), **kw).score
        return make
    A["scikit-bio"] = {"align_global": skb("global"), "align_local": skb("local")}

    def deepblast():
        import torch
        torch.set_num_threads(1)
        from deepblast.nw import NeedlemanWunschDecoder
        from nwgrad.matrices import BLOSUM62
        alpha = BLOSUM62.alphabet
        M = torch.tensor(BLOSUM62.to_matrix(), dtype=torch.float32, requires_grad=True)
        dec = NeedlemanWunschDecoder("softmax")
        idx = {c: i for i, c in enumerate(alpha)}
        def f(a, b):
            ia = torch.tensor([idx[c] for c in a]); ib = torch.tensor([idx[c] for c in b])
            theta = M[ia][:, ib].unsqueeze(0)                      # (1, m, n), grad flows to M
            A_ = torch.full_like(theta, -float(GE))
            v = dec(theta, A_)
            v.sum().backward()
            M.grad = None
            return float(v.sum())
        return f
    A["deepblast"] = {"soft_grad": deepblast}
    return A


TASKS = {
    "score_global": "global affine, score only",
    "align_global": "global affine, score + alignment",
    "align_local":  "local affine, score + alignment",
    "soft_grad":    "soft (forward-backward) gradient",
}


# ── child: time one arm at one length ─────────────────────────────────────────────

def child(args):
    with open(args.pairs_in) as f:
        pairs = json.load(f)["pairs"][str(args.length)]
    f = arm_factories()[args.tool][args.task]()
    for a, b in pairs[:3]:
        f(a, b)                                      # warm-up (numba JIT, lazy imports)
    cells = sum(len(a) * len(b) for a, b in pairs)
    best = None
    for _ in range(args.repeats):
        t0 = time.perf_counter(); done = 0
        for a, b in pairs:
            f(a, b); done += len(a) * len(b)
            if time.perf_counter() - t0 > args.budget: break
        dt = time.perf_counter() - t0
        rate = done / dt
        best = rate if best is None or rate > best else best
    print(json.dumps({"tool": args.tool, "task": args.task, "length": args.length,
                      "mcells_per_s": best / 1e6, "pair_cells": cells}))


# ── parent: gate, then orchestrate ────────────────────────────────────────────────

def available(tool):
    mod = {"nwgrad": "nwgrad", "nwgrad-linear": "nwgrad", "parasail": "parasail", "biopython": "Bio", "pyopal": "pyopal",
           "scikit-bio": "skbio", "deepblast": "deepblast"}[tool]
    try:
        __import__(mod); return True
    except Exception:
        return False


def correctness_gate(fixture, factories, tools):
    import nwgrad
    p = nwgrad_params(nwgrad)
    refs = {"score_global": nwgrad.nw_score_affine_double, "align_global": nwgrad.nw_score_affine_double,
            "align_local": nwgrad.sw_score_affine_double}
    gate_pairs = [pr for L in fixture["pairs"] for pr in fixture["pairs"][L][:8]
                  if len(pr[0]) <= 1200]
    for tool in tools:
        for task, make in factories[tool].items():
            if task not in refs: continue
            f = make()
            for a, b in gate_pairs:
                want, got = refs[task](a, b, p), f(a, b)
                if abs(got - want) > 1e-6 * max(1.0, abs(want)):
                    sys.exit(f"GATE FAILED: {tool}/{task} scored {got} where nwgrad (float64) "
                             f"scores {want} on a {len(a)}x{len(b)} pair -- not the same problem")
    print(f"correctness gate: every score-reporting arm reproduces nwgrad's float64 score "
          f"on {len(gate_pairs)} pairs", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lengths", type=int, nargs="+", default=[100, 300, 1000, 3000])
    ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--pairs-in"); ap.add_argument("--pairs-out")
    ap.add_argument("--seed", type=int, default=20261002)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--budget", type=float, default=20.0, help="seconds per pass (slow arms stop early)")
    ap.add_argument("--quick", action="store_true", help="fewer pairs, 1 repeat")
    ap.add_argument("--json", help="also write raw results here")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--tool"); ap.add_argument("--task"); ap.add_argument("--length", type=int)
    args = ap.parse_args()
    if args.child:
        return child(args)

    # ~3e7 cells per length (~4e6 with --quick), at least 2 pairs
    n_for = lambda L: max(2, int((4e6 if args.quick else 3e7) / (L * L)))
    if args.quick: args.repeats = 1
    if args.pairs_in:
        with open(args.pairs_in) as f: fixture = json.load(f)
        args.lengths = [int(L) for L in fixture["pairs"]]
    else:
        fixture = build_pairs(args.lengths, n_for, args.fasta, args.seed)
    path = args.pairs_out or os.path.join("/tmp", f"nwgrad_bench_pairs_{os.getpid()}.json")
    with open(path, "w") as f: json.dump(fixture, f)

    import nwgrad
    factories = arm_factories()
    tools = [t for t in factories if available(t)]
    missing = [t for t in factories if t not in tools]
    print(f"nwgrad {nwgrad.__version__} ({nwgrad.simd_isa()}, {nwgrad.compiled_with()}); "
          f"fixture: {fixture['source']}, lengths {args.lengths}; "
          f"tools: {', '.join(tools)}" + (f"; MISSING: {', '.join(missing)}" if missing else ""), flush=True)
    correctness_gate(fixture, factories, tools)

    results = []
    for task in TASKS:
        for L in args.lengths:
            for tool in tools:
                if task not in factories[tool]: continue
                cmd = [sys.executable, os.path.abspath(__file__), "--child", "--pairs-in", path,
                       "--tool", tool, "--task", task, "--length", str(L),
                       "--repeats", str(args.repeats), "--budget", str(args.budget)]
                out = subprocess.run(cmd, capture_output=True, text=True)
                line = [l for l in out.stdout.splitlines() if l.startswith("{")]
                if out.returncode or not line:
                    print(f"  {tool}/{task}/{L}: FAILED\n{out.stderr[-600:]}", flush=True); continue
                results.append(json.loads(line[-1]))
                print(f"  {task:13s} L={L:<5d} {tool:11s} {results[-1]['mcells_per_s']:9.1f} Mcell/s", flush=True)
    if args.json:
        with open(args.json, "w") as f: json.dump(results, f, indent=1)

    print()
    for task, title in TASKS.items():
        rows = [r for r in results if r["task"] == task]
        if not rows: continue
        tools_t = [t for t in tools if any(r["tool"] == t for r in rows)]
        print(f"## {title}  (Mcell/s, single thread; x = nwgrad / tool)")
        print("| L | " + " | ".join(tools_t) + " |")
        print("|---|" + "---|" * len(tools_t))
        for L in args.lengths:
            get = {r["tool"]: r["mcells_per_s"] for r in rows if r["length"] == L}
            nw = get.get("nwgrad")
            cells = []
            for t in tools_t:
                v = get.get(t)
                if v is None: cells.append("—")
                elif t == "nwgrad" or nw is None: cells.append(f"{v:.1f}")
                else: cells.append(f"{v:.1f} ({nw / v:.2g}x)")
            print(f"| {L} | " + " | ".join(cells) + " |")
        print()


if __name__ == "__main__":
    main()
