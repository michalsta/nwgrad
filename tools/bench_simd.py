#!/usr/bin/env python3
"""Benchmark the scalar and simd Viterbi kernels across every ISA this CPU offers.

    python tools/bench_simd.py                      # sensible defaults
    python tools/bench_simd.py --seq-len 200 400    # sweep lengths
    python tools/bench_simd.py --threads 1 8 60     # sweep thread counts
    python tools/bench_simd.py --quick              # rough numbers, ~4x faster

Requires only the installed package (`pip install -e .`) — no cmake build.


WHY THIS IS MORE PARANOID THAN A BENCHMARK USUALLY NEEDS TO BE
=============================================================

Every guard below exists because a naive version of this script produced a *confidently
wrong* answer during development.  In order of how badly each one bit:

1. NEVER COMPARE ACROSS SEPARATELY-COMPILED BINARIES.
   A microbenchmark reported the simd kernel at 0.80x — slower than scalar — on an
   AVX2 machine.  It was reproducible, and its control arm was clean.  It was also
   nonsense: the *same* scalar source measured 535 Mcell/s in one binary and 338 in
   another.  Code layout (a hot loop falling in or out of the uop cache) moved the
   result by 1.6x, which was far larger than the effect being measured.
   -> So: one process, one binary, everything selected at RUNTIME.  The kernel is a
      runtime argument and the ISA is a runtime env var precisely so that every arm
      shares identical codegen.  This script never compiles anything.

2. A CONTROL ARM, ALWAYS.
   Scalar-vs-scalar must report 1.00x.  When it did not, the harness was measuring
   itself: the first-run arm was paying page-fault and warm-up costs and the
   "speedup" was pure artifact.  If the control drifts, this script says so loudly
   and refuses to let you read the real numbers as gospel.

3. RUNS MUST BE LONG.
   Timings of 0.03-0.5s on a 60-core NUMA box were noise, and they said the simd win
   "evaporated at 60 threads".  It did not.  Batch size is auto-scaled here until a
   single measurement takes at least --min-time seconds.

4. MINIMUM, NOT MEAN.
   Interference (thermal, frequency, neighbours) only ever makes a run slower.  The
   minimum over interleaved repeats is the robust estimator; the mean is a measure of
   how busy the machine was.

5. INTERLEAVE THE ARMS.
   Round-robin, never all-of-A-then-all-of-B, so any drift over the run hits both
   arms equally instead of accruing entirely to whichever went first.

6. LC_ALL=C.
   Not a performance issue — a parsing one.  A locale that groups thousands turns
   "3 000" into a number your regex reads as 3.

7. CHECK THE RESULTS MATCH.
   The two kernels are bit-exact by design.  A speedup obtained by computing something
   different is not a speedup, so scores and gradients are compared byte-for-byte and
   a mismatch is a hard failure, not a footnote.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time

# The ISA levels the dispatch table knows about, weakest first.
ISA_LEVELS = ["baseline", "avx", "avx2", "avx512"]


# ─────────────────────────────────────────────────────────────────────────────
# Worker: runs inside a subprocess with NWGRAD_ISA fixed.
#
# It must be a subprocess: the ISA is resolved once, on first use, into a static
# function-pointer table.  A single process therefore cannot benchmark two ISA levels,
# and a script that tried would silently report the first one twice.
# ─────────────────────────────────────────────────────────────────────────────

AA = "ARNDCQEGHILKMFPSTWYV"  # the NCBI order's 20 canonical residues


class SeqPool:
    """Sequences, generated once and only ever extended.

    Two traps live here, and a naive version fell into both.  The auto-sizer calls for
    a batch several times as it grows, so regenerating from scratch each round makes
    the benchmark spend its life building strings rather than aligning them.  And
    `random.choice` per *character* is a Python-level call per residue: at the batch
    sizes a 60-thread run needs, that is tens of millions of them.  `random.choices`
    with k=length is one C-level call per sequence.

    The pool grows monotonically and is shared across thread counts, so the large batch
    a many-threaded run needs is built once and the small one is a prefix of it.
    """

    def __init__(self, length, seed):
        import random
        self.rng = random.Random(seed)
        self.length = length
        self.a: list[str] = []
        self.b: list[str] = []

    def take(self, n):
        while len(self.a) < n:
            self.a.append("".join(self.rng.choices(AA, k=self.length)))
            self.b.append("".join(self.rng.choices(AA, k=self.length)))
        return self.a[:n], self.b[:n]


def worker(args):
    import nwgrad
    from nwgrad.matrices import BLOSUM62

    isa = nwgrad.simd_isa()
    params = nwgrad.AlignParams(BLOSUM62, args.gap_open, args.gap_extend,
                                args.gap_open, args.gap_extend)

    def aligner(kernel, threads):
        return nwgrad.BatchAligner(params, band=0, gap_model=args.gap_model,
                                   mode=args.mode, grad_mode=args.grad_mode,
                                   n_threads=threads, kernel=kernel)

    def time_once(kernel, threads, sa, sb):
        al = aligner(kernel, threads)
        t0 = time.perf_counter()
        res = al.align(sa, sb)
        return time.perf_counter() - t0, res

    pool = SeqPool(args.seq_len, args.seed)

    out = []
    for threads in args.threads:
        # Auto-size the batch so one measurement is long enough to mean anything.
        # A 30ms run on a many-core NUMA box measures scheduling jitter, not the DP —
        # that mistake is what made the simd win look like it "evaporated at 60 threads".
        n = 256
        while True:
            sa, sb = pool.take(n)
            dt, _ = time_once("scalar", threads, sa, sb)
            if dt >= args.min_time or n >= args.max_n:
                break
            grow = max(2.0, min(8.0, args.min_time / max(dt, 1e-4)))
            n = min(args.max_n, int(n * grow) + 1)

        sa, sb = pool.take(n)

        # Three arms.  "control" is scalar again: it must come out at 1.00x, and if it
        # does not, this harness is measuring itself and nothing below can be trusted.
        arms = ["scalar", "simd", "control"]
        kern = {"scalar": "scalar", "simd": "simd", "control": "scalar"}
        best = {a: float("inf") for a in arms}
        results = {}

        # Warm up every arm before timing any of them: first touch of the DP tables
        # takes page faults that would otherwise be billed entirely to whoever ran first.
        for a in arms:
            time_once(kern[a], threads, sa, sb)

        # Interleaved, minimum-of-repeats.  Drift hits all arms equally; interference
        # only ever adds time, so the minimum is the robust estimator.
        for _ in range(args.repeats):
            for a in arms:
                dt, res = time_once(kern[a], threads, sa, sb)
                best[a] = min(best[a], dt)
                results[a] = res

        # A speedup from computing something else is not a speedup.  The kernels are
        # bit-exact by construction, so this is exact equality, not a tolerance.
        s_scores = list(results["scalar"].scores)
        m_scores = list(results["simd"].scores)
        exact = (s_scores == m_scores)
        if args.grad_mode != "none":
            sg = results["scalar"].grad.matrix.to_matrix().tobytes()
            mg = results["simd"].grad.matrix.to_matrix().tobytes()
            exact = exact and (sg == mg)

        cells = n * (args.seq_len + 1) ** 2
        out.append({
            "isa": isa,
            "threads": threads,
            "n": n,
            "scalar_s": best["scalar"],
            "simd_s": best["simd"],
            "control_s": best["control"],
            "scalar_pairs_s": n / best["scalar"],
            "simd_pairs_s": n / best["simd"],
            "scalar_mcells_s": cells / best["scalar"] / 1e6,
            "simd_mcells_s": cells / best["simd"] / 1e6,
            "speedup": best["scalar"] / best["simd"],
            "control_ratio": best["scalar"] / best["control"],
            "exact": exact,
        })

    print("@@JSON@@" + json.dumps(out))


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

def probe_isas():
    """Which ISA levels this CPU can actually run.

    Asked, not assumed.  NWGRAD_ISA requesting an ISA the CPU lacks falls back to
    baseline rather than trapping — which is the right runtime behaviour and a trap for
    a benchmark, because it would silently report "avx512" numbers that are really
    baseline numbers measured three times.  So: force each level, ask what we got, and
    keep only the ones that answered honestly.
    """
    available = []
    for isa in ISA_LEVELS:
        env = dict(os.environ, NWGRAD_ISA=isa, LC_ALL="C")
        try:
            got = subprocess.run(
                [sys.executable, "-c", "import nwgrad; print(nwgrad.simd_isa())"],
                env=env, capture_output=True, text=True, timeout=60, check=True,
            ).stdout.strip()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            sys.exit(f"error: could not import nwgrad ({e}). Is it installed? pip install -e .")
        if got == isa:
            available.append(isa)
    return available


def run_isa(isa, args):
    env = dict(os.environ, NWGRAD_ISA=isa, LC_ALL="C")
    cmd = [sys.executable, os.path.abspath(__file__), "--_worker",
           "--seq-len", str(args.seq_len),
           "--gap-model", args.gap_model, "--mode", args.mode,
           "--grad-mode", args.grad_mode,
           "--gap-open", str(args.gap_open), "--gap-extend", str(args.gap_extend),
           "--repeats", str(args.repeats), "--min-time", str(args.min_time),
           "--max-n", str(args.max_n), "--seed", str(args.seed),
           "--threads", *[str(t) for t in args.threads]]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit(f"worker failed for ISA={isa}:\n{proc.stdout}\n{proc.stderr}")
    for line in proc.stdout.splitlines():
        if line.startswith("@@JSON@@"):
            return json.loads(line[len("@@JSON@@"):])
    sys.exit(f"worker produced no result for ISA={isa}:\n{proc.stdout}\n{proc.stderr}")


def cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def main():
    p = argparse.ArgumentParser(
        description="Benchmark the scalar vs simd Viterbi kernels across ISA levels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--seq-len", type=int, default=200)
    p.add_argument("--threads", type=int, nargs="+", default=None,
                   help="thread counts to sweep (default: 1 and all cores)")
    p.add_argument("--gap-model", default="affine", choices=["affine", "linear"],
                   help="linear has no simd kernel by design; expect 1.00x")
    p.add_argument("--mode", default="global", choices=["global", "local"])
    p.add_argument("--grad-mode", default="hard", choices=["hard", "soft", "none"],
                   help="soft is unaffected by the kernel (shared code); expect 1.00x")
    p.add_argument("--gap-open", type=float, default=11.0)
    p.add_argument("--gap-extend", type=float, default=1.0)
    p.add_argument("--repeats", type=int, default=5,
                   help="interleaved rounds; the MINIMUM of these is reported")
    p.add_argument("--min-time", type=float, default=2.0,
                   help="seconds per measurement; batch size auto-scales to reach it")
    p.add_argument("--max-n", type=int, default=200_000,
                   help="cap on auto-scaled batch size")
    p.add_argument("--seed", type=int, default=20260714)
    p.add_argument("--quick", action="store_true",
                   help="fewer repeats and shorter runs; rough numbers only")
    p.add_argument("--json", metavar="FILE", help="also write raw results here")
    p.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.quick:
        args.repeats = 3
        args.min_time = 0.5

    if args.threads is None:
        cores = os.cpu_count() or 1
        args.threads = [1] if cores == 1 else [1, cores]

    if args._worker:
        worker(args)
        return

    isas = probe_isas()

    print()
    print(f"  host      {platform.node()}")
    print(f"  cpu       {cpu_model()}")
    print(f"  cores     {os.cpu_count()}")
    print(f"  ISAs      {' '.join(isas)}   (probed, not assumed)")
    print(f"  workload  {args.gap_model}/{args.mode}/{args.grad_mode}"
          f"  seq_len={args.seq_len}  threads={args.threads}")
    print(f"  method    min of {args.repeats} interleaved rounds, "
          f">={args.min_time}s each, one process per ISA")
    if args.gap_model == "linear":
        print("  note      linear has no simd kernel by design — 1.00x is the correct result")
    if args.grad_mode == "soft":
        print("  note      soft gradients ignore the kernel (shared code) — 1.00x is correct")
    print()

    rows = []
    for isa in isas:
        print(f"  running {isa} ...", end="", flush=True)
        t0 = time.perf_counter()
        rows.extend(run_isa(isa, args))
        print(f" {time.perf_counter() - t0:.0f}s")
    print()

    # ── table ────────────────────────────────────────────────────────────────
    hdr = (f"  {'ISA':<9} {'thr':>4} {'batch':>7} "
           f"{'scalar':>12} {'simd':>12} {'speedup':>8}  {'control':>8}  {'exact':<5}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        warn = "" if 0.97 <= r["control_ratio"] <= 1.03 else "  <-- SUSPECT"
        print(f"  {r['isa']:<9} {r['threads']:>4} {r['n']:>7} "
              f"{r['scalar_pairs_s']:>9,.0f}/s {r['simd_pairs_s']:>9,.0f}/s "
              f"{r['speedup']:>7.2f}x  {r['control_ratio']:>7.2f}x  "
              f"{'yes' if r['exact'] else 'NO!!':<5}{warn}")
    print()

    # ── the checks that decide whether any of the above may be believed ──────
    bad = [r for r in rows if not r["exact"]]
    if bad:
        print("  FAIL: the two kernels produced DIFFERENT results.")
        print("        They are supposed to be bit-exact.  A speedup obtained by")
        print("        computing something else is not a speedup — treat every number")
        print("        above as void and investigate before doing anything with it.")
        return 1

    drift = [r for r in rows if not 0.97 <= r["control_ratio"] <= 1.03]
    if drift:
        print("  WARNING: the control arm (scalar measured against scalar) did not come")
        print("           out at 1.00x, so this harness is partly measuring itself —")
        print("           machine load, thermal drift, or a noisy neighbour.  The speedups")
        print("           above are unreliable to roughly the same degree the control is")
        print("           off.  Re-run on an idle machine, or raise --min-time/--repeats.")
        print()
    else:
        print("  control arm is clean (1.00x): the harness is not biasing the comparison.")

    best = max(rows, key=lambda r: r["speedup"])
    print(f"  best: {best['speedup']:.2f}x on {best['isa']} at {best['threads']} thread(s)"
          f"  ({best['scalar_mcells_s']:.0f} -> {best['simd_mcells_s']:.0f} Mcell/s)")
    print()

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"host": platform.node(), "cpu": cpu_model(),
                       "cores": os.cpu_count(), "args": vars(args), "rows": rows}, f, indent=2)
        print(f"  raw results -> {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
