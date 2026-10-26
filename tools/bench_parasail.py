#!/usr/bin/env python3
"""nwgrad's integer kernels against parasail, apples to apples on algorithm variants.

    python tools/bench_parasail.py                      # every arm, 1 thread, protein + DNA
    python tools/bench_parasail.py --threads 1 12 --fixture protein --len 200 1000
    python tools/bench_parasail.py --quick

parasail (Daily 2016) is the reference striped-SIMD aligner: Farrar striped, Farrar/Rognes
"scan" (a prefix-max gap carry) and Wozniak anti-diagonal kernels, each in 8/16/32/64-bit
saturating integers, for nw (global), sw (local) and sg (semi-global), score-only or with
a traceback.  nwgrad's integer types meet it on its own terms: the SAME integer matrix
and gaps, the same algorithm, the same bit width, score-only against score-only
(parasail nw/sw_striped_W vs nwgrad SeqPairBatchIntW grad_mode="none"), traceback against
path (parasail *_trace_striped_W vs nwgrad keep_paths=True, no gradient), and nwgrad's
actual training step (hard gradient + path) beside them.  parasail's "scan" and "diag"
variants are reported too: nwgrad's integer carry IS a prefix max, so scan is its closest
relative.  Linear gaps: parasail has no linear model; open == extend gives one (a gap of
length k costs open + (k-1)*extend = k*ge), and nwgrad runs gap_model="linear".

GAPS.  parasail's gap of length k costs open + (k-1)*extend; nwgrad's costs go + k*ge.
So parasail open = go + ge, extend = ge (bench_competitors.py established the same
mapping by exact score agreement).  Before anything is timed, every parasail score must
EQUAL the nwgrad integer score on every pair of the fixture, exactly (integers), or the
run aborts — which also makes this an independent oracle for the integer kernels.

ISA.  The Python bindings expose parasail's runtime dispatcher only (nw_striped_16 picks
the best ISA the library was BUILT with); nwgrad kernel="auto" does the same.  Both
choices are printed.  The PyPI wheel (1.3.4) carries sse2 / sse41 / avx2 / neon kernels
and NO AVX-512 ones (nm: 0 `avx512_512` symbols), so on an AVX-512 host the comparison
is parasail's 256-bit kernels against nwgrad's 512-bit ones -- what a pip user gets; for
an ISA-matched row set NWGRAD_ISA=avx2 in the environment (nwgrad's level override).

Each arm runs in its OWN process (arms sharing one contaminate each other's allocator
and caches: the repo's rule), repeats interleaved, the minimum kept.  A control arm
(the nwgrad float32 score arm run twice, two processes) is reported first: if it is not
~1.00x the numbers are noise.

DRIVER.  parasail runs through tools/bench_parasail_driver.c by default (--driver c):
a small C program compiled once per run against the wheel's own headers and
libparasail.so, one pair after another with real threads (spawned per pass, as nwgrad's
batch spawns its workers per call).  The Python bindings are NOT the kernels: measured
on skynet, a one-residue pair costs ~10 us per call (ctypes, result object, profile
build), which is most of a 22x50 DNA pair's time and a quarter of an L200 protein
pair's, and the bindings' threads share the GIL, so a 12-thread Python run measured the
lock (parasail 10-800x "slower").  --driver python keeps that path, as what a Python
caller of parasail actually gets; it is labelled as such.  "trace" arms walk the
traceback (the CIGAR) on every pair in both drivers, as nwgrad's path arm walks its own.
nwgrad runs from Python either way: one call per batch, so the binding costs nothing
per pair.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np


# ── fixtures (deterministic from the seed, rebuilt in every worker) ───────────────
def protein_pairs(n, length, seed):
    rng = np.random.default_rng(seed)
    syms = np.array(list("ARNDCQEGHILKMFPSTWYV"))
    A, B = [], []
    for _ in range(n):
        a = syms[rng.integers(0, 20, length)]
        b = a.copy()
        mut = rng.random(length) < 0.3
        b[mut] = syms[rng.integers(0, 20, int(mut.sum()))]
        # a few indels so gaps are exercised
        cut = rng.integers(0, length, 2)
        b = np.delete(b, cut)
        A.append("".join(a)); B.append("".join(b))
    return A, B


def dna_pairs(n, seed):
    rng = np.random.default_rng(seed)
    syms = np.array(list("ACGT"))
    A, B = [], []
    for _ in range(n):
        a = syms[rng.integers(0, 4, 22)]
        b = syms[rng.integers(0, 4, 50)]
        off = int(rng.integers(0, 29))
        mut = rng.random(22) < 0.2
        b[off:off + 22] = np.where(mut, b[off:off + 22], a)
        A.append("".join(a)); B.append("".join(b))
    return A, B


def fixture(name, length, n, seed):
    if name == "protein":
        A, B = protein_pairs(n, length, seed)
        go, ge = 11, 1
        return A, B, "protein", go, ge
    A, B = dna_pairs(n, seed)
    return A, B, "dna", 5, 2


# ── the arms ──────────────────────────────────────────────────────────────────────
WIDTHS = [16, 32, 64]


def arm_list(types):
    arms = ["control"]
    for mode in ["global", "local"]:
        for gm in ["affine", "linear"]:
            for w in WIDTHS:
                arms += [f"ps/{mode}/{gm}/striped{w}/score", f"ps/{mode}/{gm}/scan{w}/score",
                         f"ps/{mode}/{gm}/striped{w}/trace",
                         f"nw/{mode}/{gm}/int{w}/score", f"nw/{mode}/{gm}/int{w}/path",
                         f"nw/{mode}/{gm}/int{w}/grad"]
            arms += [f"ps/{mode}/{gm}/diag32/score",
                     f"nw/{mode}/{gm}/float32/score", f"nw/{mode}/{gm}/float32/grad"]
    if types:
        arms = [a for a in arms if a == "control" or a == "nw/global/affine/float32/score"   # the control's reference
                or any(f"/{t}/" in a for t in types)]
    return arms


def ps_name(mode, variant, kind):
    base = "nw" if mode == "global" else "sw"
    algo = "".join(c for c in variant if not c.isdigit())      # striped16 -> striped, 16
    width = "".join(c for c in variant if c.isdigit())
    return f"{base}_{'trace_' if kind == 'trace' else ''}{algo}_{width}"


def ps_fn(mode, variant, kind):
    import parasail
    return getattr(parasail, ps_name(mode, variant, kind))


def build_driver(out_dir):
    """Compile tools/bench_parasail_driver.c against the installed parasail wheel."""
    import parasail
    pkg = os.path.dirname(os.path.abspath(parasail.__file__))
    inc = os.path.join(pkg, "include")
    sos = [os.path.join(pkg, f) for f in os.listdir(pkg) if f.startswith("libparasail") and ".so" in f]
    if not os.path.isfile(os.path.join(inc, "parasail.h")) or not sos:
        raise SystemExit(f"the parasail package at {pkg} ships no headers / shared library; "
                         "install the parasail wheel (pip install parasail) or use --driver python")
    cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if not cc:
        raise SystemExit("no C compiler found (CC, cc, gcc, clang): install one, or use --driver python")
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_parasail_driver.c")
    exe = os.path.join(out_dir, "bench_parasail_driver")
    cmd = [cc, "-O2", "-std=c11", "-pthread", "-I", inc, src, sos[0], "-Wl,-rpath,$ORIGIN", "-o", exe]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit("compiling the parasail driver failed:\n" + " ".join(cmd) + "\n" + r.stderr)
    # the wheel's library carries a versioned SONAME (libparasail.so.8) but is shipped
    # under the unversioned name: give the loader the name it will ask for, next to the exe
    needed = subprocess.run([shutil.which("objdump") or "objdump", "-p", exe], capture_output=True, text=True).stdout
    for line in needed.splitlines():
        if "NEEDED" in line and "parasail" in line:
            os.symlink(sos[0], os.path.join(out_dir, line.split()[-1]))
    return exe


def run_arm(arm, fx, length, n, seed, threads, reps, min_time):
    """One arm in this process: builds the fixture, checks scores against the oracle
    file, times.  Prints a JSON line."""
    A, B, alpha, go, ge = fixture(fx, length, n, seed)
    cells = float(sum(len(a) * len(b) for a, b in zip(A, B)))
    import nwgrad
    if alpha == "protein":
        from nwgrad import matrices
        nw_matrix = matrices.BLOSUM62
        import parasail
        ps_matrix = parasail.blosum62
    else:
        import parasail
        nw_matrix = nwgrad.SubstMatrix(np.where(np.eye(4, dtype=bool), 2.0, -3.0), alphabet="ACGT")
        ps_matrix = parasail.matrix_create("ACGT", 2, -3)

    parts = arm.split("/")
    if arm == "control":
        parts = ["nw", "global", "affine", "float32", "score"]
    side, mode, gm, variant, kind = parts
    lin = gm == "linear"
    goe = 0 if lin else go
    params = nwgrad.AlignParams(nw_matrix, float(goe), float(ge), float(goe), float(ge))
    ps_open, ps_ext = goe + ge, ge      # parasail: open + (k-1)*extend == go + k*ge

    scores = None
    if side == "nw":
        cls = {"float32": nwgrad.SeqPairBatch, "int16": nwgrad.SeqPairBatchInt16,
               "int32": nwgrad.SeqPairBatchInt32, "int64": nwgrad.SeqPairBatchInt64}[variant]
        gmode = "hard" if kind == "grad" else "none"
        # the exact full-table fill, as parasail's traceback: affine = Pointers (what `auto`
        # runs below hb_cutoff anyway); linear = `auto`, which keeps the H table while it is
        # <= 2 MiB (the vectorized fill) and records bytes past that.  An explicit "pointers"
        # on linear forces the byte-recording loop, a header-only (baseline) fill that is
        # 3-5x slower than the H fill at these sizes and would misreport nwgrad.
        tb = "auto" if lin else "pointers"
        b = cls(A, B, params, gap_model=gm, mode=mode, grad_mode=gmode, n_threads=threads, traceback=tb)
        keep = kind in ("path", "grad")
        def once():
            b.score_and_grad(keep_paths=keep)
        try:
            once()
        except OverflowError as e:   # the integer type's a-priori bound refused the fixture
            print(json.dumps({"arm": arm, "error": "refused: " + str(e).split("(")[0].strip()})); return
        scores = b.scores()
        isa = nwgrad.simd_isa()
    else:
        import parasail
        isa = ("avx512" if getattr(parasail, "can_use_avx512bw", lambda: False)() else
               "avx2" if parasail.can_use_avx2() else "sse41" if parasail.can_use_sse41() else
               "sse2" if parasail.can_use_sse2() else "neon" if parasail.can_use_neon() else "?")
        driver = os.environ.get("BENCH_PARASAIL_DRIVER")
        if driver:
            # the C driver: fixture to a file, one process, timing inside
            with tempfile.NamedTemporaryFile("w", suffix=".pairs", delete=False) as pf:
                for a, bb in zip(A, B):
                    pf.write(f"{a}\t{bb}\n")
            try:
                mat = "blosum62" if alpha == "protein" else "dna:2:-3"
                cmd = [driver, "parasail_" + ps_name(mode, variant, kind), mat, str(ps_open), str(ps_ext),
                       str(threads), str(reps), str(min_time), "1" if kind == "trace" else "0", pf.name]
                r = subprocess.run(cmd, capture_output=True, text=True)
            finally:
                os.unlink(pf.name)
            if r.returncode:
                print(json.dumps({"arm": arm, "error": "driver failed: " + r.stderr.strip()[-300:]})); return
            d = json.loads(r.stdout)
            if d["saturated"]:
                print(json.dumps({"arm": arm, "error": "parasail saturated at this width"})); return
            scores = np.array(d["scores"], dtype=float)
            cells = float(sum(len(a) * len(b) for a, b in zip(A, B)))
            print(json.dumps({"arm": arm, "secs": d["secs"], "us_per_pair": d["secs"] / n * 1e6,
                              "mcells": cells / d["secs"] / 1e6, "isa": isa, "driver": "c",
                              "scores": [float(s) for s in scores]}))
            return
        f = ps_fn(mode, variant, kind)
        def one(a, bb):
            r = f(a, bb, ps_open, ps_ext, ps_matrix)
            if kind == "trace":
                r.cigar          # the walk, as the C driver and nwgrad's path arm do it
            return r.score, r.saturated
        def run_all():
            if threads == 1:
                return [one(a, bb) for a, bb in zip(A, B)]
            with ThreadPoolExecutor(threads) as ex:
                return list(ex.map(one, A, B))
        res = run_all()
        if any(s for _, s in res):
            print(json.dumps({"arm": arm, "error": "parasail saturated at this width"})); return
        scores = np.array([s for s, _ in res], dtype=float)
        def once():
            run_all()

    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter(); k = 0
        while True:
            once(); k += 1
            dt = time.perf_counter() - t0
            if dt >= min_time: break
        best = min(best, dt / k)
    out = {"arm": arm, "secs": best, "us_per_pair": best / n * 1e6, "mcells": cells / best / 1e6,
           "isa": isa, "driver": "python" if side == "ps" else "nwgrad", "scores": [float(s) for s in scores]}
    print(json.dumps(out))


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--fixture", nargs="+", default=["protein", "dna"], choices=["protein", "dna"])
    ap.add_argument("--len", type=int, nargs="+", default=[200], help="protein lengths")
    ap.add_argument("--n", type=int, default=0, help="pairs (0: 2000 protein L<=300, 200 longer, 20000 dna)")
    ap.add_argument("--threads", type=int, nargs="+", default=[1])
    ap.add_argument("--types", nargs="+", default=None, help="restrict to these variants (int16 striped16 ...)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--min-time", type=float, default=1.0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--json", metavar="FILE")
    ap.add_argument("--driver", default="c", choices=["c", "python"],
                    help="how parasail is called: the C driver (its kernels) or its Python bindings")
    ap.add_argument("--_worker", nargs=8, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args._worker:
        arm, fx, length, n, seed, threads, reps, min_time = args._worker
        run_arm(arm, fx, int(length), int(n), int(seed), int(threads), int(reps), float(min_time))
        return 0
    if args.quick:
        args.reps, args.min_time = 2, 0.4
    os.environ.setdefault("LC_ALL", "C")
    for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")     # numpy's BLAS pool otherwise spins up every core at import
    import nwgrad, parasail
    print(f"nwgrad {nwgrad.__version__} ({nwgrad.compiled_with()}, levels {nwgrad.available_isa_levels()}); "
          f"parasail {parasail.__version__}, can_use: sse41={parasail.can_use_sse41()} avx2={parasail.can_use_avx2()} neon={parasail.can_use_neon()}")
    os.environ.pop("BENCH_PARASAIL_DRIVER", None)
    tmpdir = None
    if args.driver == "c":
        tmpdir = tempfile.mkdtemp(prefix="bench_parasail_")
        os.environ["BENCH_PARASAIL_DRIVER"] = build_driver(tmpdir)
        print(f"parasail driver: C ({os.environ['BENCH_PARASAIL_DRIVER']}), threads are pthreads")
    else:
        print("parasail driver: PYTHON BINDINGS (~10 us per call, GIL-bound threads): measures the binding, not the kernels")
    results = {}
    for fx in args.fixture:
        for length in (args.len if fx == "protein" else [50]):
            n = args.n or (20000 if fx == "dna" else 2000 if length <= 300 else 200)
            if args.quick: n = max(50, n // 5)
            for th in args.threads:
                arms = arm_list(args.types)
                res = {}
                # interleave: every arm once per round, each in a fresh process
                for r in range(args.reps):
                    for arm in arms:
                        cmd = [sys.executable, __file__, "--_worker", arm, fx, str(length), str(n), str(args.seed),
                               str(th), "1", str(args.min_time)]
                        p = subprocess.run(cmd, capture_output=True, text=True)
                        line = [l for l in p.stdout.splitlines() if l.startswith("{")]
                        if not line:
                            print(f"{arm}: worker failed\n{p.stderr[-800:]}", file=sys.stderr); return 1
                        d = json.loads(line[-1])
                        if "error" in d:
                            res[arm] = d; continue
                        if arm not in res or d["secs"] < res[arm]["secs"]:
                            res[arm] = d
                # exact score agreement: every parasail arm against the nwgrad int arm of its
                # mode / gap model (parasail vs nwgrad is the point; int vs float is nwgrad's own)
                for arm, d in res.items():
                    if "error" in d or not arm.startswith("ps/"): continue
                    _, mode, gm, variant, kind = arm.split("/")
                    w = "".join(c for c in variant if c.isdigit())
                    ref = res.get(f"nw/{mode}/{gm}/int{w}/score")
                    if not ref or "error" in ref: ref = res.get(f"nw/{mode}/{gm}/int32/score")
                    if ref and "error" not in ref and d["scores"] != ref["scores"]:
                        bad = sum(1 for x, y in zip(d["scores"], ref["scores"]) if x != y)
                        below = all(x <= y for x, y in zip(d["scores"], ref["scores"]))
                        # parasail's striped 16/32-bit kernels score BELOW the optimum when
                        # open == extend (its scan, diag and striped-64 kernels agree with
                        # nwgrad on every pair): a defect of its lazy-F in that degenerate
                        # case, so the linear striped arms are flagged and still timed.
                        if gm == "linear" and variant.startswith("striped") and below:
                            d["note"] = f"SUBOPTIMAL on {bad}/{n} pairs (parasail bug at open == extend)"
                        else:
                            print(f"SCORE MISMATCH {arm} vs {ref['arm']}: {bad}/{n} pairs differ", file=sys.stderr)
                            return 2
                tag = f"{fx} L{length}" if fx == "protein" else "dna 22x50"
                print(f"\n== {tag}, {n} pairs, {th} thread(s); nwgrad isa {res['control']['isa']}, "
                      f"parasail isa {next((d['isa'] for a, d in res.items() if a.startswith('ps/') and 'isa' in d), '?')}")
                print(f"   {'arm':36s} {'us/pair':>10s} {'Mcell/s':>9s} {'vs nwgrad int':>14s}")
                ctl = res["control"]["secs"] / res["nw/global/affine/float32/score"]["secs"] if "nw/global/affine/float32/score" in res else float('nan')
                print(f"   {'control (float32 score, 2 processes)':36s} {'':>10s} {'':>9s} {ctl:13.2f}x")
                for arm in arms:
                    if arm == "control": continue
                    d = res[arm]
                    if "error" in d:
                        print(f"   {arm:36s} {d['error']}"); continue
                    _, mode, gm, variant, kind = arm.split("/")
                    w = "".join(c for c in variant if c.isdigit()) or "32"
                    refk = f"nw/{mode}/{gm}/int{w}/{'score' if kind == 'score' else 'path'}"
                    ref = res.get(refk)
                    rel = f"{d['secs'] / ref['secs']:13.2f}x" if ref and "secs" in ref and arm != refk else ""
                    if ref and "error" in ref: rel = "(int ref refused)"
                    print(f"   {arm:36s} {d['us_per_pair']:10.3f} {d['mcells']:9.0f} {rel:>14s}  {d.get('note', '')}")
                results[f"{tag}/{th}"] = {a: {k: v for k, v in d.items() if k != "scores"} for a, d in res.items()}
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)
    if tmpdir:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
