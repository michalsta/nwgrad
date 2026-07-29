"""Which compiler should build the wheel for THIS machine's ISA?

`.github/workflows/wheels.yml` pins x86_64 Linux to clang and aarch64/macOS to gcc.  That
policy rests on a measurement taken on the PROTOTYPE striped kernel (variant A, W=4,
len=200, forward-only) — before striping shipped, before the 64-byte allocator, before the
Hirschberg sweeps were vectorized, before hb_base was vectorized, and before the prefix-max
carry existed and became the float32 default.  Every one of those changed the code the
compiler is being judged on.  This re-measures it against what is actually shipped.

Run the SAME script against two builds of the same tree (CC/CXX=gcc, then clang) and
compare.  `tools/bench_toolchain.sh` does the whole dance; this script is one half of it.

METHODOLOGY — the trap here is real and CLAUDE.md names it: comparing separately-compiled
binaries produced code-layout (µop-cache) artifacts LARGER than the effect under test, which
is why tools/bench_simd.py deliberately compiles nothing and switches arms at runtime
inside one process.  We cannot do that here — two compilers is the whole question, so two
binaries is unavoidable.  Three things are done about it instead:

  * A CONTROL ARM: `scalar_ctl` runs the same DP through kernel="scalar_fallback", which
    contains no std::simd at all.  Its clang/gcc ratio is the compiler's GENERAL codegen
    difference on this code.  Every simd arm should be read RELATIVE to it — an arm that
    moves exactly as much as the control says nothing about std::simd.
  * best-of-N, not mean: layout noise and scheduler interference are one-sided (they only
    ever make a run slower), so the minimum is the cleanest estimator of the code's speed.
  * Every arm is a real shipped path, at a realistic size, not a microbenchmark — the
    effect being measured has to survive contact with the actual workload or it is not
    worth a shipping decision.

ARMS (all affine+global+full, hard gradient, native ISA unless stated):

    hb_pmax_f32   long pairs, traceback="auto" at float32  -> the prefix-max sweep.
                  THE DEFAULT PATH for the default dtype: the single most important arm.
    hb_exact_f32  long pairs, traceback="hirschberg"       -> the exact lazy-F sweep.
    hb_exact_f64  long pairs, double                       -> the double default.
    hb_base_f32   pairs BELOW hb_cutoff                    -> the striped hb_base fill,
                  never splits.  Most real proteins land here, so it carries more weight
                  than its runtime suggests.
    ptr_f32       long pairs, traceback="pointers"         -> the striped Full kernel
                  (still the Local default and what `scores`/`pointers` users get).
    scalar_ctl    the CONTROL, kernel="scalar_fallback".
"""
import argparse, gc, json, sys, time
import numpy as np
import nwgrad
from nwgrad.matrices import BLOSUM62

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--threads", type=int, default=1,
                help="1 isolates CODEGEN; higher counts add memory-bandwidth effects that "
                     "belong to the machine, not the compiler")
ap.add_argument("--reps", type=int, default=3, help="timed repeats; the BEST is reported")
ap.add_argument("--long-len", type=int, default=2000)
ap.add_argument("--long-n", type=int, default=12)
ap.add_argument("--short-len", type=int, default=400, help="must be < cutoff to hit hb_base")
ap.add_argument("--short-n", type=int, default=120)
ap.add_argument("--cutoff", type=int, default=512)
ap.add_argument("--rate", type=float, default=0.3,
                help="mutation rate; related pairs are the realistic case AND the one the "
                     "pmax/lazy-F difference is sensitive to")
ap.add_argument("--json", type=str, default="", help="write results here for the comparer")
args = ap.parse_args()

AA = list(BLOSUM62.alphabet)
params = nwgrad.AlignParams(BLOSUM62, gap_open_a=11.0, gap_extend_a=1.0,
                            gap_open_b=11.0, gap_extend_b=1.0)


def related_pairs(n, L, rate, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        a = "".join(rng.choice(AA, L))
        arr = np.frombuffer(a.encode(), np.uint8).copy()
        hit = rng.random(L) < rate
        if hit.any():
            sub = "".join(rng.choice(AA, int(hit.sum()))).encode()
            arr[hit] = np.frombuffer(sub, np.uint8)
        out.append((a, arr.tobytes().decode()))
    return out


LONG = related_pairs(args.long_n, args.long_len, args.rate, 3)
SHORT = related_pairs(args.short_n, args.short_len, args.rate, 4)


def run(pairs, tb, dtype, kernel="auto"):
    cls = nwgrad.SeqPairBatchDouble if dtype == "double" else nwgrad.SeqPairBatch
    b = cls(n_threads=args.threads, traceback=tb)
    if tb.startswith("hirschberg") or tb == "auto":
        try:
            b.hb_cutoff = args.cutoff
        except AttributeError:
            pass
    b.add_many([x for x, _ in pairs], [y for _, y in pairs], params,
               gap_model="affine", mode="global", grad_mode="hard", kernel=kernel)
    b.score_and_grad()                      # warm up / fault in
    best = float("inf")
    for _ in range(args.reps):
        t = time.perf_counter()
        b.score_and_grad()
        best = min(best, time.perf_counter() - t)
    del b
    gc.collect()
    return best


ARMS = [
    ("hb_pmax_f32",  LONG,  "auto",       "float",  "auto"),
    ("hb_exact_f32", LONG,  "hirschberg", "float",  "auto"),
    ("hb_exact_f64", LONG,  "hirschberg", "double", "auto"),
    ("hb_base_f32",  SHORT, "auto",       "float",  "auto"),
    ("ptr_f32",      LONG,  "pointers",   "float",  "auto"),
    ("scalar_ctl",   LONG,  "auto",       "float",  "scalar_fallback"),
]

cells = {id(LONG): float(sum((len(a) + 1) * (len(b) + 1) for a, b in LONG)),
         id(SHORT): float(sum((len(a) + 1) * (len(b) + 1) for a, b in SHORT))}

res = {"isa": nwgrad.simd_isa(), "compiler": nwgrad.compiled_with(),
       "threads": args.threads, "arms": {}}
print(f"isa={res['isa']}  built_with={res['compiler']}  threads={args.threads}  "
      f"reps={args.reps}  cutoff={args.cutoff}")
print(f"{'arm':>14s} {'secs':>10s} {'Mcell/s':>10s}")
for name, pairs, tb, dtype, kern in ARMS:
    s = run(pairs, tb, dtype, kern)
    mc = cells[id(pairs)] / s / 1e6
    res["arms"][name] = {"secs": s, "mcells": mc}
    print(f"{name:>14s} {s:10.4f} {mc:10.1f}")

if args.json:
    with open(args.json, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"\nwrote {args.json}", file=sys.stderr)
