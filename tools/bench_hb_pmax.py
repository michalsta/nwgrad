"""Is the prefix-max carry actually faster than the exact Hirschberg sweep?

That is the load-bearing unknown behind `traceback="hirschberg_pmax"`.  The exact sweep
walks a serial `max(open, VY[c-1] - ge_a)` chain and then repairs the lane boundaries with
lazy-F; the pmax sweep replaces both with a max-only prefix scan plus W scalar maxes.  It
trades roughly two extra streaming passes over the row for a shorter dependency chain and
no correction rounds — so it wins only if that chain was the ceiling.

Three arms, all global affine, hard gradient, at the host's native ISA:

    pointers          the O(m*n) reference
    hirschberg        the exact linear-space sweep (today's default for this case)
    hirschberg_pmax   the same, closed-form carry

The number that decides it is pmax/hb — below 1.00 means pmax is faster.  Everything else
is context.

THE FIXTURE IS NOT A DETAIL HERE.  Lazy-F is a FIXPOINT: it sweeps the row again and
again until no gap extension still improves a cell, so its cost is DATA-DEPENDENT.  The
more of the DP rectangle is won by long gap runs, the more rounds it pays.  The prefix-max
carry has no fixpoint and no data dependence — it does the same fixed work per row on
every input.  So the two arms are not separated by a constant factor; the gap between them
is a function of how gap-dominated the problem is:

  * `homology` — pairs at a sweep of mutation rates, from identical to unrelated.  This is
    the fixture that shows the mechanism: hold length fixed, vary only relatedness.
    Diagonal-dominant (related) pairs make the off-diagonal DP a field of long gap chains,
    which is exactly what lazy-F re-sweeps for.
  * `proteome` — real human proteins, self-pairs.  What users actually run, and firmly at
    the related end of the range above.
  * `skew`     — synthetic pairs with |a| << |b|, forcing very long gap-in-a runs.

Vector width matters to the comparison and is not something this script can vary: lazy-F
costs up to W correction rounds per row, so the wider the ISA the more pmax stands to
save.  Run it on every host in the fleet before drawing a conclusion.
"""
import argparse, gc, time
import numpy as np
import nwgrad
from nwgrad.matrices import BLOSUM62

FASTA = "GCF_000001405.40_GRCh38.p14_protein.faa"
ARMS = ["pointers", "hirschberg", "hirschberg_pmax"]

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--fixture", choices=["proteome", "skew", "homology", "all"], default="all",
                nargs="+")
ap.add_argument("--min-len", type=int, default=512)
ap.add_argument("--max-len", type=int, default=10000)
ap.add_argument("--threads", type=int, nargs="+", default=[1, 4, 16, 30, 60])
ap.add_argument("--cutoff", type=int, default=512)
ap.add_argument("--limit", type=int, default=400, help="subsample to N sequences (0 = all)")
ap.add_argument("--reps", type=int, default=2, help="timed repeats; the best is reported")
ap.add_argument("--skew-lens", type=int, nargs=2, default=[400, 6000],
                metavar=("SHORT", "LONG"), help="skew fixture pair shape")
ap.add_argument("--skew-n", type=int, default=64)
ap.add_argument("--hom-len", type=int, default=2000)
ap.add_argument("--hom-n", type=int, default=24)
ap.add_argument("--hom-rates", type=float, nargs="+", default=[0.0, 0.1, 0.3, 0.6, 1.0],
                help="mutation rate; 0.0 = identical self-pair, 1.0 = unrelated")
ap.add_argument("--no-oracle", action="store_true")
args = ap.parse_args()

AA = list(BLOSUM62.alphabet)
params = nwgrad.AlignParams(BLOSUM62, gap_open_a=11.0, gap_extend_a=1.0,
                            gap_open_b=11.0, gap_extend_b=1.0)


def load_proteome(path):
    allowed = set(BLOSUM62.alphabet)
    tomat = str.maketrans({"U": "C", "O": "X"})
    seqs, chunks = [], []
    with open(path) as fh:
        for line in fh:
            if line[0] == ">":
                if chunks: seqs.append("".join(chunks)); chunks = []
            else:
                chunks.append(line.strip())
    if chunks: seqs.append("".join(chunks))
    out = []
    for s in seqs:
        s = s.upper().translate(tomat)
        if not allowed.issuperset(s):
            s = "".join(c if c in allowed else "X" for c in s)
        out.append(s)
    return out


def proteome_pairs():
    seqs = [s for s in load_proteome(FASTA)
            if args.min_len <= len(s) <= args.max_len]
    if args.limit and len(seqs) > args.limit:
        rng = np.random.default_rng(0)
        seqs = [seqs[i] for i in sorted(rng.choice(len(seqs), args.limit, replace=False))]
    return list(zip(seqs, seqs))          # self-pairs


def skew_pairs():
    # |a| << |b| forces the optimal path through very long gap-in-a runs, which is the
    # regime where lazy-F pays the most correction rounds.
    rng = np.random.default_rng(1)
    short, long = args.skew_lens
    return [(''.join(rng.choice(AA, short)), ''.join(rng.choice(AA, long)))
            for _ in range(args.skew_n)]


def homology_pairs(rate, rng):
    """N pairs of fixed length differing by `rate` substitutions.  rate=0 is a self-pair
    (maximally diagonal-dominant); rate=1 is two unrelated random sequences."""
    L = args.hom_len
    out = []
    for _ in range(args.hom_n):
        a = ''.join(rng.choice(AA, L))
        if rate >= 1.0:
            b = ''.join(rng.choice(AA, L))
        else:
            arr = np.frombuffer(a.encode(), dtype=np.uint8).copy()
            hit = rng.random(L) < rate
            if hit.any():
                sub = ''.join(rng.choice(AA, int(hit.sum()))).encode()
                arr[hit] = np.frombuffer(sub, dtype=np.uint8)
            b = arr.tobytes().decode()
        out.append((a, b))
    return out


def build(pairs, tb, threads):
    b = nwgrad.SeqPairBatch(n_threads=threads, traceback=tb)
    if tb.startswith("hirschberg"):
        b.hb_cutoff = args.cutoff
    b.add_many([x for x, _ in pairs], [y for _, y in pairs], params,
               gap_model="affine", mode="global", grad_mode="hard", kernel="auto")
    return b


def oracle(b, pairs):
    """Self-pairs align to the diagonal: score = sum of the matched BLOSUM entries.
    Cheap, and it re-verifies every arm on exactly the data being timed."""
    M = np.asarray(BLOSUM62.to_matrix()); diag = np.diag(M)
    lut = np.zeros(256, dtype=np.uint8)
    for i, c in enumerate(BLOSUM62.alphabet):
        lut[ord(c)] = i
    bad = 0
    for i, (a, _) in enumerate(pairs):
        want = float(np.bincount(lut[np.frombuffer(a.encode(), dtype=np.uint8)],
                                 minlength=len(diag)) @ diag)
        if b[i].score != want:
            bad += 1
    return bad


def run_fixture(name, pairs):
    cells = float(sum((len(a) + 1) * (len(b) + 1) for a, b in pairs))
    lens = np.array([len(b) for _, b in pairs])
    print(f"\n### {name}: {len(pairs)} pairs, |b| {lens.min()}-{lens.max()} "
          f"(mean {lens.mean():.0f}), {cells / 1e9:.3f} Gcells/pass, "
          f"cutoff={args.cutoff}, isa={nwgrad.simd_isa()}")
    self_paired = all(a == b for a, b in pairs)

    print(f"{'threads':>7s} " + " ".join(f"{a:>12s}" for a in ARMS) +
          f" {'pmax/hb':>9s} {'hb/ptr':>8s} {'ctl':>6s}   Mcell/s")
    for th in args.threads:
        secs, nt = {}, None
        # Control arm: the exact sweep timed a second time, LAST, after both others.  If
        # it does not come back at ~1.00x the first run then run order, thermal state or
        # background load is moving the numbers and nothing else in the row is safe to read.
        for arm in ARMS + ["hirschberg"]:
            b = build(pairs, arm, th)
            b.score_and_grad()                       # warm up / fault in
            if self_paired and not args.no_oracle:
                bad = oracle(b, pairs)
                if bad:
                    print(f"  *** {arm}: {bad}/{len(pairs)} self-pairs WRONG ***")
            best = float("inf")
            for _ in range(args.reps):
                t = time.perf_counter(); b.score_and_grad()
                best = min(best, time.perf_counter() - t)
            # the repeat of "hirschberg" lands in the control slot, not over the first run
            secs["control" if arm in secs else arm] = best
            nt = b.n_threads
            del b; gc.collect()
        mc = {a: cells / secs[a] / 1e6 for a in ARMS}
        print(f"{nt:7d} " + " ".join(f"{secs[a]:11.3f}s" for a in ARMS) +
              f" {secs['hirschberg_pmax'] / secs['hirschberg']:8.3f}x"
              f" {secs['hirschberg'] / secs['pointers']:7.3f}x"
              f" {secs['control'] / secs['hirschberg']:5.2f}x   " +
              " ".join(f"{mc[a]:7.1f}" for a in ARMS))


want = set(args.fixture)
if "all" in want:
    want = {"homology", "proteome", "skew"}
if "homology" in want:
    rng = np.random.default_rng(4)
    for rate in args.hom_rates:
        label = "identical" if rate == 0 else ("unrelated" if rate >= 1 else f"{rate:.0%} mutated")
        run_fixture(f"homology {args.hom_len}aa, {label}", homology_pairs(rate, rng))
if "proteome" in want:
    run_fixture("proteome self-pairs", proteome_pairs())
if "skew" in want:
    run_fixture(f"skew {args.skew_lens[0]}x{args.skew_lens[1]} (long gap runs)", skew_pairs())
