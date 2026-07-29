"""Local (Smith-Waterman) Hirschberg vs Pointers: thread-scaling sweep + oracle.

Self-pairs from the human proteome, full square DP, hard gradient, mode=local.  Self-
pairs are the WORST case for local Hirschberg (the optimal local alignment is the whole
diagonal, so the box spans the full matrix and HB pays two full endpoint scans PLUS the
full recursion, ~4x Pointers' cells) — so a win here is a conservative win.

Filtered to sequences >= --min-len because that is the only regime where the choice
matters: a pair no longer than hb_cutoff never splits and runs AS Pointers (bit-exact,
same speed), so the default only bites on longer pairs.

Two things are measured:
  * SPEED, swept over --threads, at the host's native ISA (kernel="auto"): does local HB
    overtake local Pointers as threads rise (Pointers hits the memory-bandwidth wall,
    HB's O(n) working set does not)?
  * MEMORY: run with --single (one arm) and read peak RSS in isolation.

Correctness: an oracle checks every self-pair against the closed form (score = sum of
BLOSUM diagonal, gradient = diagonal residue counts, gap fields zero) at the native ISA
— so this also re-verifies local HB bit-for-bit across the filtered proteome.
"""
import argparse, gc, time, resource
import numpy as np
import nwgrad
from nwgrad.matrices import BLOSUM62

FASTA = "GCF_000001405.40_GRCh38.p14_protein.faa"

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--min-len", type=int, default=512)
ap.add_argument("--max-len", type=int, default=10000)
ap.add_argument("--threads", type=int, nargs="+", default=[1, 4, 16, 30, 60])
ap.add_argument("--cutoff", type=int, default=512)
ap.add_argument("--limit", type=int, default=0, help="subsample to N sequences (0 = all)")
ap.add_argument("--single", choices=["pointers", "hirschberg"], default=None,
                help="run ONE arm at threads[0] and report peak RSS (memory isolation)")
ap.add_argument("--no-oracle", action="store_true")
args = ap.parse_args()


def load(path):
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


proteome = load(FASTA)
seqs = [s for s in proteome if args.min_len <= len(s) <= args.max_len]
if args.limit and len(seqs) > args.limit:
    rng = np.random.default_rng(0)
    seqs = [seqs[i] for i in sorted(rng.choice(len(seqs), args.limit, replace=False))]
lens = np.array([len(s) for s in seqs])
cells = float(((lens + 1.0) ** 2).sum())
print(f"{len(seqs)} self-pairs {args.min_len}-{int(lens.max())} aa, "
      f"mean {lens.mean():.0f}, {cells / 1e9:.2f} Gcells/pass, cutoff={args.cutoff}, "
      f"isa={nwgrad.simd_isa()}")

params = nwgrad.AlignParams(BLOSUM62, gap_open_a=11.0, gap_extend_a=1.0,
                            gap_open_b=11.0, gap_extend_b=1.0)

# closed form (same as proteome_test): self local == self global == diagonal
M = np.asarray(BLOSUM62.to_matrix()); diag = np.diag(M)
lut = np.zeros(256, dtype=np.uint8)
for i, c in enumerate(BLOSUM62.alphabet):
    lut[ord(c)] = i
counts = [np.bincount(lut[np.frombuffer(s.encode(), dtype=np.uint8)], minlength=len(diag)) for s in seqs]
expected = np.array([c @ diag for c in counts], dtype=float)


def build(tb, threads):
    b = nwgrad.SeqPairBatch(n_threads=threads, traceback=tb)
    if tb == "hirschberg":
        b.hb_cutoff = args.cutoff
    b.add_many(seqs, seqs, params, gap_model="affine", mode="local",
               grad_mode="hard", kernel="auto")
    return b


def oracle(b):
    bad = 0
    for i, c in enumerate(counts):
        p = b[i]
        if p.score != expected[i]:
            bad += 1; continue
        g = p.grad.to_dict()
        if not np.array_equal(g["matrix"], np.diag(c.astype(float))): bad += 1; continue
        if (g["gap_open_a"], g["gap_extend_a"], g["gap_open_b"], g["gap_extend_b"]) != (0,0,0,0):
            bad += 1
    print(f"  oracle: {len(seqs) - bad}/{len(seqs)} self-pairs are the identity"
          + ("  *** MISMATCH ***" if bad else ""))


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)  # Linux: KB


if args.single:
    b = build(args.single, args.threads[0])
    b.score_and_grad()                       # warmup / fault-in
    t = time.perf_counter(); total = b.score_and_grad(); dt = time.perf_counter() - t
    print(f"{args.single:10s} {b.n_threads:3d}t  {dt:7.2f}s  {cells/dt/1e6:8.1f} Mcell/s  "
          f"peakRSS {rss_gb():6.2f} GB  (sum={total:.0f})")
    raise SystemExit

print(f"\n{'threads':>7s} {'pointers':>12s} {'hirschberg':>12s} {'hb/ptr':>8s}   Mcell/s(p,hb)")
did_oracle = False
for th in args.threads:
    bp = build("pointers", th)
    bp.score_and_grad()
    t = time.perf_counter(); sp = bp.score_and_grad(); dp = time.perf_counter() - t
    nt = bp.n_threads
    del bp; gc.collect()

    bh = build("hirschberg", th)
    if not did_oracle and not args.no_oracle:
        bh.score_and_grad(); oracle(bh); did_oracle = True
    bh.score_and_grad()
    t = time.perf_counter(); sh = bh.score_and_grad(); dh = time.perf_counter() - t
    del bh; gc.collect()

    assert sp == sh, (sp, sh)
    print(f"{nt:7d} {dp:11.2f}s {dh:11.2f}s {dh/dp:7.2f}x   "
          f"{cells/dp/1e6:7.1f} {cells/dh/1e6:7.1f}")
