#!/usr/bin/env python3
"""Single-batch stress runner for BatchAligner.

Generates a random batch of sequence pairs, runs alignment, and prints
timing and score/gradient statistics.

Examples
--------
    python stress_batch.py
    python stress_batch.py --n 10000 --seq-len 200
    python stress_batch.py --n 500 --seq-len 50 300 --n-threads 8 --grad-mode hard
    python stress_batch.py --gap-model linear --mode local --grad-mode none
    python stress_batch.py --n 1000 --seq-len 100 --n-threads 1 4 8 16
"""

import argparse
import sys
import time
from tqdm import tqdm

import numpy as np

# ── BLOSUM62 (canonical AA order: ACDEFGHIKLMNPQRSTVWY) ──────────────────────

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"

BLOSUM62 = np.array([
    [ 4,  0, -2, -1, -2,  0, -2, -1, -1, -1, -1, -2, -1, -1, -1,  1,  0,  0, -3, -2],
    [ 0,  9, -3, -4, -2, -3, -3, -1, -3, -1, -1, -3, -3, -3, -3, -1, -1, -1, -2, -2],
    [-2, -3,  6,  2, -3, -1, -1, -3, -1, -4, -3,  1, -1,  0, -2,  0, -1, -3, -4, -3],
    [-1, -4,  2,  5, -3, -2,  0, -3,  1, -3, -2,  0, -1,  2,  0,  0, -1, -2, -3, -2],
    [-2, -2, -3, -3,  6, -3, -1,  0, -3,  0,  0, -3, -4, -3, -3, -2, -2, -1,  1,  3],
    [ 0, -3, -1, -2, -3,  6, -2, -4, -2, -4, -3,  0, -2, -2, -2,  0, -2, -3, -2, -3],
    [-2, -3, -1,  0, -1, -2,  8, -3, -1, -3, -2,  1, -2,  0,  0, -1, -2, -3, -2,  2],
    [-1, -1, -3, -3,  0, -4, -3,  4, -3,  2,  1, -3, -3, -3, -3, -2, -1,  3, -3, -1],
    [-1, -3, -1,  1, -3, -2, -1, -3,  5, -2, -1,  0, -1,  1,  2,  0, -1, -2, -3, -2],
    [-1, -1, -4, -3,  0, -4, -3,  2, -2,  4,  2, -3, -3, -2, -2, -2, -1,  1, -2, -1],
    [-1, -1, -3, -2,  0, -3, -2,  1, -1,  2,  5, -2, -2,  0, -1, -1, -1,  1, -1, -1],
    [-2, -3,  1,  0, -3,  0,  1, -3,  0, -3, -2,  6, -2,  0,  0,  1,  0, -3, -4, -2],
    [-1, -3, -1, -1, -4, -2, -2, -3, -1, -3, -2, -2,  7, -1, -2, -1, -1, -2, -4, -3],
    [-1, -3,  0,  2, -3, -2,  0, -3,  1, -2,  0,  0, -1,  5,  1,  0, -1, -2, -2, -1],
    [-1, -3, -2,  0, -3, -2,  0, -3,  2, -2, -1,  0, -2,  1,  5, -1, -1, -3, -3, -2],
    [ 1, -1,  0,  0, -2,  0, -1, -2,  0, -2, -1,  1, -1,  0, -1,  4,  1, -2, -3, -2],
    [ 0, -1, -1, -1, -2, -2, -2, -1, -1, -1, -1,  0, -1, -1, -1,  1,  5,  0, -2, -2],
    [ 0, -1, -3, -2, -1, -3, -3,  3, -2,  1,  1, -3, -2, -2, -3, -2,  0,  4, -3, -1],
    [-3, -2, -4, -3,  1, -2, -2, -3, -3, -2, -1, -4, -4, -2, -3, -3, -2, -3, 11,  2],
    [-2, -2, -3, -2,  3, -3,  2, -1, -2, -1, -1, -2, -3, -1, -2, -2, -2, -1,  2,  7],
], dtype=np.float64)


# ── arg parsing ───────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Run a single BatchAligner stress batch and report timing/stats.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n", type=int, default=1000, metavar="N",
                   help="number of sequence pairs")
    p.add_argument("--seq-len", type=int, nargs="+", default=[50], metavar="LEN",
                   help="sequence length: one value for fixed length, two for [LO, HI] range")
    p.add_argument("--gap-model", choices=["linear", "affine"], default="affine")
    p.add_argument("--mode", choices=["global", "local"], default="global")
    p.add_argument("--grad-mode", choices=["none", "hard", "soft"], default="hard")
    p.add_argument("--gap-open", type=float, default=11.0,
                   help="gap-open penalty (affine only)")
    p.add_argument("--gap-extend", type=float, default=1.0,
                   help="gap-extend penalty")
    p.add_argument("--n-threads", type=int, default=4, metavar="T",
                   help="thread count to run")
    p.add_argument("--seed", type=lambda x: int(x, 0), default=None,
                   help="RNG seed (hex or decimal); random if omitted")
    p.add_argument("--warmup", action="store_true",
                   help="run one silent warmup pass before timing")
    return p.parse_args(argv)


# ── sequence generation ───────────────────────────────────────────────────────

def generate_seqs(rng, n, seq_len_arg):
    alphabet = list(AA_ORDER)
    if len(seq_len_arg) == 1:
        lo = hi = seq_len_arg[0]
    else:
        lo, hi = seq_len_arg[0], seq_len_arg[1]
    lengths = rng.integers(lo, hi + 1, size=n)
    seqs = ["".join(rng.choice(alphabet, size=int(l)).tolist())
            for l in tqdm(lengths, desc="generating seqs", unit="seq", leave=False)]
    return seqs, int(lengths.mean()), int(lengths.min()), int(lengths.max())


# ── reporting ─────────────────────────────────────────────────────────────────

def print_header(args):
    lo_hi = (f"{args.seq_len[0]}" if len(args.seq_len) == 1
             else f"{args.seq_len[0]}–{args.seq_len[1]}")
    print(f"n={args.n}  seq_len={lo_hi}  "
          f"gap_model={args.gap_model}  mode={args.mode}  "
          f"grad_mode={args.grad_mode}  "
          f"gap_open={args.gap_open}  gap_extend={args.gap_extend}  "
          f"seed={args.seed:#x}")


def print_run(n_threads, elapsed, scores, grad, n_pairs):
    pairs_per_sec = n_pairs / elapsed
    print(f"  n_threads={n_threads:2d}  "
          f"time={elapsed:.3f}s  "
          f"pairs/s={pairs_per_sec:,.0f}  "
          f"score min={scores.min():.1f} mean={scores.mean():.1f} max={scores.max():.1f}", end="")
    if grad is not None:
        total_counts = grad.sum()
        print(f"  grad_sum={total_counts:.0f}", end="")
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main(argv=None):
    import nwgrad

    args = parse_args(argv)

    seed = args.seed if args.seed is not None else int(np.random.SeedSequence().entropy & 0xFFFFFFFFFFFFFFFF)
    args.seed = seed  # store for print_header
    rng = np.random.default_rng(seed)

    seqs_a, mean_len, min_len, max_len = generate_seqs(rng, args.n, args.seq_len)
    seqs_b, _, _, _                    = generate_seqs(rng, args.n, args.seq_len)

    blosum = nwgrad.BlosumMatrix(BLOSUM62)

    go = args.gap_open if args.gap_model == "affine" else 0.0
    ge = args.gap_extend

    print_header(args)
    print(f"actual seq_len: mean={mean_len}  min={min_len}  max={max_len}")
    print()

    aligner = nwgrad.BatchAligner(
        matrix=blosum,
        gap_open=go, gap_extend=ge,
        gap_model=args.gap_model,
        mode=args.mode,
        grad_mode=args.grad_mode,
        n_threads=args.n_threads,
    )

    if args.warmup:
        aligner.align(seqs_a[:min(args.n, 16)], seqs_b[:min(args.n, 16)])

    print("running...", end="", flush=True)
    t0 = time.perf_counter()
    result = aligner.align(seqs_a, seqs_b)
    elapsed = time.perf_counter() - t0
    print(" done.")

    scores = np.array(result.scores)
    grad   = np.array(result.grad) if args.grad_mode != "none" else None

    print_run(args.n_threads, elapsed, scores, grad, args.n)

    return 0


if __name__ == "__main__":
    sys.exit(main())
