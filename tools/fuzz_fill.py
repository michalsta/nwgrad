#!/usr/bin/env python3
"""Differential fuzz: SeqPairBatch fill="interpair" against fill="striped".

Each seed builds one random batch -- several add_many() groups, each with its own gap
model, mode, grad mode and parameters (ties, integer, real-valued; cheap and dear gaps),
empty and one-residue sequences, lengths past hb_cutoff, shared and ragged len B, both
precisions, every traceback -- and runs score_and_grad() and then banded_grad(bw) under
both fills.  Hard scores and gradients must be bit-identical (the documented contract),
soft ones equal within 1e-9.  An exception must be the same exception under both.

Every seed runs in its own process, so a crash is attributed to its seed instead of
ending the sweep:

    python tools/fuzz_fill.py 0 400 [--jobs 8]
    python tools/fuzz_fill.py --seed 6        # one seed, in this process (for gdb)

This is the harness that found the linear/affine DpBuffer overrun (test_buffer_reuse.cpp):
39 of 400 seeds crashed before the guard fix, none after.
"""
import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import nwgrad

CLS = {"double": nwgrad.SeqPairBatchDouble, "float32": nwgrad.SeqPairBatch}


def rparams(rng, alpha, gm):
    n = len(alpha)
    kind = rng.integers(3)
    if kind == 0:  # ties
        m = np.full((n, n), -1.0)
        np.fill_diagonal(m, 2.0)
    elif kind == 1:  # integer, BLOSUM-like
        m = rng.integers(-4, 6, (n, n)).astype(float)
    else:
        m = rng.normal(size=(n, n))
    go = float(rng.choice([0.0, 0.5, 2.5, 11.0, 1e-4]))
    ge = float(rng.choice([0.1, 0.7, 1.0, 1e-4]))
    if gm == "linear":
        go = 0.0
    return nwgrad.AlignParams(nwgrad.SubstMatrix(m, alphabet=alpha), go, ge,
                              go * float(rng.choice([1, 1.3])), ge * float(rng.choice([1, 0.8])))


def rseq(rng, alpha, n):
    return "".join(rng.choice(list(alpha), n)) if n else ""


def rlen(rng):
    r = rng.random()
    if r < 0.05:
        return 0
    if r < 0.10:
        return 1
    if r < 0.80:
        return int(rng.integers(2, 60))
    if r < 0.95:
        return int(rng.integers(60, 300))
    return int(rng.integers(500, 800))


def make_case(seed):
    rng = np.random.default_rng(seed)
    alpha = str(rng.choice(["ACGT", nwgrad.PROTEIN.symbols, "ACGTN"]))
    prec = str(rng.choice(["double", "float32"]))
    tb = str(rng.choice(["auto", "pointers", "scores"]))
    specs = []
    for _ in range(int(rng.integers(1, 5))):
        gm = str(rng.choice(["affine", "linear"]))
        mode = str(rng.choice(["global", "local"]))
        gr = str(rng.choice(["hard", "hard", "none", "soft"]))
        p = rparams(rng, alpha, gm)
        k = int(rng.integers(1, 40))
        lb = rlen(rng)
        A = [rseq(rng, alpha, rlen(rng)) for _ in range(k)]
        B = [rseq(rng, alpha, lb if rng.random() < 0.7 else rlen(rng)) for _ in range(k)]
        specs.append((A, B, p, gm, mode, gr))
    bw = int(rng.integers(1, 12))
    return alpha, prec, tb, specs, bw


def run_fill(prec, tb, specs, fill, bw):
    b = CLS[prec](n_threads=3, traceback=tb)
    b.fill = fill
    for A, B, p, gm, mode, gr in specs:
        b.add_many(A, B, p, gap_model=gm, mode=mode, grad_mode=gr)
    try:
        b.score_and_grad()
    except Exception as e:  # must match across fills, checked by the caller
        return ("raised", type(e).__name__)
    full = b.scores()
    try:
        grads = b.grads()
    except Exception:
        grads = None
    try:
        b.banded_grad(bw)
        banded = b.scores()
    except Exception as e:
        banded = ("raised", type(e).__name__)
    return full, grads, banded


def check_seed(seed):
    """None if the fills agree, else a one-line description of the first disagreement."""
    alpha, prec, tb, specs, bw = make_case(seed)
    tag = (f"seed={seed} alphabet={len(alpha)} {prec} traceback={tb} bw={bw} groups="
           f"{[(len(s[0]), s[3], s[4], s[5]) for s in specs]}")
    s = run_fill(prec, tb, specs, "striped", bw)
    ip = run_fill(prec, tb, specs, "interpair", bw)
    if isinstance(s[0], str) or isinstance(ip[0], str):
        return None if s == ip else f"EXCEPTION MISMATCH {tag}: striped={s[:2]} interpair={ip[:2]}"
    soft = np.array([spec[5] == "soft" for spec in specs for _ in spec[0]])

    def compare(what, x, y):
        same = (x == y) | (np.isnan(x) & np.isnan(y))
        bad = np.where(~same & ~soft)[0]
        if len(bad):
            i = bad[0]
            return f"{what} MISMATCH {tag}: pair {i} striped={x[i]!r} interpair={y[i]!r}"
        close = np.isclose(x, y, rtol=1e-9, atol=1e-9) | same
        bad = np.where(~close & soft)[0]
        if len(bad):
            i = bad[0]
            return f"SOFT {what} MISMATCH {tag}: pair {i} striped={x[i]!r} interpair={y[i]!r}"
        return None

    r = compare("SCORE", s[0], ip[0])
    if r:
        return r
    if s[1] is not None and ip[1] is not None:
        hard = ~soft
        if not (np.array_equal(s[1][0][hard], ip[1][0][hard]) and
                np.array_equal(s[1][1][hard], ip[1][1][hard])):
            return f"GRADIENT MISMATCH {tag}"
    if isinstance(s[2], tuple) or isinstance(ip[2], tuple):
        return None if s[2] == ip[2] else f"BANDED EXCEPTION MISMATCH {tag}: {s[2]} / {ip[2]}"
    return compare("BANDED SCORE", s[2], ip[2])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("lo", type=int, nargs="?", default=0)
    ap.add_argument("hi", type=int, nargs="?", default=100)
    ap.add_argument("--seed", type=int, help="run one seed in this process")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    if a.seed is not None:
        r = check_seed(a.seed)
        print(r or f"seed={a.seed} ok")
        sys.exit(1 if r else 0)

    def one(seed):
        p = subprocess.run([sys.executable, __file__, "--seed", str(seed)],
                           capture_output=True, text=True)
        if p.returncode < 0:
            return f"seed={seed} CRASHED (signal {-p.returncode})"
        if p.returncode != 0:
            return (p.stdout.strip() or p.stderr.strip().splitlines()[-1])
        return None

    with ThreadPoolExecutor(a.jobs) as ex:
        bad = [r for r in ex.map(one, range(a.lo, a.hi)) if r]
    for r in bad:
        print(r)
    print(f"{a.hi - a.lo} seeds, {len(bad)} failures")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
