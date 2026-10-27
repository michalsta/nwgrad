#!/usr/bin/env python3
"""Build every fixture of the fleet benchmark, deterministically, into one directory.

    make_fixtures.py --fasta PROTEOME.faa --manakov manakov_fit_rc.tsv --out fixtures/
    make_fixtures.py --manakov manakov_fit_rc.tsv --out fixtures/ --only mirna   # on a host
                                                       # that has the TSV but not the rest

Fixtures (one TSV per set, "a<TAB>b" per line):
  protein_L{L}_{st,mt}  human proteins (GRCh38 RefSeq), each paired with a mutated copy:
                        20 % substitutions, ~3 % deletions, ~3 % insertion runs of 1-5
  dna_L{L}_{st,mt}      uniform random DNA, mutated the same way at 10 % substitutions
  mirna_{st,mt}         Manakov miRNA x target-site pairs (22 x 50), ACGT rows only
st = single-thread set, ~3e7 DP cells (at least 2 pairs); mt = multithread set, ~3e8.
"""
import argparse
import hashlib
import os

import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"
DNA = "ACGT"
ST_CELLS, MT_CELLS = 3e7, 3e8
PROTEIN_ST = [100, 300, 1000, 3000, 10000]
PROTEIN_MT = [100, 300, 1000, 3000]
DNA_ST = [100, 1000, 10000]
DNA_MT = [100, 1000]


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
    ok = set(AA)
    return sorted({s for s in seqs if set(s) <= ok})          # dedupe isoforms, fix order


def mutator(rng, alphabet, sub):
    letters = np.array(list(alphabet))
    def mutate(s):
        out = []
        for c in s:
            if rng.random() < 0.03: continue
            out.append(c if rng.random() > sub else str(rng.choice(letters)))
            if rng.random() < 0.03: out.extend(rng.choice(letters, int(rng.integers(1, 6))))
        return "".join(out)
    return mutate


def n_pairs(L, cells):
    return max(2, int(round(cells / (L * L))))


def write(out, name, pairs):
    path = os.path.join(out, name + ".tsv")
    with open(path, "w") as f:
        for a, b in pairs:
            f.write(f"{a}\t{b}\n")
    return path


def protein(out, fasta, seed):
    pool = read_fasta(fasta)
    rng = np.random.default_rng(seed)
    mutate = mutator(rng, AA, 0.20)
    for tag, cells, lengths in (("st", ST_CELLS, PROTEIN_ST), ("mt", MT_CELLS, PROTEIN_MT)):
        for L in lengths:
            cand = [s for s in pool if 0.9 * L <= len(s) <= 1.1 * L]
            n = n_pairs(L, cells)
            crop = not cand
            if crop:                     # none that long: a ~L window of a longer protein
                longer = [s for s in pool if len(s) > 1.1 * L]
                cand = []
                for i in rng.choice(len(longer), size=n, replace=n > len(longer)):
                    w = int(rng.integers(int(0.9 * L), int(1.1 * L) + 1))
                    o = int(rng.integers(0, len(longer[i]) - w + 1))
                    cand.append(longer[i][o:o + w])
                src = f"windows of {len(longer)} longer proteins"
            else:
                src = f"{len(cand)} candidates"
            idx = rng.choice(len(cand), size=n, replace=n > len(cand))
            write(out, f"protein_L{L}_{tag}", [(cand[i], mutate(cand[i])) for i in idx])
            print(f"protein_L{L}_{tag}: {n} pairs from {src}")


def dna(out, seed):
    rng = np.random.default_rng(seed + 1)
    mutate = mutator(rng, DNA, 0.10)
    letters = np.array(list(DNA))
    for tag, cells, lengths in (("st", ST_CELLS, DNA_ST), ("mt", MT_CELLS, DNA_MT)):
        for L in lengths:
            n = n_pairs(L, cells)
            base = ["".join(rng.choice(letters, L)) for _ in range(n)]
            write(out, f"dna_L{L}_{tag}", [(a, mutate(a)) for a in base])
            print(f"dna_L{L}_{tag}: {n} pairs")


def mirna(out, tsv, seed):
    rows = []
    with open(tsv) as f:
        next(f)
        for line in f:
            a, b = line.rstrip("\n").split("\t")[:2]
            if set(a) <= set(DNA) and set(b) <= set(DNA) and a and b:
                rows.append((a, b))
    rng = np.random.default_rng(seed + 2)
    for tag, cells in (("st", ST_CELLS), ("mt", MT_CELLS)):
        n = int(cells / 1100)                                   # 22 x 50 cells a pair
        idx = rng.choice(len(rows), size=n, replace=False)
        write(out, f"mirna_{tag}", [rows[i] for i in idx])
        print(f"mirna_{tag}: {n} pairs of {len(rows)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fasta"); ap.add_argument("--manakov")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20261027)
    ap.add_argument("--only", choices=["protein", "dna", "mirna"], action="append")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    only = set(args.only or ["protein", "dna", "mirna"])
    if "protein" in only: protein(args.out, args.fasta, args.seed)
    if "dna" in only: dna(args.out, args.seed)
    if "mirna" in only: mirna(args.out, args.manakov, args.seed)
    for name in sorted(os.listdir(args.out)):
        if name.endswith(".tsv"):
            h = hashlib.md5(open(os.path.join(args.out, name), "rb").read()).hexdigest()
            print(f"{h}  {name}")


if __name__ == "__main__":
    main()
