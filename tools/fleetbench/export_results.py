#!/usr/bin/env python3
"""results/*.jsonl -> OUT/<name>.jsonl.xz for the repository: one JSON object per arm,
without the soft arms' cached per-pair scores (a resume aid, not a result).

    export_results.py --results results/ --out tools/fleetbench/results/
"""
import argparse
import glob
import json
import lzma
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for path in sorted(glob.glob(os.path.join(a.results, "*.jsonl"))):
        name = os.path.basename(path)[:-6]
        if name == "test":
            continue
        recs = []
        for line in open(path):
            r = json.loads(line)
            r.pop("_scores", None)
            recs.append(json.dumps(r, separators=(",", ":")))
        dst = os.path.join(a.out, name + ".jsonl.xz")
        with lzma.open(dst, "wt", preset=9) as f:
            f.write("\n".join(recs) + "\n")
        print(f"{dst}: {len(recs)} records, {os.path.getsize(dst) // 1024} KiB")


if __name__ == "__main__":
    main()
