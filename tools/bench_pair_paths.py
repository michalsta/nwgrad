#!/usr/bin/env python3
"""Compare optional SeqPair path storage inside one installed binary.

The two keep_paths=True arms are a control for drift. Read that ratio before
interpreting the keep_paths=False / True ratio. No package is compiled here.
"""
import argparse
import gc
import math
import os
import statistics
import time

import numpy as np
import nwgrad


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", nargs="+", type=int, default=[50, 200, 1000])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seconds", type=float, default=0.05, help="target seconds per timing arm")
    parser.add_argument("--precision", choices=["double", "float32"], default="double")
    parser.add_argument("--grad", nargs="+", choices=["hard", "soft"], default=["hard", "soft"])
    parser.add_argument("--cpu", type=int, help="Linux CPU affinity (otherwise leave affinity unchanged)")
    args = parser.parse_args()
    if min(args.seq_len) < 1 or args.repeats < 1 or not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("lengths, repeats and seconds must be positive")
    if args.cpu is not None:
        if not hasattr(os, "sched_setaffinity"):
            parser.error("--cpu requires Linux sched_setaffinity")
        os.sched_setaffinity(0, {args.cpu})
    params = nwgrad.AlignParams(np.full((4, 4), -1.) + np.eye(4) * 4, alphabet="ACGT",
                                gap_open_a=2., gap_extend_a=.5, gap_open_b=2., gap_extend_b=.5)
    cls = nwgrad.SeqPairDouble if args.precision == "double" else nwgrad.SeqPair
    gc.disable()
    print(f"{nwgrad.compiled_with()}, {nwgrad.simd_isa()}, {args.precision}; traceback=pointers")
    print("False/True < 1 means less time without paths; control True/True should be near 1.")
    print("grad len     use        True us      False us  False/True  control")
    for grad in args.grad:
        for length in args.seq_len:
            a = ("ACGT" * ((length + 3) // 4))[:length]
            b = a[:length // 3] + "T" + a[length // 3 + 1:]
            def pair():
                return cls(a, b, params, grad_mode=grad, traceback="pointers")
            for cold in (False, True):
                objects = [pair() for _ in range(3)]
                for obj, keep in zip(objects, (True, False, True)):
                    obj.score_and_grad(keep_paths=keep)
                start = time.perf_counter()
                obj = pair() if cold else objects[0]
                obj.score_and_grad()
                calls = max(1, math.ceil(args.seconds / max(1e-9, time.perf_counter() - start)))
                times = [[], [], []]
                for turn in range(args.repeats):
                    for arm in ((0, 1, 2) if turn % 2 == 0 else (2, 1, 0)):
                        keep = (True, False, True)[arm]
                        start = time.perf_counter()
                        for _ in range(calls):
                            obj = pair() if cold else objects[arm]
                            obj.score_and_grad(keep_paths=keep)
                        times[arm].append((time.perf_counter() - start) / calls)
                med = [statistics.median(t) for t in times]
                print(f"{grad:4} {length:4} {'construct' if cold else 'reuse':9} "
                      f"{med[0]*1e6:12.2f} {med[1]*1e6:12.2f} {med[1]/med[0]:11.3f} {med[2]/med[0]:8.3f}", flush=True)


if __name__ == "__main__":
    main()
