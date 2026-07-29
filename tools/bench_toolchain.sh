#!/usr/bin/env bash
# Build this tree with gcc AND clang into two throwaway venvs, run tools/bench_toolchain.py
# in each, and print the clang/gcc ratio per arm.
#
# This exists to re-validate the COMPILER POLICY in .github/workflows/wheels.yml, which
# pins x86_64 Linux to clang on the strength of a measurement taken on the prototype
# kernel.  Run it on every x86 host in the fleet (widths W=2/4/8 disagree — that is the
# whole reason the policy is per-arch and not global).
#
# TRAP, and the reason --no-cache-dir appears below: pip caches built wheels keyed by the
# source tree, NOT by CC/CXX.  Without it the clang build silently reinstalls the wheel gcc
# just built and every arm ties at exactly 1.00x — a convincing, entirely fake result.
# The script verifies the two builds actually differ by asking the extension itself
# (nwgrad.compiled_with()) and ABORTS if they report the same compiler.
#
# usage: tools/bench_toolchain.sh [threads] [reps]
set -euo pipefail

THREADS="${1:-1}"
REPS="${2:-3}"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${TMPDIR:-/tmp}/nwgrad-toolchain-$$"
mkdir -p "$OUT"

build_and_run () {
    local tag="$1" cc="$2" cxx="$3"
    local venv="$OUT/venv-$tag"
    echo "=== building with $cxx ===" >&2
    python3 -m venv "$venv"
    "$venv/bin/pip" install -q --upgrade pip >/dev/null
    "$venv/bin/pip" install -q numpy >/dev/null
    # --no-cache-dir: see the trap note above.  --force-reinstall for the same reason.
    CC="$cc" CXX="$cxx" "$venv/bin/pip" install -q --no-cache-dir --force-reinstall \
        "$SRC" >/dev/null
    "$venv/bin/python" "$SRC/tools/bench_toolchain.py" \
        --threads "$THREADS" --reps "$REPS" --json "$OUT/$tag.json"
}

build_and_run gcc   gcc   g++
echo
build_and_run clang clang clang++
echo

python3 - "$OUT/gcc.json" "$OUT/clang.json" <<'PY'
import json, sys
g = json.load(open(sys.argv[1])); c = json.load(open(sys.argv[2]))
if g["compiler"] == c["compiler"]:
    sys.exit(f"ABORT: both builds report compiled_with()={g['compiler']!r} — the second "
             f"build did not happen (pip cache?).  Any ratio below would be fake.")
print(f"isa={g['isa']}  threads={g['threads']}")
print(f"gcc   build: {g['compiler']}")
print(f"clang build: {c['compiler']}")
print()
print(f"{'arm':>14s} {'gcc Mcell/s':>12s} {'clang Mcell/s':>14s} {'clang/gcc':>10s}  "
      f"{'vs control':>10s}")
ctl = c["arms"]["scalar_ctl"]["secs"] and \
      g["arms"]["scalar_ctl"]["secs"] / c["arms"]["scalar_ctl"]["secs"]
for k in g["arms"]:
    gs, cs = g["arms"][k]["secs"], c["arms"][k]["secs"]
    sp = gs / cs                     # >1 means clang is FASTER
    rel = sp / ctl if ctl else float("nan")
    tag = "  <- CONTROL" if k == "scalar_ctl" else ""
    print(f"{k:>14s} {g['arms'][k]['mcells']:12.1f} {c['arms'][k]['mcells']:14.1f} "
          f"{sp:9.3f}x {rel:9.3f}x{tag}")
print()
print("clang/gcc > 1.00 means clang is faster.  'vs control' divides out the compiler's")
print("general codegen difference on this code, isolating the std::simd kernels.")
PY
echo "artifacts in $OUT" >&2
