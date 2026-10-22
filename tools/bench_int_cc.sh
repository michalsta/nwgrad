#!/usr/bin/env bash
# Build this tree with gcc AND clang into two throwaway venvs (release-wheel codegen: the
# hardening the distro toolchains inject is switched off, as the wheels have it), run
# tools/bench_int.py in each, and print the clang/gcc time ratio per arm.
#
#   tools/bench_int_cc.sh OUTDIR [bench_int args...]
#
# --no-cache-dir / --force-reinstall: pip caches wheels by source tree, not by CC, so
# without them the clang build silently reinstalls gcc's (see bench_toolchain.sh).
set -euo pipefail
OUT="$1"; shift
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$OUT"
FLAGS="-fcf-protection=none -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=0 -fno-stack-protector -fno-stack-clash-protection"
case "$(uname -m)" in x86_64) FLAGS="$FLAGS -mtls-dialect=gnu";; esac

build () {
    local tag="$1" cc="$2" cxx="$3"
    local venv="$OUT/venv-$tag"
    [ -x "$venv/bin/python" ] || python3 -m venv "$venv"
    "$venv/bin/pip" install -q numpy >/dev/null
    CC="$cc" CXX="$cxx" "$venv/bin/pip" install -q --no-cache-dir --force-reinstall \
        -C cmake.define.CMAKE_CXX_FLAGS="$FLAGS" "$SRC" >/dev/null
    "$venv/bin/python" -c "import nwgrad; print('$tag build:', nwgrad.compiled_with())"
}
build gcc gcc g++
build clang clang clang++
for tag in gcc clang; do
    echo "=== $tag ==="
    "$OUT/venv-$tag/bin/python" "$SRC/tools/bench_int.py" --json "$OUT/$tag.json" "$@"
done
python3 - "$OUT/gcc.json" "$OUT/clang.json" <<'PY'
import json, sys
g = json.load(open(sys.argv[1])); c = json.load(open(sys.argv[2]))
if g["compiler"] == c["compiler"]:
    sys.exit("ABORT: both builds report the same compiler -- the second build did not happen")
print(f"\ngcc {g['compiler']} | clang {c['compiler']}\n{'arm':44s} {'gcc us':>9s} {'clang us':>9s} {'clang/gcc':>10s}")
for k in g["arms"]:
    gs, cs = g["arms"][k], c["arms"][k]
    flag = "  <-- " + ("clang slower" if cs / gs > 1.07 else "gcc slower") if abs(cs / gs - 1) > 0.07 else ""
    print(f"{k:44s} {gs*1e6:9.3f} {cs*1e6:9.3f} {cs/gs:10.3f}{flag}")
PY
