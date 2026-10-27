#!/usr/bin/env bash
# Build a patched nwgrad beside the measured one, for an A/B run:
#   ab_setup.sh ROOT PATCH TAG      -> ROOT/nwgrad-TAG (source), ROOT/venv-TAG-gcc, ROOT/venv-TAG-clang
# Same release-wheel flags and build path as setup_host.sh; numpy and the build deps are
# reused from venv-gcc's wheel cache where pip has them.
set -euo pipefail
ROOT=$(realpath "$1"); PATCH=$(realpath "$2"); TAG=$3
cd "$ROOT"
ARCH=$(uname -m)
if [ "$ARCH" = x86_64 ]; then
    HARD="-fcf-protection=none -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=0 -fno-stack-protector -fno-stack-clash-protection -mtls-dialect=gnu"
else
    HARD="-U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=0 -fno-stack-protector -fno-stack-clash-protection"
fi
if [ ! -d nwgrad-$TAG ]; then
    cp -r nwgrad nwgrad-$TAG
    (cd nwgrad-$TAG && patch -p1 -s < "$PATCH")
fi
for cc in gcc clang; do
    v=venv-$TAG-$cc
    cxx=$([ $cc = gcc ] && echo g++ || echo clang++)
    [ -x $v/bin/python ] || python3 -m venv $v
    $v/bin/pip install -q numpy nanobind scikit-build-core setuptools
    if ! $v/bin/python -I -c "import nwgrad" 2>/dev/null; then
        CC=$cc CXX=$cxx CFLAGS="$HARD" CXXFLAGS="$HARD" $v/bin/pip install -q --no-cache-dir \
            --no-build-isolation ./nwgrad-$TAG
    fi
    $v/bin/python -I -c "import nwgrad; print('$v:', nwgrad.compiled_with(), nwgrad.simd_isa())"
done
