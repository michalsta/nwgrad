#!/usr/bin/env bash
# Build bench/drivers/nwgrad_cpp.cpp against an nwgrad source tree the way the wheel is
# built: -O3 -DNDEBUG, -ffp-contract=off -fopenmp-simd, each ISA level TU with its own -m
# flags, clang's -fno-slp-vectorize on the TU holding the scalar code, and the
# release-wheel codegen (no host hardening).
#   build_nwgrad_cpp.sh NWGRAD_SRC OUT_BINARY CXX
set -euo pipefail
SRC=$1; OUT=$2; CXX=${3:-g++}
HERE=$(cd "$(dirname "$0")" && pwd)
NW=$SRC/src/nwgrad/cpp/nwgrad
ARCH=$(uname -m)
BASE="-O3 -DNDEBUG -std=c++20 -ffp-contract=off -fopenmp-simd -I$NW -I$HERE/drivers"
if [ "$ARCH" = x86_64 ]; then
    BASE="$BASE -fcf-protection=none -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=0 -fno-stack-protector -fno-stack-clash-protection -mtls-dialect=gnu"
else
    BASE="$BASE -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=0 -fno-stack-protector -fno-stack-clash-protection"
fi
OBJ=$(mktemp -d)
trap 'rm -rf "$OBJ"' EXIT
NOSLP=""
$CXX --version | grep -q clang && NOSLP="-fno-slp-vectorize"
if [ "$ARCH" = x86_64 ]; then
    $CXX $BASE -c $NW/simd_levels.cpp   -o $OBJ/levels.o &
    $CXX $BASE -c $NW/level_baseline.cpp -o $OBJ/l0.o &
    $CXX $BASE -msse4.2 -mpopcnt -c $NW/level_sse4.cpp -o $OBJ/l1.o &
    $CXX $BASE -mavx2 -mfma -c $NW/level_avx2.cpp -o $OBJ/l2.o &
    $CXX $BASE -mavx512f -mavx512dq -mavx512vl -mavx512bw -c $NW/level_avx512.cpp -o $OBJ/l3.o &
else
    $CXX $BASE -c $NW/simd_levels.cpp -o $OBJ/levels.o &
    $CXX $BASE -c $NW/level_neon.cpp -o $OBJ/l0.o &
fi
$CXX $BASE $NOSLP -c $HERE/drivers/nwgrad_cpp.cpp -o $OBJ/drv.o &
wait
$CXX $BASE $OBJ/*.o -o "$OUT" -lpthread
echo "built $OUT with $($CXX --version | head -1)"
