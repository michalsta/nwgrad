#!/bin/bash
# Build bench_grid with release-wheel codegen (no distro hardening), nwgrad's level TUs linked in.
set -e
cd "$(dirname "$0")"
NW=${NW:-../../src/nwgrad/cpp/nwgrad}
CXX=${CXX:-g++}
FLAGS="${EXTRA:-} -std=c++20 -O3 -DNDEBUG -ffp-contract=off -fopenmp-simd -fcf-protection=none -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=0 -fno-stack-protector -fno-stack-clash-protection -mtls-dialect=gnu -I$NW/.. -I$NW"
OUT=${OUT:-.}
mkdir -p $OUT/obj
# clang: nwgrad's CMake turns SLP off in the TU with the scalar fills (py_exports.cpp);
# here that is bench_grid.cpp (Aligner's scalar code and the grid base case).
SLP=""; case "$($CXX --version | head -1)" in *clang*) SLP="-fno-slp-vectorize";; esac
c() { $CXX $FLAGS $2 -c $1 -o $OUT/obj/$(basename $1 .cpp).o & }
c $NW/simd_levels.cpp ""
c $NW/level_baseline.cpp ""
c $NW/level_sse4.cpp "-msse4.2 -mpopcnt"
c $NW/level_avx2.cpp "-mavx2 -mfma"
c $NW/level_avx512.cpp "-mavx512f -mavx512dq -mavx512vl -mavx512bw"
c gk_level_baseline.cpp ""
c gk_level_sse4.cpp "-msse4.2 -mpopcnt"
c gk_level_avx2.cpp "-mavx2 -mfma"
c gk_level_avx512.cpp "-mavx512f -mavx512dq -mavx512vl -mavx512bw"
c bench_grid.cpp "$SLP"
wait
$CXX -o $OUT/bench_grid $OUT/obj/*.o -lpthread
