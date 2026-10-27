#!/usr/bin/env bash
# Build everything the runner needs on THIS host, inside ROOT (default ~/nwbench), which
# holds nwgrad/ (source), bench/, ext/ (competitor sources) and fixtures/.
#   setup_host.sh [ROOT] [--metered]
# --metered: skip the large Python downloads (scikit-bio and its scipy/pandas stack,
#            torch + DeepBLAST); those arms are then absent on this host.
# Idempotent: a step whose output exists is skipped.  Logs to ROOT/setup.log.
set -uo pipefail
ROOT=$(realpath "${1:-$HOME/nwbench}")
METERED=0; [ "${2:-}" = "--metered" ] && METERED=1
cd "$ROOT"
exec > >(tee -a "$ROOT/setup.log") 2>&1
echo "=== setup $(date) on $(hostname) metered=$METERED"
ARCH=$(uname -m)
NPROC=$(nproc)
PY=${PYTHON:-python3}
# Release-wheel codegen: the host toolchains inject hardening the manylinux wheels do not have.
if [ "$ARCH" = x86_64 ]; then
    HARD="-fcf-protection=none -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=0 -fno-stack-protector -fno-stack-clash-protection -mtls-dialect=gnu"
else
    HARD="-U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=0 -fno-stack-protector -fno-stack-clash-protection"
fi
NATIVE="-march=native"
[ "$ARCH" = aarch64 ] && NATIVE="-mcpu=native"
mkdir -p bin work bin/lib

flags_reach_cc1() {   # $1 = compiler: fail loudly if the hardening survives to cc1/cc1plus
    # The predefined macros are what the compiler proper ends up with, whatever order the
    # distro's injected flags and ours reach it in.
    local m line
    m=$(echo 'int main(){}' | $1 -x c++ -O3 $HARD -dM -E - 2>&1)
    if echo "$m" | grep -q -E '__SSP(_STRONG|_ALL)?__|__CET__|define _FORTIFY_SOURCE [1-9]'; then
        echo "!!! $1: hardening still active:"; echo "$m" | grep -E '__SSP|__CET__|_FORTIFY_SOURCE'; return 1
    fi
    # stack-clash protection has no macro: our -fno- must come after any injected -f
    line=$(echo 'int main(){}' | $1 -x c++ -O3 $HARD -### -c - -o /dev/null 2>&1 | grep -E 'cc1plus|"-cc1"' | tr ' ' '\n' | tr -d '"' | grep -n 'stack-clash' | tail -1)
    case "$line" in *-fstack-clash-protection) echo "!!! $1: stack-clash protection last on the cc1 line"; return 1 ;; esac
    echo "$1: release-wheel flags verified (macros and -###)"
}
flags_reach_cc1 g++ || exit 1
flags_reach_cc1 clang++ || exit 1

make_venv() {   # $1 = dir, $2 = CC, $3 = CXX
    local v=$1
    if [ ! -x "$v/bin/python" ]; then $PY -m venv "$v" || return 1; fi
    "$v/bin/pip" install -q --upgrade pip 2>/dev/null
    "$v/bin/pip" install -q numpy nanobind scikit-build-core setuptools || return 1
    if ! "$v/bin/python" -I -c "import nwgrad" 2>/dev/null; then
        CC=$2 CXX=$3 CFLAGS="$HARD" CXXFLAGS="$HARD" "$v/bin/pip" install -q --no-cache-dir \
            --no-build-isolation ./nwgrad || return 1
    fi
    "$v/bin/python" -I -c "import nwgrad; print('$v:', nwgrad.compiled_with(), nwgrad.simd_isa(), nwgrad.available_isa_levels())"
}
make_venv venv-gcc gcc g++ || { echo "!!! venv-gcc failed"; exit 1; }
make_venv venv-clang clang clang++ || echo "!!! venv-clang failed"
G=$(venv-gcc/bin/python -I -c "import nwgrad; print(nwgrad.compiled_with())")
C=$(venv-clang/bin/python -I -c "import nwgrad; print(nwgrad.compiled_with())" 2>/dev/null)
case "$C" in clang*) ;; *) echo "!!! venv-clang reports '$C', not clang: removing it"; rm -rf venv-clang ;; esac

# competitors (binary packages)
venv-gcc/bin/pip install -q parasail biopython pyopal edlib scoring-matrices || echo "!!! some competitor wheels failed"
if [ $METERED = 0 ]; then
    venv-gcc/bin/pip install -q scikit-bio || echo "!!! scikit-bio failed"
    venv-gcc/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu || echo "!!! torch failed"
    venv-gcc/bin/pip install -q numba git+https://github.com/flatironinstitute/deepblast.git || echo "!!! deepblast failed"
fi

# nwgrad C++ driver, both compilers
[ -x bin/nwgrad_cpp_gcc ] || bench/build_nwgrad_cpp.sh nwgrad bin/nwgrad_cpp_gcc g++ 2>&1 | grep -v -i warning | grep -E 'error|built'
[ -x bin/nwgrad_cpp_clang ] || bench/build_nwgrad_cpp.sh nwgrad bin/nwgrad_cpp_clang clang++ 2>&1 | grep -v -i warning | grep -E 'error|built'

CXXF="-O3 -DNDEBUG $NATIVE $HARD -std=c++20 -Ibench/drivers"
CF="-O3 -DNDEBUG $NATIVE $HARD"
# parasail: link the wheel's own library
PS=$(venv-gcc/bin/python -I -c "import parasail, os; print(os.path.dirname(parasail.__file__))")
if [ ! -x bin/parasail_drv ] && [ -n "$PS" ]; then
    LIBP=$(ls "$PS"/libparasail*.so* "$PS"/*.dylib 2>/dev/null | head -1)
    ln -sf "$LIBP" bin/lib/libparasail.so.8
    g++ $CXXF -I"$PS/include" bench/drivers/parasail_drv.cpp "$LIBP" -Wl,-rpath,"$ROOT/bin/lib" -lpthread -o bin/parasail_drv \
        && echo "built parasail_drv against $LIBP"
fi

# SSE intrinsics on ARM: SSW ships sse2neon.h; KSW2 gets the same header as minimap2 does
SSEINC=""
if [ "$ARCH" = aarch64 ]; then
    mkdir -p work/sse2neon
    for h in emmintrin.h smmintrin.h immintrin.h; do echo '#include "sse2neon.h"' > work/sse2neon/$h; done
    cp ext/ssw/sse2neon.h work/sse2neon/
    SSEINC="-Iwork/sse2neon -D__SSE2__"
fi
if [ ! -x bin/ssw_drv ]; then
    gcc $CF $SSEINC -c ext/ssw/ssw.c -o work/ssw.o && g++ $CXXF $SSEINC -Iext/ssw bench/drivers/ssw_drv.cpp work/ssw.o -lpthread -lm -o bin/ssw_drv && echo "built ssw_drv"
fi
if [ ! -x bin/ksw2_drv ]; then
    gcc $CF $SSEINC -c ext/ksw2/ksw2_extz2_sse.c -o work/ksw2.o && g++ $CXXF $SSEINC -Iext/ksw2 bench/drivers/ksw2_drv.cpp work/ksw2.o -lpthread -o bin/ksw2_drv && echo "built ksw2_drv"
fi
# WFA2: clang -O3 -march=native (WFA2's own flags): gcc 16's -O3 -march=native build
# crashes in its blockwise extend kernel (gcc -O3, -O2 -march=native and clang are fine)
if [ ! -x bin/wfa2_drv ]; then
    # the library's C sources compiled directly (its Makefile also wants the bindings/)
    (cd ext/WFA2-lib && rm -rf build lib && mkdir -p build lib && \
        ls alignment/*.c system/*.c utils/*.c wavefront/*.c | xargs -P $NPROC -I{} sh -c \
            'clang -Wall -fPIE -O3 '"$NATIVE $HARD"' -I. -c {} -o build/$(basename {} .c).o' && \
        ar rcs lib/libwfa.a build/*.o)
    clang $CF -Iext/WFA2-lib -c bench/drivers/wfa2_shim.c -o work/wfa2_shim.o && \
        g++ $CXXF bench/drivers/wfa2_drv.cpp work/wfa2_shim.o ext/WFA2-lib/lib/libwfa.a -lpthread -lm -o bin/wfa2_drv && echo "built wfa2_drv"
fi

# SeqAn3 / SeqAn2: one binary per (alphabet, dtype, mode), in parallel
declare -A T=([int16]=int16_t [int32]=int32_t [int64]=int64_t [float32]=float [float64]=double)
declare -A T2=([int16]=short [int32]=int [int64]=int64_t [float32]=float [float64]=double)
jobs=()
for alph in protein dna; do
    AD=""; [ $alph = dna ] && AD="-DALPH_DNA"
    for dt in int16 int32 int64 float32 float64; do
        o=bin/seqan3_${alph}_${dt}_scalar
        [ -x $o ] || jobs+=("g++ ${CXXF/c++20/c++23} -Iext/seqan3/include $AD -DSCORE_T=${T[$dt]} -DDTYPE_NAME=$dt bench/drivers/seqan3_drv.cpp -lpthread -o $o")
        if [ ${dt:0:3} = int ]; then
            o=bin/seqan3_${alph}_${dt}_vec
            [ -x $o ] || jobs+=("g++ ${CXXF/c++20/c++23} -Iext/seqan3/include $AD -DVECTORISED -DSCORE_T=${T[$dt]} -DDTYPE_NAME=$dt bench/drivers/seqan3_drv.cpp -lpthread -o $o")
        fi
        o=bin/seqan2_${alph}_${dt}
        V=""; [ "$ARCH" = x86_64 ] && { [ $dt = int16 ] || [ $dt = int32 ]; } && V="-DSEQAN2_VEC"
        [ -x $o ] || jobs+=("g++ $CXXF -DSEQAN_ENABLE_DEBUG=0 -Iext/seqan2/include $AD $V '-DSCORE_T=${T2[$dt]}' -DDTYPE_NAME=$dt bench/drivers/seqan2_drv.cpp -lpthread -o $o")
    done
done
printf '%s\n' "${jobs[@]}" > work/seqan_jobs.txt
if [ ${#jobs[@]} -gt 0 ]; then
    # a SeqAn compile takes up to ~3 GB: bound the parallelism by memory as well
    MEMGB=$(awk '/MemTotal/ {print int($2/1048576)}' /proc/meminfo)
    PAR=$(( MEMGB / 4 )); [ $PAR -lt 1 ] && PAR=1; [ $PAR -gt $NPROC ] && PAR=$NPROC
    xargs -P "$PAR" -I{} sh -c '{} > /dev/null 2>work/seqan_err_$$.txt || { echo "!!! failed: {}" | cut -c1-200; grep -m2 error work/seqan_err_$$.txt | cut -c1-300; }' < work/seqan_jobs.txt
fi
echo "binaries: $(ls bin | grep -v lib | tr '\n' ' ')"
echo "=== setup done $(date)"
