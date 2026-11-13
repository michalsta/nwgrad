cmake_minimum_required(VERSION 3.15)

project(nwgrad LANGUAGES CXX)

if (MSVC)
    message(FATAL_ERROR
        "nwgrad cannot be built with MSVC or clang-cl: the vectorized kernels use "
        "<experimental/simd>, which only libstdc++ ships.  On Windows, build with "
        "MinGW-w64 gcc (MSYS2 UCRT64: pacman -S mingw-w64-ucrt-x86_64-gcc), e.g. "
        "CMAKE_GENERATOR=Ninja CC=gcc CXX=g++ pip install nwgrad --no-binary nwgrad, "
        "or install the prebuilt win_amd64 wheel.")
endif()

# CPython 3.10 / 3.11's Windows pyconfig.h sets MS_WIN64 only under _MSC_VER (3.12 added
# a __MINGW32__ branch); without it Py_ssize_t is a 32-bit int against a 64-bit
# interpreter.  Defined empty so it is identical to pyconfig's own #define where both
# exist.  Directory-wide, so nanobind's static library (built in this directory by
# nanobind_add_module) gets it too.
if (MINGW AND CMAKE_SIZEOF_VOID_P EQUAL 8)
    add_compile_definitions("MS_WIN64=")
endif()
# MinGW must target the UCRT, as CPython does (MSYS2's UCRT64 environment, not MINGW64,
# winlibs-msvcrt or Strawberry Perl).  An MSVCRT MinGW is not just a second C runtime in
# the process: its libmsvcrt.a links mingw-w64's own x87 exp / log / log1p, whose last
# bit follows the calling thread's x87 precision-control word — and a thread the batch
# engine starts does not share the interpreter thread's.  Seen (cross build, wine): batch
# scores 1 ULP apart between thread counts, logistic probabilities off scipy's expit.
# The UCRT build imports the SSE2 ones from ucrtbase.  NWGRAD_ALLOW_MSVCRT=ON overrides,
# for experiments only.
if (MINGW)
    include(CheckCXXSourceCompiles)
    check_cxx_source_compiles("#include <cstdlib>
#ifndef _UCRT
#error not a UCRT MinGW
#endif
int main() { return 0; }" NWGRAD_MINGW_UCRT)
    option(NWGRAD_ALLOW_MSVCRT "Allow building with an MSVCRT (non-UCRT) MinGW" OFF)
    if (NOT NWGRAD_MINGW_UCRT AND NOT NWGRAD_ALLOW_MSVCRT)
        message(FATAL_ERROR
            "nwgrad: this MinGW targets MSVCRT, not the UCRT that CPython uses; its x87 "
            "exp/log make results depend on the thread that computes them.  Use MSYS2's "
            "UCRT64 gcc (pacman -S mingw-w64-ucrt-x86_64-gcc).  "
            "-DNWGRAD_ALLOW_MSVCRT=ON builds anyway.")
    endif()
endif()

# PE/COFF caps an object at 32767 sections, and the template-heavy TUs (every kernel
# instantiated per scalar type, each in its own COMDAT section) pass it: "too many
# sections" / "file too big" from the assembler.  -mbig-obj lifts the cap.
if (MINGW)
    add_compile_options(-Wa,-mbig-obj)
endif()
find_package(Python 3.10
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)
find_package(nanobind REQUIRED CONFIG)
# nanobind >= 3.0: the bindings call its backend keep-alive entry (NB_CALL) directly.
# Checked here rather than as find_package(nanobind 3.0): nanobind's version file uses
# same-major compatibility, which would also reject 4.x, and the requirement is a floor.
if (nanobind_VERSION VERSION_LESS 3.0)
    message(FATAL_ERROR "nwgrad needs nanobind >= 3.0 (found ${nanobind_VERSION})")
endif()

set(CMAKE_CXX_STANDARD 20)

# No FMA contraction anywhere.  Every simd level promises tables bit-identical to the
# scalar path, and a multiply next to an add is exactly what a compiler fuses into one
# rounding where the scalar build — compiled for a target without FMA — keeps two.  It
# happened: clang (18 and 22, not gcc) fused the striped kernels' closed-form borders
# -(go + j*ge) in the avx2/avx512 TUs, 8 FMAs per TU, so with a non-representable
# gap_extend a border cell differed by an ULP and kernel="avx2" returned a different
# path than kernel="scalar_fallback".  The third instance after the pmax ramp and
# weighted_grad, each fixed by hand; this closes the class.  Free here: the kernels are
# max/add chains, with nothing for an FMA to speed up.
add_compile_options($<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-ffp-contract=off>)
# -fopenmp-simd enables ONLY `#pragma omp simd` (no OpenMP runtime, no other effect).
# The scaled soft path (aligner.hpp, SoftImpl::Scaled) marks its reductions with it so
# they vectorize; that path is tolerance-tested, not bit-exact.  No Viterbi/hard-path
# loop carries the pragma, so their bit-exactness is untouched.
add_compile_options($<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-fopenmp-simd>)

# ── Per-ISA-level kernel TUs ──────────────────────────────────────────────────
#
# The affine Viterbi kernels are compiled once per instruction-set level, each in its
# own TU with that level's real -march flag, so std::simd picks the right register
# width (a #pragma cannot widen std::simd — its ABI is fixed at instantiation, so a
# genuine compile flag defining __AVX2__ etc. is required).  The level TUs register
# their kernels at load; the extension dispatches at runtime.  std::simd needs
# libstdc++ (Apple libc++ / MSVC do not ship <experimental/simd>).
set(NWGRAD_LEVEL_DIR src/nwgrad/cpp/nwgrad)
set(NWGRAD_LEVEL_SOURCES "")
# register_level lives in simd_levels.cpp, compiled at the plain baseline (no wide
# flags below touch it): its struct build/store runs at static init on every CPU, so
# it must be baseline-legal code.  See the note in simd_levels.hpp.
#
# Windows is built with MinGW-w64 gcc (libstdc++), never MSVC: the level TUs #include
# <experimental/simd>, which MSVC's STL does not ship.  Rather than fail at that include,
# an MSVC (or clang-cl) configure stops here with the way out; see the MinGW block below
# for what the gcc build on Windows needs.
if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
    set(NWGRAD_LEVEL_SOURCES
        ${NWGRAD_LEVEL_DIR}/simd_levels.cpp
        ${NWGRAD_LEVEL_DIR}/level_baseline.cpp
        ${NWGRAD_LEVEL_DIR}/level_sse4.cpp
        ${NWGRAD_LEVEL_DIR}/level_avx2.cpp
        ${NWGRAD_LEVEL_DIR}/level_avx512.cpp
        # the soft kernels of each level, their own TU (level_common.inc, NWGRAD_LEVEL_SOFT_TU)
        ${NWGRAD_LEVEL_DIR}/level_baseline_soft.cpp
        ${NWGRAD_LEVEL_DIR}/level_sse4_soft.cpp
        ${NWGRAD_LEVEL_DIR}/level_avx2_soft.cpp
        ${NWGRAD_LEVEL_DIR}/level_avx512_soft.cpp
        # the inter-pair hard kernels of each level, their own TU (NWGRAD_LEVEL_INTER_TU)
        ${NWGRAD_LEVEL_DIR}/level_baseline_inter.cpp
        ${NWGRAD_LEVEL_DIR}/level_sse4_inter.cpp
        ${NWGRAD_LEVEL_DIR}/level_avx2_inter.cpp
        ${NWGRAD_LEVEL_DIR}/level_avx512_inter.cpp)
    # baseline is the x86-64 default (SSE2); the other levels get their flags per-file.
    # sse4 is the x86-64-v2 feature set (SSE4.2 + POPCNT), spelled as -m flags like the
    # others so no -march/-mtune change rides along.
    # The JCC padding below is for Skylake-derived cores, which all run the avx2 level or
    # higher; baseline and sse4 run on the others -- Piledriver, pre-Haswell Intel, Atoms.
    # gas pads with up to 5 segment prefixes on the instructions before a jump, and
    # Bulldozer-family decoders slow down on more than three prefixes: cs cs cs in front of
    # a 66-prefixed SSE op is four.  On skynet (Piledriver) an unrelated 30-byte shift put
    # them into the soft linear fill's hot loop, 1.1-1.2x slower.  These TUs and their soft
    # halves cap the padding at 2 prefixes per instruction (66 + 2 = 3).  skynet, bench_int at 1 thread
    # (hard, score, stream; 5 types, DNA and protein): one arm in five 1.2-4x faster (local
    # affine float32/double hard 0.32-0.46, DNA int16 score 0.25), the cap at 0 (all NOPs)
    # and 1 the same.  On cudak (Westmere; not a target) NOPs cost up to 1.15x, the cap at 2
    # less: baseline worst 1.05, sse4 1.14 in one arm, 10 of 60 above 1.03.
    set(NWGRAD_PAD_CAP_OPTS "$<$<CXX_COMPILER_ID:GNU>:-Wa,-malign-branch-prefix-size=2>")
    set_source_files_properties(${NWGRAD_LEVEL_DIR}/level_baseline.cpp ${NWGRAD_LEVEL_DIR}/level_baseline_soft.cpp ${NWGRAD_LEVEL_DIR}/level_baseline_inter.cpp PROPERTIES
        COMPILE_OPTIONS "${NWGRAD_PAD_CAP_OPTS}")
    set_source_files_properties(${NWGRAD_LEVEL_DIR}/level_sse4.cpp ${NWGRAD_LEVEL_DIR}/level_sse4_soft.cpp ${NWGRAD_LEVEL_DIR}/level_sse4_inter.cpp PROPERTIES
        COMPILE_OPTIONS "-msse4.2;-mpopcnt;${NWGRAD_PAD_CAP_OPTS}")
    # The avx2 and avx512 levels are the ones Skylake-derived cores run (6th-10th gen Core,
    # Skylake-SP/Cascade Lake Xeons), whose JCC-erratum microcode keeps any jump that crosses
    # or ends on a 32-byte boundary out of the uop cache: a hot loop that lands there runs from
    # the legacy decoder.  Which kernel lands there is layout luck — on oldsolace (i7-7700HQ),
    # shifting the avx2 TU by 0..30 bytes put nearly every hot kernel at 50-97% legacy decode
    # at 2-3 of 16 offsets, 1.2-1.7x slower (inter_fill int16/int32 1.6x, linear f64 1.4x).
    # The assembler's padding keeps every jump clear: legacy decode 0-7% at all offsets.  On
    # gcc it comes with its own loop/jump alignment off, which otherwise pins small loops to
    # fixed offsets and, combined with the padding, cost nighthaven float32 cells 6-9%; as here,
    # nighthaven geomean 1.002 over 116 cells (worst 0.966), solace 0.98-1.02.
    #
    # Under gcc it covers every TU of the extension (applied to the target below), not only
    # the level TUs: the scalar fills in py_exports.cpp have the same exposure, and padding
    # only the level TUs moves them by itself.  Every TU vs level TUs only: oldsolace 1.007
    # (42 cells faster, none slower), nighthaven 1.000 (worst 0.986).  Under clang it stays on
    # the level TUs: its padding is NOPs, some inside hot loops, and the gain is a reshuffle —
    # level TUs vs none, oldsolace 1.001 (28 faster, 16 slower), nighthaven 0.997 (worst 0.94);
    # every TU, nighthaven float32 cells down to 0.952.
    set(NWGRAD_JCC_GNU_OPTS
        "-Wa,-mbranches-within-32B-boundaries;-falign-functions=32;-falign-loops=1;-falign-jumps=1;-falign-labels=1")
    set(NWGRAD_JCC_CLANG_OPTS "$<$<CXX_COMPILER_ID:Clang,AppleClang>:-mbranches-within-32B-boundaries>")
    set_source_files_properties(${NWGRAD_LEVEL_DIR}/level_avx2.cpp ${NWGRAD_LEVEL_DIR}/level_avx2_soft.cpp ${NWGRAD_LEVEL_DIR}/level_avx2_inter.cpp PROPERTIES
        COMPILE_OPTIONS "-mavx2;-mfma;${NWGRAD_JCC_CLANG_OPTS}")
    set_source_files_properties(${NWGRAD_LEVEL_DIR}/level_avx512.cpp ${NWGRAD_LEVEL_DIR}/level_avx512_soft.cpp ${NWGRAD_LEVEL_DIR}/level_avx512_inter.cpp PROPERTIES
        COMPILE_OPTIONS "-mavx512f;-mavx512dq;-mavx512vl;-mavx512bw;${NWGRAD_JCC_CLANG_OPTS}")
    # MinGW: the Win64 ABI only guarantees a 16-byte-aligned stack, and older gcc spilled
    # 32- and 64-byte vectors there with ALIGNED moves (vmovapd / vmovdqa64; GCC PR 54412)
    # — a fault the first time a kernel spills at the wrong offset.  gcc 13 and 16 already
    # emit only unaligned forms in these TUs (checked: no vmovap*/vmovdqa* with a memory
    # operand in either level object), so this is insurance for the older gcc a source
    # build may use (README: GCC 11+): the assembler rewrites every aligned vector move
    # into its unaligned twin (binutils >= 2.38), free on aligned data on any AVX core.
    # Only the 32/64-byte TUs need it; the 16-byte SSE levels are aligned by the ABI.
    if (MINGW)
        foreach(wide_tu level_avx2.cpp level_avx2_soft.cpp level_avx512.cpp level_avx512_soft.cpp)
            set_property(SOURCE ${NWGRAD_LEVEL_DIR}/${wide_tu} APPEND PROPERTY
                COMPILE_OPTIONS "-Wa,-muse-unaligned-vector-move")
        endforeach()
    endif()
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    # NEON is mandatory baseline on AArch64 — one level, no runtime dispatch.
    set(NWGRAD_LEVEL_SOURCES
        ${NWGRAD_LEVEL_DIR}/simd_levels.cpp
        ${NWGRAD_LEVEL_DIR}/level_neon.cpp
        ${NWGRAD_LEVEL_DIR}/level_neon_soft.cpp
        ${NWGRAD_LEVEL_DIR}/level_neon_inter.cpp)
endif()

# Placement experiment knob (off by default; see placement_shift.cpp): shifts all of the
# extension's code by N bytes without changing an instruction, to measure what code
# placement alone does to a timing.  Its TU goes first on the link line.
set(NWGRAD_PLACEMENT_SHIFT "" CACHE STRING
    "Experiment: shift all extension code by this many bytes (empty = off)")
set(NWGRAD_PLACEMENT_SHIFT_AT "front" CACHE STRING "Experiment: where the shift pad goes (front | text | mid | grp)")
set(NWGRAD_PLACEMENT_SOURCES "")
if (NOT NWGRAD_PLACEMENT_SHIFT STREQUAL "")
    set(NWGRAD_PLACEMENT_SOURCES src/nwgrad/cpp/nwgrad/placement_shift.cpp)
    set_source_files_properties(src/nwgrad/cpp/nwgrad/placement_shift.cpp PROPERTIES
        COMPILE_DEFINITIONS "NWGRAD_PLACEMENT_SHIFT=${NWGRAD_PLACEMENT_SHIFT};$<$<NOT:$<STREQUAL:${NWGRAD_PLACEMENT_SHIFT_AT},front>>:NWGRAD_PLACEMENT_SHIFT_TEXT>;$<$<STREQUAL:${NWGRAD_PLACEMENT_SHIFT_AT},grp>:NWGRAD_PLACEMENT_SHIFT_GRP>")
endif()

# AT=mid links the pad between the bindings and the level units instead ("a level unit
# linked earlier grew by N": the kernels move against the bindings' code, not with it).
set(NWGRAD_PLACEMENT_SOURCES_MID "")
if (NWGRAD_PLACEMENT_SHIFT_AT STREQUAL "mid")
    set(NWGRAD_PLACEMENT_SOURCES_MID ${NWGRAD_PLACEMENT_SOURCES})
    set(NWGRAD_PLACEMENT_SOURCES "")
endif()

nanobind_add_module(nwgrad_ext
    NB_STATIC NOMINSIZE
    ${NWGRAD_PLACEMENT_SOURCES}
    src/nwgrad/cpp/nwgrad/py_exports.cpp
    src/nwgrad/cpp/nwgrad/logistic/py_logistic.cpp
    ${NWGRAD_PLACEMENT_SOURCES_MID}
    ${NWGRAD_LEVEL_SOURCES})
if (DEFINED NWGRAD_JCC_GNU_OPTS)
    target_compile_options(nwgrad_ext PRIVATE "$<$<CXX_COMPILER_ID:GNU>:${NWGRAD_JCC_GNU_OPTS}>")
endif()
# ── Code placement (ELF, GNU ld) ─────────────────────────────────────────────
# A kernel's speed depends on its address mod 4096 (nighthaven: up to 1.14x with identical
# instructions; AGENTS.md "Code placement"), and with the default script that address was the
# size of everything linked before it, so a change anywhere moved every kernel after it.
# nwgrad_text.ld (generated by tools/gen_text_ld.py) gives each translation unit its own block
# starting at a fixed offset within a page -- the offset it had in main's own link -- so a unit
# keeps its addresses mod 4096 whatever the other units do.  "groups" is the finer variant
# (nwgrad_text_groups.ld); "off" links as before.  macOS (ld64) and Windows (PE) link as before.
set(NWGRAD_KERNEL_LAYOUT "units" CACHE STRING "Code layout script (GNU ld, ELF): units | groups | off")
if (NWGRAD_KERNEL_LAYOUT STREQUAL "ON")
    set(NWGRAD_KERNEL_LAYOUT "units")
endif()
if (NWGRAD_KERNEL_LAYOUT MATCHES "^(units|groups)$" AND NOT APPLE AND NOT WIN32
        AND NOT CMAKE_VERSION VERSION_LESS 3.18)
    include(CheckLinkerFlag)
    if (NWGRAD_KERNEL_LAYOUT STREQUAL "groups")
        set(NWGRAD_TEXT_LD "${CMAKE_CURRENT_SOURCE_DIR}/src/nwgrad/cpp/nwgrad/nwgrad_text_groups.ld")
    elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
        set(NWGRAD_TEXT_LD "${CMAKE_CURRENT_SOURCE_DIR}/src/nwgrad/cpp/nwgrad/nwgrad_text_aarch64.ld")
    else()
        set(NWGRAD_TEXT_LD "${CMAKE_CURRENT_SOURCE_DIR}/src/nwgrad/cpp/nwgrad/nwgrad_text.ld")
    endif()
    check_linker_flag(CXX "-Wl,-T,${NWGRAD_TEXT_LD}" NWGRAD_TEXT_LD_OK)
    if (NWGRAD_TEXT_LD_OK)
        target_link_options(nwgrad_ext PRIVATE "-Wl,-T,${NWGRAD_TEXT_LD}")
        set_property(TARGET nwgrad_ext APPEND PROPERTY LINK_DEPENDS "${NWGRAD_TEXT_LD}")
    endif()
endif()

# clang's SLP vectorizer, in the TU that holds the scalar code (every Aligner fill,
# traceback and walk; scalar_fallback; the linear Pointers fill at every level): it packs
# pairs of independent scalar chains (the X and Y updates, the argmax operands) into
# 2-wide vectors with shuffles on the serial carry, or extracts a packed max through a
# movmsk and a branch.  Off there (nighthaven, clang 22, scalar_fallback): score-only
# 1.9-2.6x faster, affine DNA fills 1.5x, protein Pointers 1.6x.  The per-ISA level TUs
# keep it: the float32 std::simd kernels lose 1.35-1.6x without it.  gcc is unaffected.
#
# And, on x86, its cmov converter (X86CmovConversion): it turns a cmov on a loop's
# critical path into a branch, which in these fills is data-dependent — the Local clamp
# max(v, 0) and the left-carry max, where unrelated proteins clamp to 0 at random.
# Local linear int64 hard, protein L300: clang 0.50x gcc on nighthaven, 0.82x at sse4
# (the scalar H fill; gcc keeps every cmov).  Source forms do not survive it: the clamp as
# v & ~(v >> 63) is folded back into smax, and moving the clamp off the carry made it
# convert the carry's max instead.  Header-only users building with clang can pass the
# same -mllvm flag.
set(NWGRAD_SCALAR_TU_CLANG_OPTS -fno-slp-vectorize)
if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64|i.86")
    list(APPEND NWGRAD_SCALAR_TU_CLANG_OPTS -mllvm -x86-cmov-converter=false)
endif()
set_source_files_properties(src/nwgrad/cpp/nwgrad/py_exports.cpp PROPERTIES
    COMPILE_OPTIONS "$<$<CXX_COMPILER_ID:Clang,AppleClang>:${NWGRAD_SCALAR_TU_CLANG_OPTS}>")

# MinGW: link the gcc runtime (libstdc++, libgcc, winpthread) statically, so the .pyd
# imports nothing but Windows system DLLs, the UCRT and python3XY.dll.  Shipped wheels
# would otherwise need libstdc++-6.dll & co. on the user's PATH — and whichever copies
# another MinGW-built extension put there first.  The wheel workflow checks the result.
# The test and benchmark executables likewise, so they run without the runtime DLLs on PATH.
if (MINGW)
    target_link_options(nwgrad_ext PRIVATE -static)
    string(APPEND CMAKE_EXE_LINKER_FLAGS " -static")
endif()

# Add -Wall -Wextra for Unix-style compilers (GCC/Clang)
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(nwgrad_ext PRIVATE -Wall -Wextra)
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(nwgrad_ext PRIVATE DEBUG_MODE)
    target_compile_options(nwgrad_ext PRIVATE -g -Og)
    # Range-checked containers for any Debug build.  These change no ABI and need
    # no sanitizer, so they are safe on a plain Debug build.  The ASan-only vector
    # annotations (_GLIBCXX_SANITIZE_VECTOR — which link-error without ASan) and the
    # clang-vs-gcc handling live in the sanitizer block above, applied whenever a
    # sanitizer is on.  (This line previously carried -DGLIBCXX_DEBUG, a typo for
    # _GLIBCXX_DEBUG that expanded to nothing — dropped, not fixed: real _GLIBCXX_DEBUG
    # changes the container ABI and is mutually exclusive with the vector annotations.)
    target_compile_definitions(nwgrad_ext PRIVATE
        _GLIBCXX_ASSERTIONS
        $<$<CXX_COMPILER_ID:Clang,AppleClang>:_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_DEBUG>)
endif()

install(TARGETS nwgrad_ext LIBRARY DESTINATION nwgrad)
install(DIRECTORY src/nwgrad/cpp/nwgrad DESTINATION nwgrad/cpp
    FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h")

# ── Sanitizers (opt-in; CI builds the C++ tests with these) ──────────────────
#
# -fno-sanitize-recover=all is the flag that makes this worth anything: by
# default UBSan prints a diagnostic and carries on, so the test binary would
# still exit 0 and CI would stay green on undefined behaviour.  With it, the
# first violation aborts and ctest fails.
option(NWGRAD_SANITIZE
    "Build the C++ tests with AddressSanitizer + UndefinedBehaviorSanitizer" OFF)

# The C++ tests exercise the headers.  They cannot exercise the *bindings*: in
# C++ the caller owns the AlignParams and scoping keeps it honest, so the
# bare-pointer-with-no-owner hazard simply does not arise there.  It arises only
# once Python holds the other end of the lifetime — which is exactly the layer
# that was never under a sanitizer.  This builds the extension itself with them,
# so pytest runs against an instrumented nwgrad_ext.
option(NWGRAD_SANITIZE_EXT
    "Build the Python extension (nwgrad_ext) with sanitizers" OFF)

# Which sanitizers, for either option.  Split so ASan and UBSan can run as
# separate CI jobs: ASan needs an LD_PRELOAD dance and must give up leak
# detection inside CPython, UBSan needs neither.
#
# "thread" (TSan) is the third configuration, and it must be its own build: TSan
# cannot be combined with ASan.  The batch workers hand SeqPairs, DP buffers and
# summed gradients across threads; TSan is what sees two workers touching one of
# them, which ASan and UBSan cannot.  Verified live: it reports the duplicate-pair
# race (two workers in one Aligner::set_problem) at its first collision.
set(NWGRAD_SANITIZERS "address,undefined" CACHE STRING
    "Comma-separated -fsanitize list for NWGRAD_SANITIZE / NWGRAD_SANITIZE_EXT (or \"thread\")")

if (NWGRAD_SANITIZE OR NWGRAD_SANITIZE_EXT)
    if (NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        message(FATAL_ERROR "NWGRAD_SANITIZE requires GCC or Clang")
    endif()
    if (NWGRAD_SANITIZERS MATCHES "thread" AND NWGRAD_SANITIZERS MATCHES "address")
        message(FATAL_ERROR "NWGRAD_SANITIZERS: thread and address cannot be combined; "
                            "configure a separate build for each")
    endif()
    set(NWGRAD_SANITIZE_FLAGS
        -fsanitize=${NWGRAD_SANITIZERS}
        -fno-sanitize-recover=all
        -fno-omit-frame-pointer
        -g)

    # Hardened / range-checked standard library, and — under ASan — container
    # annotations.  Applied to EVERY sanitizer build (both nwgrad_tests and
    # nwgrad_ext, gcc and clang), not just the Debug ext build: the annotations
    # are what catch a read past a vector's .size() through .data() — raw-pointer
    # accesses the range-checked operator[] never sees — and that is precisely the
    # class of bug (a striped-kernel .data() over-read) that a clang C++ job could
    # not see before, because it had neither.
    set(NWGRAD_SANITIZE_DEFS
        _GLIBCXX_ASSERTIONS)                       # libstdc++: range-checked operator[] etc. (no ABI change)
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang")
        # libc++ range checks (its ASan container annotations are automatic).
        list(APPEND NWGRAD_SANITIZE_DEFS _LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_DEBUG)
    endif()
    if (NWGRAD_SANITIZERS MATCHES "address")
        # libstdc++ std::vector ASan annotations.  GCC auto-enables these under
        # -fsanitize=address; clang does NOT define __SANITIZE_ADDRESS__, so
        # libstdc++ leaves them off unless told — which is exactly why the
        # container overflow was invisible to the clang build.  Requires ASan (it
        # calls the ASan annotation runtime), hence gated on the sanitizer list;
        # every TU touching the vector must agree, and here all do.
        list(APPEND NWGRAD_SANITIZE_DEFS _GLIBCXX_SANITIZE_VECTOR)
    endif()
endif()

if (NWGRAD_SANITIZE_EXT)
    target_compile_options(nwgrad_ext PRIVATE ${NWGRAD_SANITIZE_FLAGS})
    target_link_options(nwgrad_ext PRIVATE ${NWGRAD_SANITIZE_FLAGS})
    target_compile_definitions(nwgrad_ext PRIVATE ${NWGRAD_SANITIZE_DEFS})
endif()

# Direct CMake builds retain the developer targets by default. Python package
# builds disable both options in pyproject.toml: the sdist excludes tests/.
option(NWGRAD_BUILD_TESTS "Build the C++ unit tests" ON)
option(NWGRAD_BUILD_BENCHMARKS "Build the C++ benchmarks" ON)

if (NWGRAD_BUILD_TESTS OR NWGRAD_BUILD_BENCHMARKS)
    find_package(Threads REQUIRED)
endif()

if (NWGRAD_BUILD_TESTS)
    # ── C++ unit tests (Catch2 v2, vendored single header) ───────────────────────
    add_executable(nwgrad_tests
        tests/cpp/main.cpp
        tests/cpp/test_subst_matrix.cpp
        tests/cpp/test_align_params.cpp
        tests/cpp/test_aligner.cpp
        tests/cpp/test_alignment_path.cpp
        tests/cpp/test_preconditions.cpp
        tests/cpp/test_gradient.cpp
        tests/cpp/test_batch.cpp
        tests/cpp/test_banded.cpp
        tests/cpp/test_seq_pair.cpp
        tests/cpp/test_traceback_modes.cpp
        tests/cpp/test_dense_codes.cpp
        tests/cpp/test_viterbi_then_fwdbwd.cpp
        tests/cpp/test_buffer_reuse.cpp
        tests/cpp/test_batch_engine.cpp
        tests/cpp/test_hirschberg_scan.cpp
        tests/cpp/test_hirschberg_pmax.cpp
        tests/cpp/test_batch_build.cpp
        tests/cpp/test_interpair.cpp
        tests/cpp/test_weighted_grad.cpp
        tests/cpp/test_logistic.cpp
        tests/cpp/test_scan_fixes.cpp
        tests/cpp/test_score_only.cpp
        tests/cpp/test_int8_inter.cpp
        tests/cpp/test_simd_bitexact.cpp
        tests/cpp/test_simd_bitexact_f32.cpp
        tests/cpp/test_tracks.cpp
        tests/cpp/test_integer.cpp
        tests/cpp/test_linear_striped.cpp
        tests/cpp/test_grid.cpp
        ${NWGRAD_LEVEL_SOURCES}   # per-ISA kernel TUs, so kernel="simd" exercises the dispatched striped path
    )
    target_include_directories(nwgrad_tests PRIVATE
        src/nwgrad/cpp/nwgrad
        tests/cpp
    )
    target_compile_features(nwgrad_tests PRIVATE cxx_std_20)
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        target_compile_options(nwgrad_tests PRIVATE -Wall -Wextra)
    endif()
    target_link_libraries(nwgrad_tests PRIVATE Threads::Threads)

    # Real header-only consumer: no ISA-level TUs or registrars are linked, so
    # every dispatch slot stays empty and alignment exercises the header fallback.
    add_executable(nwgrad_header_tests
        tests/cpp/main.cpp
        tests/cpp/test_header_only.cpp
        tests/cpp/test_buffer_reuse.cpp)
    target_include_directories(nwgrad_header_tests PRIVATE src/nwgrad/cpp/nwgrad tests/cpp)
    target_compile_features(nwgrad_header_tests PRIVATE cxx_std_20)
    target_link_libraries(nwgrad_header_tests PRIVATE Threads::Threads)
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        target_compile_options(nwgrad_header_tests PRIVATE -Wall -Wextra)
    endif()
    # A build of the usual test target must provide every executable ctest runs.
    add_dependencies(nwgrad_tests nwgrad_header_tests)

    # Hardened standard library for the test binary in EVERY build, not only the
    # sanitizer ones: the plain gcc/clang and NEON CI jobs, and any local
    # `cmake -B build`, otherwise ran the tests with unchecked operator[].  Neither
    # macro changes the ABI.  Deliberately not -fhardened: its
    # -ftrivial-auto-var-init=zero would zero exactly the uninitialized reads a test
    # build exists to expose, and _FORTIFY_SOURCE needs optimization.
    foreach(test_target nwgrad_tests nwgrad_header_tests)
        target_compile_definitions(${test_target} PRIVATE
            _GLIBCXX_ASSERTIONS
            $<$<CXX_COMPILER_ID:Clang,AppleClang>:_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_DEBUG>)

        # Sanitizers must be passed to both the compile and the link step.
        if (NWGRAD_SANITIZE)
            target_compile_options(${test_target} PRIVATE ${NWGRAD_SANITIZE_FLAGS})
            target_link_options(${test_target} PRIVATE ${NWGRAD_SANITIZE_FLAGS})
            target_compile_definitions(${test_target} PRIVATE ${NWGRAD_SANITIZE_DEFS})
        endif()
    endforeach()

    enable_testing()
    add_test(NAME cpp_tests COMMAND nwgrad_tests)
    add_test(NAME cpp_header_tests COMMAND nwgrad_header_tests)

    # ── The simd kernel, once per instruction set ────────────────────────────────
    #
    # The ISA is chosen once per process from a function-pointer table, so a single run
    # only ever exercises one of them.  These re-run the bit-exactness sweep with the
    # choice forced, which is what makes the AVX2 / AVX-512 clones *tested* rather than
    # merely compiled: a CI runner with AVX2 executes the AVX2 kernel here.
    #
    # An ISA the CPU cannot run resolves to auto (best available) rather than trapping, so
    # every one of these is safe to run everywhere — it simply re-tests the best level on a
    # machine that lacks the named hardware.  That is why they are unconditional.  Names are
    # the unified backend vocabulary (sse2 replaces the old "baseline"; neon is the AArch64
    # level, a no-op on x86).
    foreach(isa sse2 sse4 avx2 avx512 neon)
        add_test(NAME cpp_tests_isa_${isa} COMMAND nwgrad_tests "[simd]")
        set_tests_properties(cpp_tests_isa_${isa}
            PROPERTIES ENVIRONMENT "NWGRAD_ISA=${isa}")
    endforeach()

    # ── the pointer traceback, once per instruction set ──────────────────────────
    #
    # TracebackMode::Pointers has its own kernel at every level, so "compiled" and
    # "tested" are as far apart for it as for the score-table kernel.  [traceback]
    # carries the pointers-vs-scores equivalence cases; forcing the ISA runs them
    # against each level's pointer kernel in turn.
    foreach(isa sse2 sse4 avx2 avx512 neon)
        add_test(NAME cpp_tests_traceback_${isa} COMMAND nwgrad_tests "[traceback]")
        set_tests_properties(cpp_tests_traceback_${isa}
            PROPERTIES ENVIRONMENT "NWGRAD_ISA=${isa}")
    endforeach()

    # ── tracks, once per instruction set ─────────────────────────────────────────
    #
    # Every tracked leveled kernel against the scalar tracked fill (test_tracks.cpp).
    foreach(isa sse2 sse4 avx2 avx512 neon)
        add_test(NAME cpp_tests_tracks_${isa} COMMAND nwgrad_tests "[tracks]")
        set_tests_properties(cpp_tests_tracks_${isa}
            PROPERTIES ENVIRONMENT "NWGRAD_ISA=${isa}")
    endforeach()

endif()

if (NWGRAD_BUILD_BENCHMARKS)
    # ── C++ stress benchmark ──────────────────────────────────────────────────────
    add_executable(stress_batch tests/cpp/stress_batch.cpp)
    target_include_directories(stress_batch PRIVATE src/nwgrad/cpp/nwgrad)
    target_compile_features(stress_batch PRIVATE cxx_std_20)
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        target_compile_options(stress_batch PRIVATE -O3 -Wall -Wextra)
    endif()
    target_link_libraries(stress_batch PRIVATE Threads::Threads)

    # ── float32 vs double kernel benchmark ────────────────────────────────────────
    # Links the per-ISA level TUs, so kernel=Simd dispatches the real striped kernel at
    # both precisions (viterbi / viterbi_f) rather than the scalar header-only fallback.
    add_executable(bench_f32 tests/cpp/bench_f32.cpp ${NWGRAD_LEVEL_SOURCES})
    target_include_directories(bench_f32 PRIVATE src/nwgrad/cpp/nwgrad)
    target_compile_features(bench_f32 PRIVATE cxx_std_20)
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        target_compile_options(bench_f32 PRIVATE -O3 -Wall -Wextra)
    endif()
    target_link_libraries(bench_f32 PRIVATE Threads::Threads)

    # ── score-only kernel benchmark (every scalar type, the integer arms included) ───
    add_executable(bench_score tests/cpp/bench_score.cpp ${NWGRAD_LEVEL_SOURCES})
    target_include_directories(bench_score PRIVATE src/nwgrad/cpp/nwgrad)
    target_compile_features(bench_score PRIVATE cxx_std_20)
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        target_compile_options(bench_score PRIVATE -O3 -Wall -Wextra)
    endif()
    target_link_libraries(bench_score PRIVATE Threads::Threads)
endif()
