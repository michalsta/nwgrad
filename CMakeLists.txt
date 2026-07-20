cmake_minimum_required(VERSION 3.15)

project(nwgrad LANGUAGES CXX)
find_package(Python 3.8
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)
find_package(nanobind REQUIRED CONFIG)

set(CMAKE_CXX_STANDARD 20)

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
# There is deliberately no MSVC special-case here.  The level TUs #include
# <experimental/simd>, which MSVC's STL does not ship, so a source build under MSVC fails
# at that include — an honest compile error, not a silently-degraded scalar wheel.  The
# library does not support Windows: its reason to exist is the vectorized kernels, and a
# scalar-only build there would be slower than tools already available.  The gate is on
# the CPU only, never the compiler, so the day MSVC ships <experimental/simd> this begins
# to work with no change (mingw/gcc, which use libstdc++, already do).
if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
    set(NWGRAD_LEVEL_SOURCES
        ${NWGRAD_LEVEL_DIR}/simd_levels.cpp
        ${NWGRAD_LEVEL_DIR}/level_baseline.cpp
        ${NWGRAD_LEVEL_DIR}/level_avx2.cpp
        ${NWGRAD_LEVEL_DIR}/level_avx512.cpp)
    # baseline is the x86-64 default (SSE2); the wider levels get their flags per-file
    set_source_files_properties(${NWGRAD_LEVEL_DIR}/level_avx2.cpp    PROPERTIES
        COMPILE_OPTIONS "-mavx2;-mfma")
    set_source_files_properties(${NWGRAD_LEVEL_DIR}/level_avx512.cpp  PROPERTIES
        COMPILE_OPTIONS "-mavx512f;-mavx512dq;-mavx512vl;-mavx512bw")
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    # NEON is mandatory baseline on AArch64 — one level, no runtime dispatch.
    set(NWGRAD_LEVEL_SOURCES
        ${NWGRAD_LEVEL_DIR}/simd_levels.cpp
        ${NWGRAD_LEVEL_DIR}/level_neon.cpp)
endif()

nanobind_add_module(nwgrad_ext
    NB_STATIC NOMINSIZE
    src/nwgrad/cpp/nwgrad/py_exports.cpp
    ${NWGRAD_LEVEL_SOURCES})

# Add -Wall -Wextra for Unix-style compilers (GCC/Clang)
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(nwgrad_ext PRIVATE -Wall -Wextra)
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(nwgrad_ext PRIVATE DEBUG_MODE)
    target_compile_options(nwgrad_ext PRIVATE -g -Og -D_GLIBCXX_ASSERTIONS -D_GLIBCXX_SANITIZE_VECTOR -DGLIBCXX_DEBUG -D_LIBCPP_ENABLE_ASSERTIONS=1)
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
set(NWGRAD_SANITIZERS "address,undefined" CACHE STRING
    "Comma-separated -fsanitize list for NWGRAD_SANITIZE / NWGRAD_SANITIZE_EXT")

if (NWGRAD_SANITIZE OR NWGRAD_SANITIZE_EXT)
    if (NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        message(FATAL_ERROR "NWGRAD_SANITIZE requires GCC or Clang")
    endif()
    set(NWGRAD_SANITIZE_FLAGS
        -fsanitize=${NWGRAD_SANITIZERS}
        -fno-sanitize-recover=all
        -fno-omit-frame-pointer
        -g)
endif()

if (NWGRAD_SANITIZE_EXT)
    target_compile_options(nwgrad_ext PRIVATE ${NWGRAD_SANITIZE_FLAGS})
    target_link_options(nwgrad_ext PRIVATE ${NWGRAD_SANITIZE_FLAGS})
endif()

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
    tests/cpp/test_add_many.cpp
    tests/cpp/test_simd_bitexact.cpp
    tests/cpp/test_simd_bitexact_f32.cpp
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
find_package(Threads REQUIRED)
target_link_libraries(nwgrad_tests PRIVATE Threads::Threads)

# Sanitizers must be passed to both the compile and the link step.
if (NWGRAD_SANITIZE)
    target_compile_options(nwgrad_tests PRIVATE ${NWGRAD_SANITIZE_FLAGS})
    target_link_options(nwgrad_tests PRIVATE ${NWGRAD_SANITIZE_FLAGS})
endif()

enable_testing()
add_test(NAME cpp_tests COMMAND nwgrad_tests)

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
foreach(isa sse2 avx2 avx512 neon)
    add_test(NAME cpp_tests_isa_${isa} COMMAND nwgrad_tests "[simd]")
    set_tests_properties(cpp_tests_isa_${isa}
        PROPERTIES ENVIRONMENT "NWGRAD_ISA=${isa}")
endforeach()

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
