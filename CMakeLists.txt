cmake_minimum_required(VERSION 3.15)

project(nwgrad LANGUAGES CXX)
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
    src/nwgrad/cpp/nwgrad/logistic/py_logistic.cpp
    ${NWGRAD_LEVEL_SOURCES})

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
        tests/cpp/test_viterbi_then_fwdbwd.cpp
        tests/cpp/test_buffer_reuse.cpp
        tests/cpp/test_batch_engine.cpp
        tests/cpp/test_hirschberg_scan.cpp
        tests/cpp/test_hirschberg_pmax.cpp
        tests/cpp/test_add_many.cpp
        tests/cpp/test_interpair.cpp
        tests/cpp/test_weighted_grad.cpp
        tests/cpp/test_logistic.cpp
        tests/cpp/test_scan_fixes.cpp
        tests/cpp/test_score_only.cpp
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
    target_link_libraries(nwgrad_tests PRIVATE Threads::Threads)

    # Hardened standard library for the test binary in EVERY build, not only the
    # sanitizer ones: the plain gcc/clang and NEON CI jobs, and any local
    # `cmake -B build`, otherwise ran the tests with unchecked operator[].  Neither
    # macro changes the ABI.  Deliberately not -fhardened: its
    # -ftrivial-auto-var-init=zero would zero exactly the uninitialized reads a test
    # build exists to expose, and _FORTIFY_SOURCE needs optimization.
    target_compile_definitions(nwgrad_tests PRIVATE
        _GLIBCXX_ASSERTIONS
        $<$<CXX_COMPILER_ID:Clang,AppleClang>:_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_DEBUG>)

    # Sanitizers must be passed to both the compile and the link step.
    if (NWGRAD_SANITIZE)
        target_compile_options(nwgrad_tests PRIVATE ${NWGRAD_SANITIZE_FLAGS})
        target_link_options(nwgrad_tests PRIVATE ${NWGRAD_SANITIZE_FLAGS})
        target_compile_definitions(nwgrad_tests PRIVATE ${NWGRAD_SANITIZE_DEFS})
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

    # ── the pointer traceback, once per instruction set ──────────────────────────
    #
    # TracebackMode::Pointers has its own kernel at every level, so "compiled" and
    # "tested" are as far apart for it as for the score-table kernel.  [traceback]
    # carries the pointers-vs-scores equivalence cases; forcing the ISA runs them
    # against each level's pointer kernel in turn.
    foreach(isa sse2 avx2 avx512 neon)
        add_test(NAME cpp_tests_traceback_${isa} COMMAND nwgrad_tests "[traceback]")
        set_tests_properties(cpp_tests_traceback_${isa}
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
endif()
