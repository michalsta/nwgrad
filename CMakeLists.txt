cmake_minimum_required(VERSION 3.15)

project(nwgrad LANGUAGES CXX)
find_package(Python 3.8
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)
find_package(nanobind REQUIRED CONFIG)

set(CMAKE_CXX_STANDARD 20)

nanobind_add_module(nwgrad_ext
    NB_STATIC NOMINSIZE
    src/nwgrad/cpp/nwgrad/py_exports.cpp)

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

if (NWGRAD_SANITIZE)
    if (NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        message(FATAL_ERROR "NWGRAD_SANITIZE requires GCC or Clang")
    endif()
    set(NWGRAD_SANITIZE_FLAGS
        -fsanitize=address,undefined
        -fno-sanitize-recover=all
        -fno-omit-frame-pointer
        -g)
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

# ── C++ stress benchmark ──────────────────────────────────────────────────────
add_executable(stress_batch tests/cpp/stress_batch.cpp)
target_include_directories(stress_batch PRIVATE src/nwgrad/cpp/nwgrad)
target_compile_features(stress_batch PRIVATE cxx_std_20)
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(stress_batch PRIVATE -O3 -Wall -Wextra)
endif()
target_link_libraries(stress_batch PRIVATE Threads::Threads)
