# C++ library and building from source

## C++ header-only library

The C++ implementation is header-only (`src/nwgrad/cpp/nwgrad/`). To use it directly from C++:

```bash
python -m nwgrad --include
# prints: /path/to/nwgrad/cpp
```

Add that path to your include path and `#include "nwgrad/aligner.hpp"` etc.

## Building from source

Requirements: Python ≥ 3.10, `nanobind>=3.0`, `scikit-build-core`, and a C++20
compiler with libstdc++'s `<experimental/simd>` (GCC 11+, or Clang 13+ built against
libstdc++).

```bash
pip install scikit-build-core "nanobind>=3.0"
pip install -e ".[dev]"
pytest tests/Python/
```

Python package builds compile only the extension. Direct CMake builds enable the
C++ tests and benchmarks by default; use `-DNWGRAD_BUILD_TESTS=OFF` and
`-DNWGRAD_BUILD_BENCHMARKS=OFF` to disable them.

C++ unit tests (requires CMake). The extension target does `find_package(nanobind)`,
so even a tests-only configure needs nanobind discoverable:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug \
      -Dnanobind_DIR="$(python -m nanobind --cmake_dir)"
cmake --build build
ctest --test-dir build
```

`-DNWGRAD_SANITIZE=ON` builds the C++ tests with AddressSanitizer and
UndefinedBehaviorSanitizer (`-fno-sanitize-recover=all`, so the first violation
fails the run rather than printing and carrying on). CI runs this under both GCC
and Clang.

`-DNWGRAD_SANITIZE=ON -DNWGRAD_SANITIZERS=thread` builds them with ThreadSanitizer
instead (a separate build: TSan cannot be combined with ASan). Every build of the C++
tests also enables libstdc++'s `_GLIBCXX_ASSERTIONS` bounds checks, and everything is
compiled with `-ffp-contract=off`, so no SIMD level can fuse an FMA the scalar path
does not.

`nwgrad_header_tests` links no ISA kernel translation units. It checks that the
dispatch table is empty, then exercises alignment and DP-buffer reuse through
the header-only fallback. CI runs this target with ASan and UBSan:

```bash
cmake --build build-san --target nwgrad_header_tests
ctest --test-dir build-san --output-on-failure -R '^cpp_header_tests$'
```
