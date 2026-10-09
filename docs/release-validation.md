# 0.6 release validation

NEON checks completed on 2026-10-09 against candidate
`e0e434b3eecba893ed489941637e018cc3ac1a6d`.

Host: spot, Apple M1, macOS 26.6.2, Homebrew GCC 16.2.0, Python 3.13.16.
The installed extension reported `neon gcc 16.2.0`. The committed source was
copied into an isolated directory, leaving spot's existing trees untouched:
`/Users/untrusted/nwgrad-release-e0e434b.rsNtRo`.

| Check | Result |
|---|---|
| C++ Release suite, including the new header-only consumer | 10/10 CTest groups passed |
| Extended score fuzz, including negative gap costs | 800,392 assertions in 4 test cases passed |
| Python suite, default backend | 2,701 passed, 15 skipped |
| Python suite, forced NEON, including batch methods | 2,701 passed, 15 skipped |
| Unified affine walk against the six pre-refactor walkers | 1,600,000 comparison cases, zero differences |

The C++ build used `-ffp-contract=off` and `_GLIBCXX_ASSERTIONS`; this was a
Release build without sanitizers. The header-only consumer's ASan/UBSan check
passed separately on the local x86 host before this fleet run. The 15 Python
skips comprise four Linux-only thread-limit checks, two linear-Hirschberg
coordinate fixtures, seven non-unique-optimum gradient checks, and two
sdist-content checks because `build` is not installed. The copied tree also has
no `.git`; the sdist-content checks passed locally in the earlier release audit.

Commands after installing the package and configuring the C++ Release build:

```bash
.venv/bin/ctest --test-dir build-tests --output-on-failure -j2
NWGRAD_SCORE_FUZZ=20000 NWGRAD_ISA=neon ./build-tests/nwgrad_tests "[score]"
.venv/bin/python -m pytest tests/Python -q
NWGRAD_FORCE_KERNEL=simd .venv/bin/python -m pytest tests/Python -q
```

The differential harness compiled the current `aligner.hpp` beside `5d97c38`'s
header in namespace `oldnw`, linked the current `level_neon.cpp` and
`simd_levels.cpp`, and used `-O2 -ffp-contract=off -fopenmp-simd`. Its old SIMD
header was renamed and given a distinct comment to avoid GCC's pragma-once
coalescing. Four runs of 50,000 iterations used seeds 20261009–20261012, eight
comparison cases per iteration. They covered scalar and NEON, float32 and double,
global and local, full and narrow banded problems, Scores/Pointers/Auto traceback,
empty sequences, ties and non-representable gap costs. Scores, aligned strings,
index paths, guides and both gradient consumers matched, as did 29,408 rejected
cases. The harness and logs remain under the isolated directory above.

## Remaining release checks

- Run the corresponding extended score fuzz and old-walker comparison on AVX-512.
  Solace's SSH tunnel refused connections during the preceding fleet-access check.
- Run the Wheels workflow manually against the final candidate. This verifies
  distributable wheels across platforms and Python versions, including macOS
  runtime linking; editable-install tests do not replace that check.

The GIL release, AArch64 FMA performance measurement, scheduling heuristics and
other performance items in `TODO.md` remain follow-up work.
