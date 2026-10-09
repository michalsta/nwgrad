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

## Removal of the API deprecated in 0.6.0

Checked on the local x86 host on 2026-10-09 against `389a5ac`, GCC 16.2.1,
Python 3.14.7, SSE2 runtime dispatch. The untyped batch state and deprecated
wrappers are removed; supported pair methods still preserve their gradient
lifecycle. In this build, `sizeof(SeqPairBatch)` fell from 832 to 752 bytes;
`sizeof(SeqPair)` remains 24 bytes. This measures object size, not throughput.

| Check | Result |
|---|---|
| C++ Debug suite | 10/10 CTest groups passed |
| Python suite, default backend | 2,692 passed, 9 skipped |
| Python suite, forced scalar fallback | 2,692 passed, 9 skipped |
| Full C++ suite and header-only consumer, ASan/UBSan | Both passed |
| DiscrimAlign PR #16 fast tests | 1,145 passed, 31 deselected, 1 expected failure |
| DiscrimAlign PR #16 before/after acceptance fits | 16/16 result fingerprints and final log likelihoods identical |

The consumer checkout was a detached worktree of
[DiscrimAlign PR #16](https://github.com/BioGeMT/DiscrimAlign/pull/16), commit
`778c7a74c6b1b75272c5e7c3b9b9a829d08c1452`, at
`/tmp/discrimalign-pr16-api`; its source was not modified. Its fast test command
was `pytest tests -q -m 'not slow' --ignore=tests/test_mirbench.py`.

The acceptance matrix used 3,000 Manakov pairs (sample seed 7), four problem
types (local/affine/general, global/affine/symmetric, local/linear/simple,
global/linear/general), striped/interpair fills, and returned alignments on/off.
Each fit ran six iterations, eight threads, and constant step size 5e-4. The
same PR source and fixtures ran before and after the API removal. The PR predates
the optimizer-portfolio branch, so the old optimizer-specific acceptance harness
was adapted to these 16 configurations supported by the PR. Exact fingerprints
covered all returned fields, including learned parameters, scores and paths.

Logs and comparison results remain under `/tmp/nwgrad-api-*`,
`/tmp/discrimalign-pr16-tests.log`, and `/tmp/nwgrad-pr16-{before,after}.json`.
The preceding NEON results describe their explicitly named earlier candidate;
this API-removal validation is local.

## Remaining release checks

- Run the corresponding extended score fuzz and old-walker comparison on AVX-512.
  Solace's SSH tunnel refused connections during the preceding fleet-access check.
- Run the Wheels workflow manually against the final candidate. This verifies
  distributable wheels across platforms and Python versions, including macOS
  runtime linking; editable-install tests do not replace that check.

The GIL release, AArch64 FMA performance measurement, scheduling heuristics and
other performance items in `TODO.md` remain follow-up work.
