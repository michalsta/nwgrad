# Release validation

## 0.7 release validation

Checked on 2026-10-19 against candidate
`948f9bed0e65057f64b97bf847872f4470f801d2` (version 0.7.0), on skynet (x86,
GCC 16.2.1, runtime dispatch `sse4`). The candidate was exported with
`git archive` and installed non-editably into fresh virtual environments; the
consumer checkouts were not modified. CI on `main` passed for this commit.

| Check | Result |
|---|---|
| DiscrimAlign `main` (`8f564c2`, after the PR #16 port) fast tests, Python 3.14.7, numpy 2.5.4, scipy 1.18.1 | 1,152 passed, 31 deselected, 1 expected failure |
| The same with `-W 'error:SeqPairBatch.add_many'` | 131 failed, 1,021 passed: DiscrimAlign `main` still calls `add_many()` |
| PosimiR `nwgrad` branch (`bcaddc6`) tests, Python 3.12.14, numpy 1.26.4, numba 0.66.0, scikit-learn 1.9.0 | 33 passed |

DiscrimAlign's fast test command was
`pytest tests -q -m 'not slow' --ignore=tests/test_mirbench.py`; PosimiR's was
`pytest tests -q`. DiscrimAlign `main` no longer uses the API removed in 0.7
(`alloc_dp`, `align_full`, per-add problem-type arguments). It works with 0.7
through the deprecated `add_many()` and will break in 0.8 unless it moves to the
pair constructor or bounds `nwgrad<0.8`; its `pyproject.toml` requires
`nwgrad>=0.6.0` without an upper bound.

### Remaining 0.7 release checks

- Run the Wheels workflow manually against the final candidate. Its last run was
  the `v0.6.0` tag.
- Run the extended score fuzz and the old-walker comparison on AVX-512 (solace).
  They were not run for 0.6 either, CI has no AVX-512 runner, and `main` has
  changed AVX-512 code since 0.6 (`cc/avx512-lane-shift`).
- Optionally repeat the NEON checks below on spot. CI's `ubuntu-24.04-arm` job
  runs the Python suite under NEON but not the extended score fuzz.

## 0.6 release validation

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

### Removal of the API deprecated in 0.6.0

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

### Follow-up cleanup and optional pair paths

Checked locally on 2026-10-09 with GCC 16.2.1, Python 3.14.7 and SSE2 runtime
dispatch. Changes are recorded in `579462e`, `0559df0`, `ac3fd56`, `cc10315`
and `01a32eb`.

Setter plumbing now shares one macro and Python parameter setters take typed
references. The 2,441 lines of frozen legacy batch headers were replaced by
references computed directly through `Aligner`, retaining 192 stored-batch and
240 streaming configurations. The focused batch regressions passed 1,464,239
assertions.

`SeqPair.score_and_grad(keep_paths=False)` skips stored alignment paths; the
default remains `True`. The separate held-gradient flag is removed. Deferred
gradient visibility is derived from existing score, path and gradient validity
flags, preserving the supported pair lifecycle. A failed scoring fill now
invalidates cached results before it can leave a partial path behind. Twelve
regressions cover recovery after failed full/banded soft fills.

| Check | Result |
|---|---|
| C++ Debug suite | 10/10 CTest groups passed |
| Python suite, default backend | 2,756 passed, 9 skipped |
| Python suite, forced scalar fallback | 2,756 passed, 9 skipped |
| Full C++ suite and header-only consumer, ASan/UBSan | Both passed |
| DiscrimAlign PR #16 acceptance fits against the pre-removal baseline | 16/16 result fingerprints and final log likelihoods identical |

The same unmodified PR #16 checkout and acceptance fixtures described above
were used. Final results are in `/tmp/nwgrad-pr16-followup-final.json`; test
logs are `/tmp/nwgrad-followup-{ctest,python,python-scalar,san-tests}-final.log`.
These follow-up checks were local; the earlier NEON validation is unchanged.

Path-storage timing used one installed binary, CPU affinity, seven alternating
trials and two identical path-enabled control arms. Ratios below are elapsed
time without paths divided by time with paths; the control compares the two
path-enabled arms. A control away from 1 indicates timing drift.

| Gradient | Length | Operation | Without/with paths | Control |
|---|---|---|---|---|
| Hard | 50 | Reuse pair | 0.962 | 1.008 |
| Hard | 50 | Construct and align | 0.947 | 0.939 |
| Hard | 200 | Reuse pair | 0.967 | 1.025 |
| Hard | 200 | Construct and align | 0.866 | 0.860 |
| Soft | 50 | Reuse pair | 0.970 | 1.038 |
| Soft | 50 | Construct and align | 0.976 | 1.035 |
| Soft | 200 | Reuse pair | 0.962 | 0.995 |
| Soft | 200 | Construct and align | 0.978 | 0.981 |

The hard length-50 and soft length-200 reuse fixtures show about 4% less time
with stable controls. Other measurements do not establish a gain above noise;
these results do not support a general throughput claim. Raw measurements are
in `/tmp/nwgrad-followup-path-bench-pinned.jsonl`. The committed benchmark
provides the same three-arm comparison with calibrated repeat counts:

```bash
python tools/bench_pair_paths.py --seq-len 50 200 --cpu 0
```

### Remaining 0.6 release checks

- Run the corresponding extended score fuzz and old-walker comparison on AVX-512.
  Solace's SSH tunnel refused connections during the preceding fleet-access check.
- Run the Wheels workflow manually against the final candidate. This verifies
  distributable wheels across platforms and Python versions, including macOS
  runtime linking; editable-install tests do not replace that check.

The GIL release, AArch64 FMA performance measurement, scheduling heuristics and
other performance items in `TODO.md` remain follow-up work.
