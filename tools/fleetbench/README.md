# fleetbench — nwgrad against other aligners, across a fleet

The harness behind [`docs/benchmarks.md`](../../docs/benchmarks.md). Every tool is timed
on the same cells (fixture × global/local × affine/linear × output × number type ×
threads), each by its best implementation, and every per-pair score is checked against
nwgrad's float64 result before a time counts.

| file | what it does |
|---|---|
| `make_fixtures.py` | builds the fixtures (protein, DNA, miRNA; single- and multithread sets), deterministically |
| `make_bundle.sh` | nwgrad source + this harness + competitor sources at the measured versions, one `.tar.xz` |
| `setup_host.sh` | on a host: two venvs (nwgrad by gcc and by clang, release-wheel flags), PyPI competitors, every C/C++ driver |
| `runner.py` | plans every cell and runs it; nwgrad's default first, then every arm with a 10× budget; resumes from `results/<host>.jsonl` |
| `arm.py` | times one Python arm (nwgrad's Python API, Biopython, scikit-bio, pyopal, edlib, DeepBLAST) |
| `drivers/` | the C/C++ arms: nwgrad's C++ API, parasail (the wheel's own `libparasail.so`), SeqAn3, SeqAn 2, SSW, KSW2, WFA2-lib |
| `problems.py` | the scoring problems; all-integer, so nwgrad's float64 score is exact |
| `coordinator.py` | drives a fleet over ssh: ships, sets up, starts / resumes runners, pauses on foreign load, keeps laptops awake, syncs results |
| `report.py` | `preliminary-results.md`: progress, ETA, per-cell tables |
| `doc_tables.py`, `build_doc.py`, `benchmarks_skeleton.md` | regenerate `docs/benchmarks.md` from the results |
| `ab_setup.sh`, `ab_pr.py`, `ab_report.py` | A/B of a patched nwgrad against the measured one (the PR #12 section) |
| `fixup_rerun.sh`, `fixup_wfa_edit.sh` | re-run a finished host's arms after a harness fix (used once each in the published run) |
| `results/*.jsonl.xz` | every arm of the published run (one JSON object per arm), and the A/B runs (`ab_*`) |

A host's root directory holds `nwgrad/`, `bench/` (this directory), `ext/` and `fixtures/`:

```bash
tools/fleetbench/make_bundle.sh /tmp/bundle.tar.xz
python tools/fleetbench/make_fixtures.py --fasta GCF_000001405.40_GRCh38.p14_protein.faa \
       --manakov manakov_fit_rc.tsv --out /tmp/fixtures
# on the host:
mkdir ~/nwbench && tar -xJf bundle.tar.xz -C ~/nwbench --strip-components=1
cp -r fixtures ~/nwbench/
~/nwbench/bench/setup_host.sh ~/nwbench          # add --metered to skip the large PyPI stacks
~/nwbench/venv-gcc/bin/python ~/nwbench/bench/runner.py --root ~/nwbench --reps 3 --mt <cores>
python tools/fleetbench/report.py --results ~/nwbench/results --out report.md
```

`coordinator.py` holds the fleet layout of the published run in `HOSTS`; edit it for
another fleet.
