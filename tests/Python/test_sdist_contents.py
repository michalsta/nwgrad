"""What goes into the source distribution — issue 7.

A local `python -m build --sdist` collected every untracked file in the working tree:
a 106 MB proteome FASTA, whole build directories with test binaries and object files,
.claude/settings.local.json, scratch scripts.  A clean CI checkout hides this, which
is why it needs a test that does not run from a clean checkout.

The fixture copies the git-TRACKED files into a temporary tree, plants representative
untracked junk beside them, builds an sdist from that tree once, and the tests read the
archive.  Two directions, because a fix can fail either way: junk must stay out, and
everything the extension needs to compile must stay in (an over-eager exclude list
breaks the build from sdist that issue 1's fix made CI test).

Skipped outside a git checkout, or without `build` and scikit-build-core installed.
"""

import importlib.util
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]

# Untracked files of the kinds actually found in a working tree.  Paths relative to
# the project root; directories end in "/".
JUNK = [
    "GCF_000000000.1_scratch_protein.faa",
    "bld-gcc/nwgrad_tests",
    "bld-gcc/CMakeFiles/nwgrad_ext.dir/py_exports.cpp.o",
    "build-asan/CMakeCache.txt",
    ".claude/settings.local.json",
    "HANDOFF.md",
    "scratch_test.py",
    "proteome_test.py_bk",
    "tools/SCRATCH_BRIEF.md",
]


def _tracked_files():
    try:
        out = subprocess.run(["git", "-C", str(REPO), "ls-files", "-z"],
                             capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return [f for f in out.decode().split("\0") if f and (REPO / f).is_file()]


@pytest.fixture(scope="module")
def sdist(tmp_path_factory):
    for mod in ("build", "scikit_build_core"):
        if importlib.util.find_spec(mod) is None:
            pytest.skip(f"{mod} not installed")
    tracked = _tracked_files()
    if not tracked:
        pytest.skip("not a git checkout")

    tree = tmp_path_factory.mktemp("sdist-src") / "nwgrad"
    for f in tracked:
        dst = tree / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / f, dst)
    for j in JUNK:
        p = tree / j
        if p.exists():  # the junk name collides with a tracked file: not junk here
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("untracked scratch\n")

    out = tree.parent / "dist"
    proc = subprocess.run([sys.executable, "-m", "build", "--sdist", "--no-isolation",
                           "--outdir", str(out), str(tree)],
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    (archive,) = out.glob("*.tar.gz")
    with tarfile.open(archive) as tf:
        # Strip the top-level "nwgrad-<version>/" directory.
        names = {m.name.split("/", 1)[1] for m in tf.getmembers()
                 if m.isfile() and "/" in m.name}
    return names, set(tracked)


def test_sdist_contains_only_tracked_files(sdist):
    names, tracked = sdist
    generated = {"PKG-INFO"}
    stray = sorted(names - tracked - generated)
    assert not stray, f"{len(stray)} untracked file(s) in the sdist: {stray}"


def test_sdist_keeps_everything_the_build_needs(sdist):
    """The guard against fixing issue 7 too hard.  Every tracked file the extension
    build or the header-only C++ API consumes must be in the archive."""
    names, tracked = sdist
    needed = {f for f in tracked
              if f in ("CMakeLists.txt", "pyproject.toml", "README.md")
              or (f.startswith("src/") and "__pycache__" not in f)}
    missing = sorted(needed - names)
    assert not missing, f"sdist is missing {missing}"
    assert any(f.endswith(".inl") for f in names), "no .inl kernel sources in sdist"
