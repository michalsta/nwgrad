"""The `python -m nwgrad` command line.

CLAUDE.md points users at `python -m nwgrad --include` to locate the headers when
consuming nwgrad as a header-only C++ library, but nothing exercised it, so the
whole module sat at 0% coverage.  The contract worth pinning is not that it
prints *a* string — it is that the printed path is actually usable as an include
directory, i.e. the headers are really there.
"""

import shutil
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import pytest


def run(*args):
    result = subprocess.run([sys.executable, "-m", "nwgrad", *args],
                            capture_output=True, text=True)
    return result


def test_include_prints_a_directory_that_holds_the_headers():
    result = run("--include")
    assert result.returncode == 0

    include_dir = Path(result.stdout.strip())
    assert include_dir.is_dir()

    # The path is meant to be passed to a compiler as -I, with sources doing
    # #include <nwgrad/aligner.hpp>, so the headers must sit one level down.
    for header in ("aligner.hpp", "align_params.hpp", "subst_matrix.hpp",
                   "batch.hpp", "seq_pair.hpp", "seq_pair_batch.hpp"):
        assert (include_dir / "nwgrad" / header).is_file(), header


def test_include_short_flag_matches_long_flag():
    assert run("-i").stdout == run("--include").stdout


@pytest.mark.parametrize("flag", ["--version", "-v"])
def test_version_matches_the_installed_package(flag):
    result = run(flag)
    assert result.returncode == 0
    assert result.stdout.strip() == version("nwgrad")


def test_no_arguments_prints_help_and_succeeds():
    result = run()
    assert result.returncode == 0
    assert "--include" in result.stdout


def test_unknown_flag_fails():
    result = run("--nope")
    assert result.returncode != 0


def test_headers_are_importable_from_the_advertised_path():
    """Compile a translation unit against the advertised include path.

    This is the real promise of `--include`: that the headers it points at are
    self-contained and actually compile as a header-only library.
    """
    cxx = shutil.which("c++") or shutil.which("g++")
    if cxx is None:
        pytest.skip("no C++ compiler available")

    include_dir = run("--include").stdout.strip()
    source = "#include <nwgrad/seq_pair_batch.hpp>\nint main() { return 0; }\n"

    result = subprocess.run(
        [cxx, "-std=c++20", "-I", include_dir, "-fsyntax-only", "-x", "c++", "-"],
        input=source, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
