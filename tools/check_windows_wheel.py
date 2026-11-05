"""Check a MinGW-built Windows wheel's DLL imports, then hand it on unchanged.

Used as cibuildwheel's Windows repair command:

    python tools/check_windows_wheel.py {wheel} {dest_dir}

or, on an already-built module (ci.yml's Windows job, on the editable install):

    python tools/check_windows_wheel.py --pyd path/to/nwgrad_ext.pyd

The extension is linked with -static (CMakeLists.txt, MinGW block), so its .pyd must
import only Windows system DLLs, the UCRT (api-ms-win-crt-*) and python3XY.dll.  This
fails the build if any extension module imports:

  * a MinGW runtime DLL (libstdc++-6, libgcc_s_seh-1, libwinpthread-1, libgomp-1, ...):
    the static link did not take, and the wheel would load only where some other
    MinGW-built package happened to put those DLLs on PATH;
  * msvcrt.dll: the toolchain was an MSVCRT MinGW, not UCRT64.  CPython links the UCRT,
    and two C runtimes in one process do not share a heap, FILE* or errno.

Imports are read with objdump -p (MinGW binutils, on PATH next to gcc).
"""

import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

FORBIDDEN = [
    (re.compile(r"^lib.*\.dll$", re.I), "MinGW runtime DLL (the -static link did not take)"),
    (re.compile(r"^msvcrt\.dll$", re.I), "MSVCRT C runtime (build with UCRT64, not an MSVCRT MinGW)"),
]


def dll_imports(objdump, path):
    out = subprocess.run([objdump, "-p", str(path)], check=True,
                         capture_output=True, text=True).stdout
    return sorted({m.group(1) for m in re.finditer(r"DLL Name:\s*(\S+)", out)})


def check(objdump, label, path):
    """Print the module's imports; return the forbidden ones, explained."""
    imports = dll_imports(objdump, path)
    print(f"{label}: imports {', '.join(imports)}")
    return [f"{label} imports {dll}: {why}"
            for dll in imports for pattern, why in FORBIDDEN if pattern.match(dll)]


def main():
    args = sys.argv[1:]
    if len(args) != 2:
        sys.exit(f"usage: {sys.argv[0]} WHEEL DEST_DIR | --pyd MODULE")
    objdump = shutil.which("objdump")
    if objdump is None:
        sys.exit("check_windows_wheel: objdump not found on PATH (MinGW binutils)")

    if args[0] == "--pyd":
        mod = pathlib.Path(args[1])
        bad = check(objdump, mod.name, mod)
        if bad:
            sys.exit("check_windows_wheel: FAILED\n  " + "\n  ".join(bad))
        print("check_windows_wheel: OK")
        return

    wheel, dest = pathlib.Path(args[0]), pathlib.Path(args[1])
    bad = []
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(wheel) as zf:
            mods = [n for n in zf.namelist() if n.lower().endswith((".pyd", ".dll"))]
            if not mods:
                sys.exit(f"check_windows_wheel: no extension module in {wheel.name}")
            zf.extractall(tmp, members=mods)
        for name in mods:
            bad += check(objdump, name, pathlib.Path(tmp) / name)

    if bad:
        sys.exit("check_windows_wheel: FAILED\n  " + "\n  ".join(bad))
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(wheel, dest / wheel.name)
    print(f"check_windows_wheel: OK, {wheel.name} -> {dest}")


if __name__ == "__main__":
    main()
