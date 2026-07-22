"""Force the whole suite through a chosen Viterbi backend — sweeping every simd level.

    NWGRAD_FORCE_KERNEL=simd pytest tests/Python/            # sweep every simd level
    NWGRAD_FORCE_KERNEL=avx2 pytest tests/Python/            # force one backend
    NWGRAD_FORCE_KERNEL=scalar_fallback pytest tests/Python/ # force the scalar path

Every simd level writes DP tables bit-identical to the scalar one — the central claim of
aligner_simd.hpp, and what buys the exact-float-equality tracebacks the right to keep
working untouched.  If that holds, re-running the *entire* suite forced onto a simd
backend must pass with no tolerance loosened, because no test can observe a difference.
That is the strongest verification available: one environment variable turns every
gradient, alignment and biopython test into a simd test.

**Every level.**  NWGRAD_FORCE_KERNEL=simd sweeps all simd levels the CPU can run
(sse2/avx2/avx512 on x86, neon on ARM) — the whole suite runs once per level, so the
baseline (sse2) kernel is tested from Python too, not just the strongest.  It mirrors the
C++ ctest cpp_tests_isa_* loop.  The backend is passed per-call via kernel= (the unified
vocabulary in simd_levels.hpp), so no global state is mutated; scalar_fallback is left out
of the =simd sweep since that is the default (unforced) run.
"""

import os

import pytest

_KERNEL = os.environ.get("NWGRAD_FORCE_KERNEL", "").strip()
_current_backend = None  # the backend string injected into kernel=, updated per sweep param


def _sweep_backends():
    """Backend value(s) to force, or None for a normal (unforced) run.

    "simd"       -> every simd level the CPU runs (scalar_fallback excluded — it is the
                    default run); the suite runs once per level.
    <backend>    -> force exactly that backend (scalar_fallback / auto / sse2 / avx2 / ...).
    """
    if not _KERNEL:
        return None
    import nwgrad
    avail = nwgrad.available_isa_levels()  # ["scalar_fallback", <simd levels weakest-first>...]
    if _KERNEL == "simd":
        levels = [b for b in avail if b != "scalar_fallback"]
        return levels or None
    return [_KERNEL]


_SWEEP = _sweep_backends()
# params=None (the fixture default) leaves the fixture un-parametrized: the unforced suite
# stays a single run with no backend id in its test names.  A list turns it into one run
# per swept backend.  Computed at import so pytest sees it before collection.
_PARAMS = _SWEEP
_IDS = [f"backend-{b}" for b in _SWEEP] if _SWEEP else None
_patched = False


def pytest_report_header(config):
    if not _KERNEL:
        return ("nwgrad: default backend (auto = best simd); "
                "set NWGRAD_FORCE_KERNEL=simd to sweep every simd level")
    if _SWEEP and len(_SWEEP) > 1:
        return (f"nwgrad: FORCING kernel over backends {_SWEEP} "
                f"— the whole suite runs once per backend")
    return f"nwgrad: FORCING kernel={(_SWEEP or [_KERNEL])[0]!r} for every aligner call"


def _inject_kernel(fn):
    """Wrap a callable so it passes kernel=<current backend> unless the caller set one."""

    def wrapper(*args, **kwargs):
        kwargs.setdefault("kernel", _current_backend)
        return fn(*args, **kwargs)

    wrapper.__name__ = getattr(fn, "__name__", "wrapped")
    wrapper.__doc__ = getattr(fn, "__doc__", None)
    return wrapper


def _patch_entry_points(nwgrad):
    # Every entry point that accepts kernel=: the 12 convenience functions, plus the
    # classes.  Discovered from the signature rather than a hand-kept list, so a new entry
    # point cannot quietly escape the sweep.
    #
    # inspect.signature() is no use here: nanobind callables report (*args, **kwargs).  The
    # real signature is the first line of __doc__, and for a bound class it is __init__'s.
    def signature_line(obj):
        doc = obj.__init__.__doc__ if isinstance(obj, type) else obj.__doc__
        return (doc or "").splitlines()[0] if doc else ""

    patched = []
    for name in dir(nwgrad):
        if name.startswith("_"):
            continue
        obj = getattr(nwgrad, name)
        if callable(obj) and "kernel:" in signature_line(obj):
            setattr(nwgrad, name, _inject_kernel(obj))
            patched.append(name)

    assert patched, "NWGRAD_FORCE_KERNEL set, but no kernel-taking entry point was patched"
    return patched


@pytest.fixture(scope="session", autouse=True, params=_PARAMS, ids=_IDS)
def force_kernel(request):
    if not _KERNEL:
        return

    import nwgrad

    # Patch the kernel= entry points once; the wrappers read _current_backend at call time,
    # so the same wrappers serve every backend in the sweep.
    global _patched, _current_backend
    if not _patched:
        patched = _patch_entry_points(nwgrad)
        _patched = True
        print(f"\nnwgrad: kernel= forced on {len(patched)} entry points: "
              f"{', '.join(sorted(patched))}")

    _current_backend = getattr(request, "param", _KERNEL)
    print(f"nwgrad: forcing backend {_current_backend!r}")

