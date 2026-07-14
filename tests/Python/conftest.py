"""Force the whole suite through the simd Viterbi kernel.

    NWGRAD_FORCE_KERNEL=simd pytest tests/Python/

The scalar and simd kernels write bit-identical DP tables — that is the central
claim of aligner_simd.hpp, and it is what buys the exact-float-equality tracebacks
the right to keep working untouched.  If the claim holds, then re-running the
*entire* existing suite against the simd kernel must pass without a single
tolerance being loosened, because no test can observe any difference.

That makes this the strongest verification available: it costs one environment
variable and turns every gradient, alignment and biopython-comparison test in the
suite into a simd test.  A failure here is a real bug, not a tolerance to relax.
"""

import os

import pytest

_KERNEL = os.environ.get("NWGRAD_FORCE_KERNEL", "").strip()


def pytest_report_header(config):
    if _KERNEL:
        return f"nwgrad: FORCING kernel={_KERNEL!r} for every aligner call"
    return "nwgrad: kernel=scalar (default); set NWGRAD_FORCE_KERNEL=simd to force simd"


def _inject_kernel(fn, kernel):
    """Wrap a callable so it passes kernel=... unless the caller already did."""

    def wrapper(*args, **kwargs):
        kwargs.setdefault("kernel", kernel)
        return fn(*args, **kwargs)

    wrapper.__name__ = getattr(fn, "__name__", "wrapped")
    wrapper.__doc__ = getattr(fn, "__doc__", None)
    return wrapper


@pytest.fixture(scope="session", autouse=True)
def force_kernel():
    if not _KERNEL:
        return

    import nwgrad

    if _KERNEL not in ("scalar", "simd"):
        raise ValueError(f"NWGRAD_FORCE_KERNEL must be 'scalar' or 'simd', got {_KERNEL!r}")

    # Every entry point that accepts kernel=: the 12 convenience functions, plus the
    # two classes.  Discovered from the signature rather than from a hand-kept list, so
    # a new entry point cannot quietly escape the sweep.
    #
    # inspect.signature() is no use here: nanobind callables report (*args, **kwargs).
    # The real signature is the first line of __doc__, and for a bound class it is the
    # first line of __init__.__doc__.
    def signature_line(obj):
        doc = obj.__init__.__doc__ if isinstance(obj, type) else obj.__doc__
        return (doc or "").splitlines()[0] if doc else ""

    patched = []
    for name in dir(nwgrad):
        if name.startswith("_"):
            continue
        obj = getattr(nwgrad, name)
        if callable(obj) and "kernel:" in signature_line(obj):
            setattr(nwgrad, name, _inject_kernel(obj, _KERNEL))
            patched.append(name)

    assert patched, "NWGRAD_FORCE_KERNEL set, but no kernel-taking entry point was patched"
    print(f"\nnwgrad: kernel={_KERNEL} forced on {len(patched)} entry points: "
          f"{', '.join(sorted(patched))}")
