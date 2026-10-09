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
vocabulary in simd_levels.hpp), including batch add_many() and align(). Explicit
kernel arguments are preserved. The unforced run uses auto (best available SIMD);
run =scalar_fallback separately to exercise the scalar backend.
"""

import os
import re

import pytest


def running_under_asan():
    """Best-effort detection of AddressSanitizer in *this* process.

    Checked by reading our own memory map rather than an env var (ASAN_OPTIONS /
    LD_PRELOAD are set by the CI job's shell, not by the interpreter, so a test
    importing this module has no other reliable signal). Used to skip tests that
    infer heap behaviour from RSS: ASan's redzones and its quarantine of freed
    blocks (held, not released, to catch use-after-free) both inflate RSS
    regardless of whether the code under test leaked anything, so a threshold
    tuned for a normal allocator fires on a perfectly healthy build.
    """
    try:
        with open("/proc/self/maps") as f:
            return "libasan" in f.read()
    except OSError:
        return False


def run_isolated(code, timeout=120):
    """Run `code` in a fresh interpreter and return the CompletedProcess.

    For tests whose failure mode is a use-after-free or an abort: in-process, the
    bug would take pytest down with it (or corrupt its heap and fail something
    unrelated later).  The child inherits the environment, so under the CI's
    ASan job it is instrumented too and a finding surfaces as a non-zero exit.
    NWGRAD_FORCE_KERNEL is NOT applied in the child — it is wired through this
    conftest, which the child never loads — so these tests stay on the default
    backend; they test lifetimes and threading, not kernels.
    """
    import subprocess
    import sys
    import textwrap
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)],
                          capture_output=True, text=True, timeout=timeout)


def describe(proc):
    """Failure message for a run_isolated() result: exit code plus both streams."""
    return (f"child exited {proc.returncode}\n--- stdout ---\n{proc.stdout}"
            f"\n--- stderr ---\n{proc.stderr}")


_KERNEL = os.environ.get("NWGRAD_FORCE_KERNEL", "").strip()
_current_backend = None  # the backend string injected into kernel=, updated per sweep param


def _sweep_backends():
    """Backend value(s) to force, or None for a normal (unforced) run.

    "simd"       -> every simd level the CPU runs (scalar_fallback excluded);
                    the suite runs once per level.
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


def _inject_kernel(fn, kernel_position):
    """Wrap a callable so it passes kernel=<current backend> unless the caller set one."""

    def wrapper(*args, **kwargs):
        if len(args) <= kernel_position:
            kwargs.setdefault("kernel", _current_backend)
        return fn(*args, **kwargs)

    wrapper.__name__ = getattr(fn, "__name__", "wrapped")
    wrapper.__doc__ = getattr(fn, "__doc__", None)
    return wrapper


def _patch_entry_points(nwgrad):
    # Discover functions, constructors and class methods that accept kernel=.
    #
    # inspect.signature() is no use here: nanobind callables report (*args, **kwargs).  The
    # real signature is the first line of __doc__, and for a bound class it is __init__'s.
    def signature_line(obj):
        doc = obj.__init__.__doc__ if isinstance(obj, type) else obj.__doc__
        return (doc or "").splitlines()[0] if doc else ""

    def kernel_position(signature, constructor=False):
        # nanobind's first docstring line includes self for __init__, but callers
        # of the class pass no self. Method wrappers receive it in args normally.
        params = re.findall(r"(?:^|,\s*)(\w+):", signature.split("(", 1)[1])
        return params.index("kernel") - int(constructor)

    patched = []
    for name in dir(nwgrad):
        if name.startswith("_"):
            continue
        obj = getattr(nwgrad, name)
        if isinstance(obj, type):
            for method_name in dir(obj):
                method = getattr(obj, method_name)
                signature = (getattr(method, "__doc__", None) or "").splitlines()
                if method_name != "__init__" and signature and "kernel:" in signature[0]:
                    setattr(obj, method_name, _inject_kernel(
                        method, kernel_position(signature[0])))
                    patched.append(f"{name}.{method_name}")
        signature = signature_line(obj)
        if callable(obj) and "kernel:" in signature:
            setattr(nwgrad, name, _inject_kernel(
                obj, kernel_position(signature, constructor=isinstance(obj, type))))
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



class StreamAligner:
    """BatchAligner's call shape over SeqPairBatch.align() (BatchAligner was removed in 0.6).

    For the tests of the streaming semantics — score a list of pairs under fixed params,
    return a BatchResult, keep nothing per pair — which are unchanged; only the entry point
    moved.  kernel=None follows NWGRAD_FORCE_KERNEL's current backend, so these tests stay
    in the forced sweep as BatchAligner's did (its __init__ took kernel=, and was patched).
    """
    _batch_cls = None

    def __init__(self, params, band=0, gap_model="affine", mode="global", grad_mode="hard",
                 n_threads=1, kernel=None):
        import nwgrad
        cls = self._batch_cls or nwgrad.SeqPairBatch
        self.params, self.band = params, band
        self.kernel = kernel if kernel is not None else (_current_backend or "auto")
        self._b = cls(n_threads, gap_model=gap_model, mode=mode, grad_mode=grad_mode)

    def align(self, seqs_a, seqs_b, aligned_a=(), aligned_b=()):
        return self._b.align(list(seqs_a), list(seqs_b), self.params, self.band,
                             list(aligned_a), list(aligned_b), self.kernel)

    @property
    def fill(self):
        return self._b.fill

    @fill.setter
    def fill(self, v):
        if v not in ("interpair", "striped"):   # BatchAligner never had "rowwise"
            raise ValueError(f'unknown fill "{v}" (expected "interpair" or "striped")')
        self._b.fill = v

    soft_impl = property(lambda s: s._b.soft_impl, lambda s, v: setattr(s._b, "soft_impl", v))
    soft_temperature = property(lambda s: s._b.soft_temperature,
                                lambda s, v: setattr(s._b, "soft_temperature", v))
    n_threads = property(lambda s: s._b.n_threads)


class StreamAlignerDouble(StreamAligner):
    def __init__(self, *args, **kwargs):
        import nwgrad
        self._batch_cls = nwgrad.SeqPairBatchDouble
        super().__init__(*args, **kwargs)
