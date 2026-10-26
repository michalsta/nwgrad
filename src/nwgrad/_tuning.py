"""Fill-gate overrides for tuning sweeps.  UNSTABLE: not part of the API.

nwgrad picks a fill per pair through thresholds ("gates") measured on one host per ISA
level, compiled in as defaults (``simd_levels.hpp``, ``Aligner::kLinearStripedMinN``).
This module overrides them at runtime, process-wide, so a sweep can measure where each
crossover lies on another CPU without rebuilding::

    from nwgrad import _tuning
    with _tuning.override(inter_score_gather_cells=None):   # never send score-only
        batch.score_and_grad()                              # pairs to their own fill

Values: an ``int`` is the threshold (cells, bytes, or for ``linear_striped_min_n`` the
B length from which the striped linear kernel runs); ``None`` turns the gate off (a cap
never rejects; ``linear_striped_min_n`` becomes 0, the striped kernel at any length);
``"default"`` removes the override.  Gates only choose between fills that are
bit-identical, so results never change, only speed and memory.  A batch regroups its
cached inter-pair plan on the next call after any change.  Not thread-safe to change
while work is in flight.
"""
from contextlib import contextmanager

from . import nwgrad_ext as _ext

GATES = tuple(_ext._gate_names())
_OFF = _ext._gate_off_value()
_MIN_N = "linear_striped_min_n"


def _check(name):
    if name not in GATES:
        raise ValueError(f"unknown fill gate {name!r}; one of {', '.join(GATES)}")


def defaults():
    """The compiled defaults: ``{name: int or None}`` (None = no cap).
    ``linear_striped_min_n`` is per mode, type and level, so it reads ``"per type"``."""
    d = {k: (v if v > 0 else None) for k, v in _ext._gate_defaults()}
    d[_MIN_N] = "per type"
    return d


def get_gate(name):
    """The override in force: an int, None (gate off) or ``"default"`` (no override)."""
    _check(name)
    v = _ext._gate_get(name)
    return "default" if v < 0 else None if v == _OFF else v


def gates():
    """``{name: get_gate(name)}`` for every gate."""
    return {g: get_gate(g) for g in GATES}


def set_gate(name, value):
    """Override one gate: an int >= 0, None (off) or ``"default"`` (remove)."""
    _check(name)
    if isinstance(value, str):
        if value != "default":
            raise ValueError(f"gate value must be an int, None or 'default', not {value!r}")
        raw = -1
    elif value is None:
        raw = 0 if name == _MIN_N else _OFF
    else:
        raw = int(value)
        if raw < 0 or raw != value:
            raise ValueError(f"gate value must be a non-negative integer, not {value!r}")
    _ext._gate_set(name, raw)


def reset_gates():
    """Remove every override."""
    for g in GATES:
        _ext._gate_set(g, -1)


@contextmanager
def override(**values):
    """Set gates for the duration of a ``with`` block, restoring the previous ones."""
    for g in values:
        _check(g)
    saved = {g: _ext._gate_get(g) for g in values}
    try:
        for g, v in values.items():
            set_gate(g, v)
        yield
    finally:
        for g, raw in saved.items():
            _ext._gate_set(g, raw)
