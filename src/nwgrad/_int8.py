"""The int8 scalar type: int16 with parameters that fit a signed byte.

Integer score-only work runs the int8 score kernel first (saturating int8 lanes, twice
int16's; ``score8_kernel_impl.inl``) and re-runs it at int16 when the kernel cannot
prove its result fits (a cascade 8 -> 16 -> 32).  The int16
and wider types take the same tier automatically (``narrowing``), so the int8 names add
one thing: the parameter range is checked up front.  Every other output (paths, guides,
hard gradients) runs at int16, whose results are the same integers -- there is no int8
traceback.  Like every integer type, int8 has no soft path.

The tier needs gap parameters >= 0 with open + extend <= 127 per side, and for Global /
Glocal min(len_a, len_b) * (the largest substitution score) <= 126; outside that, or when
a Local score reaches 127, the score comes from the int16 kernel (the same number).
"""
import numpy as np

from . import nwgrad_ext as _ext

_LIMIT = 127


def _check(params):
    """ValueError unless every parameter of `params` is an integer within +-127."""
    for what, arr in (("substitution score", params.matrices), ("gap parameter", params.gaps)):
        a = np.asarray(arr, dtype=np.float64).ravel()
        bad = ~((a == np.round(a)) & (np.abs(a) <= _LIMIT))
        if bad.any():
            x = float(a[np.argmax(bad)])
            raise ValueError(
                f"nwgrad: {what} {x} is not an integer within +-{_LIMIT} (the int8 scalar "
                f"type takes integral parameters that fit a signed byte; quantize them "
                f"first, or use the int16 names)")
    return params


class SeqPairInt8(_ext.SeqPairInt16):
    """SeqPairInt16 with int8 parameters (checked); score-only runs the int8 tier.
    Paths and gradients are computed at int16 (the same integers)."""

    def __init__(self, seq_a, seq_b, params, *args, **kwargs):
        super().__init__(seq_a, seq_b, _check(params), *args, **kwargs)

    def set_params(self, params):
        return super().set_params(_check(params))


class SeqPairBatchInt8(_ext.SeqPairBatchInt16):
    """SeqPairBatchInt16 with int8 parameters (checked); score-only runs the int8 tier.
    Paths and gradients are computed at int16 (the same integers)."""

    def __init__(self, *args, **kwargs):
        if "params" in kwargs:
            _check(kwargs["params"])
        elif len(args) >= 3 and not isinstance(args[0], str):
            _check(args[2])
        super().__init__(*args, **kwargs)

    def set_params(self, params):
        return super().set_params(_check(params))

    def align(self, *args, **kwargs):
        params = kwargs.get("params", args[2] if len(args) >= 3 else None)
        if params is not None:
            _check(params)
        return super().align(*args, **kwargs)


def _wrap(name):
    fn = getattr(_ext, name + "_int16")

    def f(seq_a, seq_b, params, *args, **kwargs):
        return fn(seq_a, seq_b, _check(params), *args, **kwargs)

    f.__name__ = f.__qualname__ = name + "_int8"
    f.__doc__ = (fn.__doc__ or "") + "\n\nint8: the parameters must be integers within +-127 " \
        "(ValueError otherwise); the score runs the int8 tier where it proves its result."
    return f


_NAMES = ("nw_score", "sw_score", "nw_score_affine", "sw_score_affine",
          "glocal_score", "glocal_score_affine",
          "nw_grad", "sw_grad", "nw_affine_grad", "sw_affine_grad",
          "glocal_grad", "glocal_affine_grad")
globals().update({n + "_int8": _wrap(n) for n in _NAMES})
__all__ = ["SeqPairInt8", "SeqPairBatchInt8"] + [n + "_int8" for n in _NAMES]
