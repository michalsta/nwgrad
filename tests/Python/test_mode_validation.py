"""Invalid mode names must fail before constructing or appending pairs."""

import numpy as np
import pytest

import nwgrad


@pytest.mark.parametrize("suffix", ["", "Double"])
@pytest.mark.parametrize("surface", ["SeqPair", "SeqPairBatch"])
@pytest.mark.parametrize(
    "field, typo, choices",
    [
        ("gap_model", "affien", ("linear", "affine")),
        ("mode", "locla", ("global", "local")),
        ("grad_mode", "hrd", ("hard", "soft", "none")),
    ],
)
@pytest.mark.parametrize("invalid_kind", ["typo", "empty", "uppercase"])
def test_invalid_modes_raise(suffix, surface, field, typo, choices, invalid_kind):
    params = nwgrad.AlignParams(np.eye(4), alphabet="ACGT")
    value = {"typo": typo, "empty": "", "uppercase": choices[0].upper()}[invalid_kind]
    kwargs = {field: value}

    with pytest.raises(ValueError) as exc:
        if surface == "SeqPair":
            getattr(nwgrad, surface + suffix)("ACGT", "ACGT", params, **kwargs)
        elif surface == "SeqPairBatch":
            getattr(nwgrad, surface + suffix)(n_threads=1, **kwargs)

    message = str(exc.value)
    assert f'unknown {field} "{value}"' in message
    assert "expected" in message
    for choice in choices:
        assert f'"{choice}"' in message
