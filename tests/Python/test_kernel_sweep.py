"""The sweep must reach batch methods and preserve explicit kernel choices."""

from types import SimpleNamespace

import conftest


def test_batch_methods_and_positional_overrides(monkeypatch):
    class Batch:
        """A stand-in with the signatures nanobind exposes through docstrings."""

        def add_many(self, seqs_a, seqs_b, params, kernel="auto"):
            """add_many(self: Batch, seqs_a: list[str], seqs_b: list[str], params: object, kernel: str = 'auto')"""
            return kernel

        def align(self, seqs_a, seqs_b, params, kernel="auto"):
            """align(self: Batch, seqs_a: list[str], seqs_b: list[str], params: object, kernel: str = 'auto')"""
            return kernel

    module = SimpleNamespace(Batch=Batch)
    monkeypatch.setattr(conftest, "_current_backend", "scalar_fallback")
    patched = conftest._patch_entry_points(module)
    assert set(patched) == {"Batch.add_many", "Batch.align"}
    batch = module.Batch()
    for method in (batch.add_many, batch.align):
        assert method([], [], None) == "scalar_fallback"
        assert method([], [], None, kernel="auto") == "auto"
        assert method([], [], None, "auto") == "auto"


def test_constructor_and_function_positional_overrides(monkeypatch):
    class Pair:
        def __init__(self, a, b, params, kernel="auto"):
            """__init__(self: Pair, a: str, b: str, params: object, kernel: str = 'auto')"""
            self.kernel = kernel

    def score(a, b, params, kernel="auto"):
        """score(a: str, b: str, params: object, kernel: str = 'auto')"""
        return kernel

    module = SimpleNamespace(Pair=Pair, score=score)
    monkeypatch.setattr(conftest, "_current_backend", "scalar_fallback")
    conftest._patch_entry_points(module)
    assert module.Pair("A", "A", None).kernel == "scalar_fallback"
    assert module.Pair("A", "A", None, "auto").kernel == "auto"
    assert module.score("A", "A", None) == "scalar_fallback"
    assert module.score("A", "A", None, "auto") == "auto"
