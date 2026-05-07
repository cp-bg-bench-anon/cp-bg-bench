from __future__ import annotations

import importlib
import pkgutil

import cp_bg_bench_model


def test_all_public_imports():
    """Every module under cp_bg_bench_model imports without side-effect errors."""
    pkg = cp_bg_bench_model
    failures: list[tuple[str, str]] = []
    for mod in pkgutil.walk_packages(pkg.__path__, prefix=pkg.__name__ + "."):
        try:
            importlib.import_module(mod.name)
        except Exception as e:  # noqa: BLE001
            failures.append((mod.name, repr(e)))
    assert not failures, f"Import failures: {failures}"
