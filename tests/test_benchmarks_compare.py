"""Labels of the training rows in ``benchmarks/compare.py`` (C4-K4)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

COMPARE = Path(__file__).resolve().parents[1] / "benchmarks" / "compare.py"


@pytest.fixture(scope="module")
def compare() -> ModuleType:
    spec = importlib.util.spec_from_file_location("bench_compare", COMPARE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _new(train_defaults: dict[str, Any] | None) -> dict[str, Any]:
    meta = {} if train_defaults is None else {"train_defaults": train_defaults}
    return {"metadata": meta}


def test_old_result_files_have_k1_without_xla(compare: ModuleType) -> None:
    assert compare.train_defaults(_new(None)) == {"default": (1, False)}


def test_recorded_defaults_label_the_rows(compare: ModuleType) -> None:
    preset = {"steps_per_execution": 8, "jit_compile": True}
    recorded = {
        "TrainConfig": {"steps_per_execution": 8, "jit_compile": False},
        "presets": {"aspirin": preset, "stachyose": preset},
    }
    assert compare.train_defaults(_new(recorded)) == {
        "TrainConfig default": (8, False),
        "thesis presets": (8, True),
    }
    recorded["presets"]["stachyose"] = {"steps_per_execution": 1, "jit_compile": False}
    labels = compare.train_defaults(_new(recorded))
    assert labels["preset aspirin"] == (8, True)
    assert labels["preset stachyose"] == (1, False)
