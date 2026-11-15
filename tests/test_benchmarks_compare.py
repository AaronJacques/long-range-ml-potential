"""Labels and metadata rows of the generated page in ``benchmarks/compare.py``."""

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


def test_env_table_shows_measured_tree(compare: ModuleType) -> None:
    """A result measured on a dirty tree names the measured commit (``measured_tree``)."""
    meta = {"git_sha": "aaa", "git_dirty_src_or_benchmarks": True, "measured_tree": "bbb"}
    table = compare.env_table({"metadata": meta}, {"metadata": {"git_sha": "aaa"}})
    assert "| measured code state (commit) | `bbb` | `—` |" in table


def test_reference_results_name_the_measured_commit() -> None:
    import json

    for path in sorted((COMPARE.parent / "results").glob("reference-*.json")):
        meta = json.loads(path.read_text(encoding="utf-8"))["metadata"]
        if meta["git_dirty_src_or_benchmarks"]:
            assert len(meta.get("measured_tree", "")) == 40, path.name
