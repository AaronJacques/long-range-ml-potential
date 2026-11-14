"""Integrity of the committed model/training/schedule/eval golden fixtures (no legacy env)."""

from __future__ import annotations

import hashlib
import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest

try:
    from golden import fixtures as fx
except ImportError:  # pragma: no cover - depends on pytest import mode
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from golden import fixtures as fx

MANIFEST = fx.load_model_manifest()
VARIANTS = sorted(MANIFEST["variants"])
TRAIN_VARIANTS = sorted(MANIFEST["train"])
EVAL_VARIANTS = sorted(MANIFEST["eval"])


# --------------------------------------------------------------------------- files


@pytest.mark.parametrize("rel", sorted(MANIFEST["files"]))
def test_file_present_with_manifest_sha256(rel: str) -> None:
    path = fx.fixtures_dir() / rel
    assert path.is_file(), rel
    assert hashlib.sha256(path.read_bytes()).hexdigest() == MANIFEST["files"][rel]["sha256"]


def test_manifest_lists_all_model_golden_files() -> None:
    patterns = ("model_*.npz", "train_*.npz", "eval_*.json", "schedules.json")
    on_disk = {f"golden/{p.name}" for pat in patterns for p in fx.golden_dir().glob(pat)}
    assert on_disk == set(MANIFEST["files"])
    assert len(MANIFEST["legacy_commit"]) == 40
    assert VARIANTS == ["local_energy", "lr_energy", "lr_energy_k1", "lr_forces"]


def test_task_fixture_size_below_limit() -> None:
    total = sum((fx.fixtures_dir() / rel).stat().st_size for rel in MANIFEST["files"])
    total += (fx.golden_dir() / "manifest_model.json").stat().st_size
    assert total < 3 * 1024 * 1024


# --------------------------------------------------------------------------- model


@pytest.mark.parametrize("variant", VARIANTS)
def test_weights_match_layer_table(variant: str) -> None:
    g = fx.load_model_golden(variant)
    weights = g["weights_by_class"]
    table = g["layers"]
    assert set(weights) == {"Dense", "LayerNormalization"}
    for cls, layers in weights.items():
        entries = [e for e in table if e["class"] == cls]
        assert [e["index"] for e in entries] == list(range(len(layers)))
        for entry, ws in zip(entries, layers, strict=True):
            assert [list(w.shape) for w in ws] == entry["weight_shapes"], entry["legacy_name"]
            assert all(w.dtype == np.float32 for w in ws)
            m = re.fullmatch(r".*?_(\d+)", entry["legacy_name"])
            assert (int(m.group(1)) if m else 0) == entry["index"]


@pytest.mark.parametrize("variant", VARIANTS)
def test_count_params_is_sum_of_weight_sizes(variant: str) -> None:
    g = fx.load_model_golden(variant)
    total = sum(w.size for layers in g["weights_by_class"].values() for ws in layers for w in ws)
    assert int(g["count_params"]) == total == MANIFEST["variants"][variant]["count_params"]


@pytest.mark.parametrize("variant", VARIANTS)
def test_construction_order_consistent(variant: str) -> None:
    g = fx.load_model_golden(variant)
    order = g["construction_order"]
    assert [e["class"] for e in order if not e["weight_shapes"]] == g["weightless_layers"]
    assert [e for e in order if e["weight_shapes"]] == g["layers"]
    n_inputs = 4 if bool(g["use_long_range"]) else 2
    assert g["weightless_layers"][:n_inputs] == ["InputLayer"] * n_inputs
    assert {"Lambda", "Flatten", "Concatenate", "Dropout", "Add", "Reshape"} <= set(
        g["weightless_layers"]
    )


@pytest.mark.parametrize("variant", VARIANTS)
def test_inputs_outputs_shapes(variant: str) -> None:
    g = fx.load_model_golden(variant)
    n_loc, n_lr = int(g["n_max_local"]), int(g["n_max_long_range"])
    offsets = g["atom_offsets"]
    n_atoms = int(offsets[-1])
    assert len(g["index"]) == len(g["energy"]) == len(offsets) - 1 == 5
    assert g["order"].shape == (n_atoms,)
    assert g["x__local_matrix"].shape == (n_atoms, n_loc, 4)
    assert g["x__local_numbers"].shape == (n_atoms, n_loc, 2)
    if bool(g["use_long_range"]):
        assert g["x__long_range_matrix"].shape == (n_atoms, n_lr, 4)
        assert g["x__long_range_features"].shape == (n_atoms, n_lr, 101)
    else:
        assert "x__long_range_matrix" not in g
    width = 1 if bool(g["predict_only_energy"]) else 4
    assert g["output"].shape == (n_atoms, width)
    assert g["output"].dtype == np.float32
    for m in range(len(offsets) - 1):
        out = g["output"][offsets[m] : offsets[m + 1]].astype(np.float64)
        np.testing.assert_allclose(g["sum_energy"][m], out[:, 0].sum(), rtol=1e-5)
        np.testing.assert_allclose(g["sum_all"][m], out.sum(), rtol=1e-5)
        order = g["order"][offsets[m] : offsets[m + 1]]
        assert np.array_equal(np.sort(order), np.arange(len(order)))


@pytest.mark.parametrize("variant", VARIANTS)
def test_inputs_equal_dataset_golden_test_split(variant: str) -> None:
    g = fx.load_model_golden(variant)
    ds = fx.load_dataset_golden("GB")
    assert int(g["n_max_local"]) == int(ds["n_max_local"])
    assert int(g["n_max_long_range"]) == int(ds["n_max_long_range"])
    assert np.array_equal(g["index"], ds["test__index"][:5])
    np.testing.assert_array_equal(g["energy"], ds["test__energy"][:5])
    n = int(g["atom_offsets"][-1])
    assert np.array_equal(g["atom_offsets"], ds["test__atom_offsets"][:6])
    assert np.array_equal(g["order"], ds["test__order"][:n])
    names = {"local_matrix": "local_matrix", "local_numbers": "local_numbers"}
    if bool(g["use_long_range"]):
        names.update(
            {"long_range_matrix": "long_range_matrix", "long_range_features": "long_range_features"}
        )
    for key, ds_key in names.items():
        assert g[f"x__{key}"].tobytes() == ds[f"test__{ds_key}"][:n].tobytes(), key


def test_m1_hyperparameters() -> None:
    model = MANIFEST["model"]
    data = MANIFEST["dataset"]
    assert model["embedding_dims"] == [16, 32, 32]
    assert (model["M2_local"], model["M2_long"], model["activation"]) == (4, 1, "elu")
    assert model["M1_local"] == data["n_max_local"]
    assert model["M1_long"] == data["n_max_long_range"]
    assert (data["grid_size"], data["max_local_level"]) == (1.0, 2)
    assert data["shuffle_buffer_size"] == 1
    flags = {
        v: (i["use_long_range"], i["predict_only_energy"]) for v, i in MANIFEST["variants"].items()
    }
    assert flags == {
        "lr_energy": (True, True),
        "local_energy": (False, True),
        "lr_forces": (True, False),
        "lr_energy_k1": (True, True),
    }
    for variant, info in MANIFEST["variants"].items():
        if variant != "lr_energy_k1":
            assert info["hyperparameters"]["M1_local"] == data["n_max_local"]
            assert info["hyperparameters"]["M1_long"] == data["n_max_long_range"]


def test_k1_variant_differs_from_n_max() -> None:
    """lr_energy_k1: explicit K1 with M1_local < n_max_local and M1_long > n_max_long_range."""
    g = fx.load_model_golden("lr_energy_k1")
    info = MANIFEST["variants"]["lr_energy_k1"]
    n_loc, n_lr = int(g["n_max_local"]), int(g["n_max_long_range"])
    m1_loc, m1_lr = int(g["M1_local"]), int(g["M1_long"])
    assert m1_loc < n_loc and m1_lr > n_lr
    assert (info["M1_local"], info["M1_long"]) == (m1_loc, m1_lr)
    hp = info["hyperparameters"]
    assert (hp["M1_local"], hp["M1_long"], hp["M2_local"], hp["M2_long"]) == (m1_loc, m1_lr, 4, 1)
    assert hp["embedding_dims"] == MANIFEST["model"]["embedding_dims"]
    dense = g["weights_by_class"]["Dense"]
    widths = [ws[0].shape[1] for ws in dense]
    # embedding heads reshape to (n_max, M1); fitting input = M1_local*M2_local + M1_long*M2_long
    assert n_loc * m1_loc in widths and n_lr * m1_lr in widths
    assert n_loc * n_loc not in widths and n_lr * n_lr not in widths
    assert any(ws[0].shape[0] == m1_loc * 4 + m1_lr * 1 for ws in dense)
    assert "lr_energy_k1" not in MANIFEST["train"] and "lr_energy_k1" not in MANIFEST["eval"]


def test_lr_variants_share_weights_except_head() -> None:
    a = fx.load_model_golden("lr_energy")["weights_by_class"]
    b = fx.load_model_golden("lr_forces")["weights_by_class"]
    assert set(a) == set(b)
    assert b["Dense"][-1][0].shape[1] == 4 and a["Dense"][-1][0].shape[1] == 1
    for cls in a:
        assert len(a[cls]) == len(b[cls])
        layers = list(zip(a[cls], b[cls], strict=True))
        if cls == "Dense":
            layers = layers[:-1]  # the output head differs (1 vs 4 units)
        for k, (wa, wb) in enumerate(layers):
            assert len(wa) == len(wb)
            for j, (x, y) in enumerate(zip(wa, wb, strict=True)):
                assert np.array_equal(x, y), f"{cls}[{k}][{j}]"


# --------------------------------------------------------------------------- training


@pytest.mark.parametrize("variant", TRAIN_VARIANTS)
def test_train_steps_equal_train_split(variant: str) -> None:
    t = fx.load_train_golden(variant)
    n_train = len(fx.load_dataset_golden("GB")["train__index"])
    assert int(t["steps"]) == n_train * int(t["epochs"]) == MANIFEST["dataset"]["splits"]["train"]


@pytest.mark.parametrize("variant", TRAIN_VARIANTS)
def test_train_weights_shapes_and_changed(variant: str) -> None:
    t = fx.load_train_golden(variant)
    start, end = t["start_weights_by_class"], t["weights_by_class"]
    assert set(start) == set(end)
    changed = 0
    for cls in start:
        assert len(start[cls]) == len(end[cls])
        for ws, we in zip(start[cls], end[cls], strict=True):
            assert [w.shape for w in ws] == [w.shape for w in we]
            changed += sum(not np.array_equal(a, b) for a, b in zip(ws, we, strict=True))
    assert changed > 0


@pytest.mark.parametrize("variant", TRAIN_VARIANTS)
def test_train_losses_match_log(variant: str) -> None:
    t = fx.load_train_golden(variant)
    info = t["info"]
    row = dict(zip(info["training_log"]["header"], info["training_log"]["rows"][-1], strict=True))
    assert np.float32(row["Energy Loss"]) == t["energy_loss"]
    assert np.float32(row["Val Energy Loss"]) == t["val_energy_loss"]
    # Energy-only legacy run: final_loss is the validation energy loss of the final pass.
    assert t["final_loss"] == t["val_energy_loss"]
    assert math.isfinite(float(t["energy_loss"])) and float(t["energy_loss"]) > 0
    assert info["hyperparameters"]["initial_learning_rate"] == 1e-3
    assert info["hyperparameters"]["lr_decay_steps"] == 20


# --------------------------------------------------------------------------- schedules


def test_schedules_staircase() -> None:
    s = fx.load_schedules_golden()
    first, second = s["learning_rate"]
    v = dict(zip(first["steps"], first["values"], strict=True))
    assert v[0] == v[1] == v[19] == float(np.float32(1e-3))
    assert v[20] == v[21] == pytest.approx(1e-3 * 0.97, rel=1e-6)
    assert v[60] == pytest.approx(1e-3 * 0.97**3, rel=1e-6)
    assert v[59] == pytest.approx(1e-3 * 0.97**2, rel=1e-6)
    w = dict(zip(second["steps"], second["values"], strict=True))
    assert w[0] == w[87269] == float(np.float32(5e-5))
    assert w[87270] == pytest.approx(5e-5 * 0.97, rel=1e-6)


def test_weighting_schedule() -> None:
    s = fx.load_schedules_golden()
    for case in s["weighting"]:
        for step, value in zip(case["steps"], case["values"], strict=True):
            f = case["decay_rate"] ** (step / case["decay_steps"])
            assert value == pytest.approx(case["limit"] * (1 - f) + case["start"] * f, rel=1e-12)
        if case["start"] == case["limit"]:
            assert all(v == case["start"] for v in case["values"])


# --------------------------------------------------------------------------- evaluation


@pytest.mark.parametrize("variant", EVAL_VARIANTS)
def test_eval_mae(variant: str) -> None:
    e = fx.load_eval_golden(variant)
    assert e["path"] in {"evaluate_model_h5", "in_memory_fallback"}
    assert e["path"] == MANIFEST["eval"][variant]["path"]
    per = np.asarray(e["per_molecule_abs_error"], dtype=np.float32)
    assert e["test_size"] == len(per) == MANIFEST["dataset"]["splits"]["test"]
    assert np.all(per >= 0)
    assert float(np.mean(per)) == e["mae"] == MANIFEST["eval"][variant]["mae"]
    if e["mae_printed"] is not None:
        assert f"{e['mae']:.5f}" == e["mae_printed"]
