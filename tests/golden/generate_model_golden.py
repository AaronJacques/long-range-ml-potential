"""Generate the model, training, schedule and evaluation golden fixtures from the legacy code.

Run in a legacy env (Python 3.12, TF 2.21, tf-keras, pandas:
``pip install -e ".[legacy]"``, e.g. ``.venv``)::

    TF_USE_LEGACY_KERAS=1 PYTHONDONTWRITEBYTECODE=1 \\
        python tests/golden/generate_model_golden.py [--check]

Model golden M-1 on dataset G-B (grid 1.0, l_max 2, all 60 frames of
``synthetic_small.npz``, split 0.1/0.1/42):

``model_<variant>.npz``
    Initial weights (``set_random_seed(1234)`` after ``clear_session()``), the first
    five molecules of the test split and ``model(x, training=False)`` per molecule.
    ``model_lr_energy_k1.npz`` is the forward-only variant with an explicit
    K1 != N_max (``M1_local`` < n_max_local, ``M1_long`` > n_max_long_range).
``train_lr_energy.npz``
    Weights after one epoch of the unchanged ``Training.start_training`` (starting
    from exactly the weights of ``model_lr_energy.npz``), epoch losses from
    ``training_log.csv``, the returned ``final_loss`` and the optimizer step count.
``schedules.json``
    ``ExponentialDecay(staircase=True)`` and ``Training.get_weighting_from_step``.
``eval_lr_energy.json``
    Test MAE of ``evaluate.evaluate_model`` on the ``.h5`` checkpoint of epoch 0.
``manifest_model.json``
    Versions, legacy commit, all parameters, layer tables and sha256 of every file.

Reads ``synthetic_small.npz`` from ``tests/fixtures`` (written by
``generate_data_golden.py``) and writes into ``--out`` (default ``tests/fixtures/golden``).
``--check`` regenerates everything in a temp dir and compares it bit for bit with the
committed files.

Observation hooks (the legacy code itself is not modified):

* ``Layer.__init__`` of tf_keras is wrapped during the model build to record the
  construction order of all layers (``model.layers`` is in topological order).
* ``Training.get_model`` is replaced by a builder that calls the legacy
  ``Model.get_model`` after ``clear_session()`` + ``set_random_seed(1234)`` and keeps
  a reference to the model, so the in-place trained weights can be read.
* ``evaluate.np`` is replaced by a proxy of NumPy that records the arguments and the
  result of ``np.mean``, so the MAE is available at full float32 precision in addition
  to the printed ``Final MAE loss: {:.5f}``.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import re
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from golden import fixtures, legacy  # noqa: E402

GENERATOR_VERSION = 1
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "fixtures" / "golden"
MANIFEST_NAME = "manifest_model.json"
SOURCE = "synthetic_small.npz"

SEED = 1234
N_MODEL_MOLECULES = 5

# Dataset G-B (binding, see tests/fixtures/README.md).
DATASET = {"grid_size": 1.0, "max_local_level": 2}
SPLIT = {"val_split": 0.1, "test_split": 0.1, "random_state": 42}
SHUFFLE_BUFFER_SIZE = 1  # no shuffling: iteration order = stored pickle order

# Model M-1. M1_local / M1_long are set to n_max_local / n_max_long_range of the dataset.
MODEL = {"embedding_dims": [16, 32, 32], "M2_local": 4, "M2_long": 1, "activation": "elu"}

# variant -> (use_long_range, predict_only_energy)
VARIANTS: dict[str, tuple[bool, bool]] = {
    "lr_energy": (True, True),
    "local_energy": (False, True),
    "lr_forces": (True, False),
    "lr_energy_k1": (True, True),
}
# Forward-only variants with an explicit K1 (M1_local, M1_long) instead of N_max = (10, 20)
# (presets stachyose / legacy-head use K1 30/6). M2 stays 4/1, so the model keeps the
# flatten-and-concatenate branch of get_long_and_local_model.
K1_VARIANTS: dict[str, tuple[int, int]] = {"lr_energy_k1": (6, 24)}

TRAIN_VARIANT = "lr_energy"
TRAIN = {
    "initial_learning_rate": 1e-3,
    "lr_decay_steps": 20,
    "decay_rate": 0.97,
    "decay_steps": 7.5e3,
    "epochs": 1,
    "p_energy_start": 1,
    "p_energy_limit": 1,
    "p_force_start": 1,
    "p_force_limit": 0,
}
EVAL_VARIANT = "lr_energy"

LR_SCHEDULES = [
    {
        "initial_learning_rate": 1e-3,
        "decay_steps": 20,
        "decay_rate": 0.97,
        "steps": [0, 1, 19, 20, 21, 59, 60, 1000],
    },
    {
        "initial_learning_rate": 5e-5,
        "decay_steps": 8.727e4,
        "decay_rate": 0.97,
        "steps": [0, 87269, 87270, 200000],
    },
]
WEIGHTINGS = {
    "decay_steps": 7.5e3,
    "decay_rate": 0.97,
    "steps": [0, 1, 100, 7500, 75000],
    "cases": [
        {"start": 1, "limit": 1},
        {"start": 1, "limit": 0},
        {"start": 0.01, "limit": 1},
        {"start": 1000, "limit": 1},
    ],
}

SPLIT_FILES = {"train": "train.pkl.gzip", "val": "val.pkl.gzip", "test": "test.pkl.gzip"}
INPUT_KEYS_LR = ("local_matrix", "local_numbers", "long_range_matrix", "long_range_features")
WEIGHT_KEY = re.compile(r"^([A-Za-z][A-Za-z0-9]*)__(\d{3})__(\d+)$")
FINAL_MAE = re.compile(r"Final MAE loss: ([-+0-9.eEnaif]+)")


# --------------------------------------------------------------------------- helpers


def _keras(ns: Any) -> Any:
    """``tf.keras`` as seen by the legacy code (tf_keras under ``TF_USE_LEGACY_KERAS=1``)."""
    return ns.Training.tf.keras


def _tf(ns: Any) -> Any:
    return ns.Training.tf


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hyperparameters(M1_local: int, M1_long: int, **train: Any) -> SimpleNamespace:
    """Attribute object as ``Constants.Hyperparameters`` (the legacy code uses attributes).

    M-1 passes ``M1_local, M1_long = n_max_local, n_max_long_range`` (K1 = N_max).
    """
    return SimpleNamespace(
        M1_local=M1_local,
        M2_local=MODEL["M2_local"],
        M1_long=M1_long,
        M2_long=MODEL["M2_long"],
        embedding_dims=list(MODEL["embedding_dims"]),
        **train,
    )


def configure_variant(ns: Any, variant: str, n_max: tuple[int, int]) -> dict[str, Any]:
    use_long_range, predict_only_energy = VARIANTS[variant]
    consts = legacy.configure(
        ns,
        **DATASET,
        n_max_local=n_max[0],
        n_max_long_range=n_max[1],
        predict_only_energy=predict_only_energy,
        use_long_range=use_long_range,
        shuffle_buffer=SHUFFLE_BUFFER_SIZE,
    )
    ns.Constants.Model.activation = MODEL["activation"]
    consts["activation"] = ns.Constants.Model.activation
    return consts


@contextlib.contextmanager
def record_construction(ns: Any):
    """Record every tf_keras layer in the order its ``Layer.__init__`` finishes."""
    layer_cls = _keras(ns).layers.Layer
    original = layer_cls.__init__
    created: list[Any] = []

    def init(self: Any, *args: Any, **kwargs: Any) -> None:
        original(self, *args, **kwargs)
        created.append(self)

    layer_cls.__init__ = init
    try:
        yield created
    finally:
        layer_cls.__init__ = original


def build_model(ns: Any, hp: SimpleNamespace) -> tuple[Any, list[Any]]:
    """``Model.get_model`` after ``clear_session()`` + ``set_random_seed(SEED)``.

    Returns the model and its layers in construction order.
    """
    keras = _keras(ns)
    keras.backend.clear_session()
    keras.utils.set_random_seed(SEED)
    with record_construction(ns) as created:
        model = ns.Model.get_model(hp)
    in_model = {id(layer) for layer in model.layers}
    ordered = [layer for layer in created if id(layer) in in_model]
    if len(ordered) != len(model.layers):
        raise RuntimeError("construction hook missed layers of the model")
    return model, ordered


def _name_suffix(name: str) -> int:
    m = re.fullmatch(r".*?_(\d+)", name)
    return int(m.group(1)) if m else 0


def layer_tables(layers: list[Any]) -> tuple[list[dict], list[dict], list[str]]:
    """(weighted layer table, full construction order, weightless classes in order).

    ``index`` counts layers of one class in construction order. For weighted layers
    it must equal the Keras name suffix (``dense`` = 0, ``dense_7`` = 7).
    """
    per_class: dict[str, int] = {}
    weighted, order, weightless = [], [], []
    for layer in layers:
        cls = type(layer).__name__
        k = per_class.get(cls, 0)
        per_class[cls] = k + 1
        weights = layer.get_weights()
        entry = {
            "class": cls,
            "index": k,
            "legacy_name": layer.name,
            "weight_shapes": [list(w.shape) for w in weights],
        }
        order.append(entry)
        if weights:
            if _name_suffix(layer.name) != k:
                raise RuntimeError(
                    f"{cls} {layer.name}: name suffix != construction index {k} "
                    "(was clear_session() called before the build?)"
                )
            weighted.append(entry)
        else:
            weightless.append(cls)
    return weighted, order, weightless


def weight_arrays(layers: list[Any]) -> dict[str, np.ndarray]:
    """``{Class}__{k:03d}__{j}`` for every weighted layer, k in construction order per class."""
    per_class: dict[str, int] = {}
    out: dict[str, np.ndarray] = {}
    for layer in layers:
        cls = type(layer).__name__
        k = per_class.get(cls, 0)
        per_class[cls] = k + 1
        for j, w in enumerate(layer.get_weights()):
            out[f"{cls}__{k:03d}__{j}"] = np.array(w, copy=True)
    return out


# --------------------------------------------------------------------------- dataset


def create_dataset(ns: Any, source: Path, work: Path) -> tuple[Path, dict[str, Any]]:
    """Legacy ``create_dataset`` for G-B into ``work``; returns the folder and the splits."""
    import pandas as pd

    legacy.configure(ns, **DATASET, shuffle_buffer=SHUFFLE_BUFFER_SIZE)
    with contextlib.redirect_stdout(io.StringIO()):
        ns.Dataset.create_dataset(
            [str(source)],
            DATASET["grid_size"],
            save_folder=str(work / "ds"),
            n_samples_per=None,
            **{k: SPLIT[k] for k in ("val_split", "test_split")},
        )
    folders = sorted(work.glob("ds_*"))
    if len(folders) != 1:
        raise RuntimeError(f"expected one legacy dataset folder, got {folders}")
    dfs = {s: pd.read_pickle(folders[0] / f, compression="gzip") for s, f in SPLIT_FILES.items()}
    return folders[0], dfs


def molecule_inputs(ns: Any, df: Any, data: dict[str, np.ndarray], n: int) -> dict[str, np.ndarray]:
    """Padded inputs of the first ``n`` stored molecules (legacy row order) plus ``order``."""
    keys = ns.Constants.Keys
    rows = df.iloc[:n]
    cols = {
        "local_matrix": keys.LOCAL_DISTANCE_MATRIX_KEY,
        "local_numbers": keys.LOCAL_ATOMIC_NUMBERS_KEY,
        "long_range_matrix": keys.LONG_RANGE_DISTANCE_MATRIX_KEY,
        "long_range_features": keys.LONG_RANGE_ATOMIC_FEATURES_KEY,
    }
    out: dict[str, np.ndarray] = {}
    for name, col in cols.items():
        parts = list(rows[col])
        if any(p.dtype != np.float32 for p in parts):
            raise TypeError(f"{col}: legacy padding is not float32")
        out[f"x__{name}"] = np.concatenate(parts)
    counts = np.array([len(p) for p in rows[keys.LOCAL_DISTANCE_MATRIX_KEY]], dtype=np.int64)
    index = rows.index.to_numpy().astype(np.int64)
    orders = []
    for i in index:
        R = data["R"][i]
        grid = ns.Dataset.create_grid(R.tolist(), data["z"].tolist(), DATASET["grid_size"])
        orders.append(legacy.legacy_atom_order(grid, R))
    out["index"] = index
    out["energy"] = rows[keys.ENERGY_KEY].to_numpy().astype(np.float64)
    out["atom_offsets"] = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    out["order"] = np.concatenate(orders)
    return out


# --------------------------------------------------------------------------- model golden


def model_golden(
    ns: Any, variant: str, n_max: tuple[int, int], inputs: dict[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    tf = _tf(ns)
    consts = configure_variant(ns, variant, n_max)
    m1 = K1_VARIANTS.get(variant, n_max)
    hp = hyperparameters(*m1)
    model, layers = build_model(ns, hp)
    weighted, order, weightless = layer_tables(layers)
    weights = weight_arrays(layers)
    if {WEIGHT_KEY.match(k).group(1) for k in weights} - {"Dense", "LayerNormalization"}:
        raise RuntimeError(f"unexpected weighted layer classes in {sorted(weights)}")
    count_params = int(model.count_params())
    if count_params != sum(w.size for w in weights.values()):
        raise RuntimeError("count_params differs from the sum of the weight sizes")

    use_long_range, _ = VARIANTS[variant]
    x_keys = INPUT_KEYS_LR if use_long_range else INPUT_KEYS_LR[:2]
    offsets = inputs["atom_offsets"]
    outputs, sum_energy, sum_all = [], [], []
    for m in range(len(offsets) - 1):
        a, b = int(offsets[m]), int(offsets[m + 1])
        x = [tf.constant(inputs[f"x__{key}"][a:b]) for key in x_keys]
        pred = model(x, training=False)
        outputs.append(pred.numpy())
        # Training.train_step: reduce_sum(prediction[:, 0]); evaluate.py (B05) and
        # train_step_energy_only: reduce_sum(prediction) over all outputs.
        sum_energy.append(tf.math.reduce_sum(pred[:, 0]).numpy())
        sum_all.append(tf.math.reduce_sum(pred).numpy())

    arrays: dict[str, np.ndarray] = {
        **weights,
        "count_params": np.int64(count_params),
        "seed": np.int64(SEED),
        "n_max_local": np.int64(n_max[0]),
        "n_max_long_range": np.int64(n_max[1]),
        "use_long_range": np.bool_(use_long_range),
        "predict_only_energy": np.bool_(VARIANTS[variant][1]),
        "index": inputs["index"],
        "energy": inputs["energy"],
        "atom_offsets": inputs["atom_offsets"],
        "order": inputs["order"],
        **{f"x__{key}": inputs[f"x__{key}"] for key in x_keys},
        "output": np.concatenate(outputs),
        "sum_energy": np.asarray(sum_energy, dtype=np.float32),
        "sum_all": np.asarray(sum_all, dtype=np.float32),
    }
    if variant in K1_VARIANTS:
        arrays["M1_local"] = np.int64(m1[0])
        arrays["M1_long"] = np.int64(m1[1])
    info = {
        "use_long_range": use_long_range,
        "predict_only_energy": VARIANTS[variant][1],
        "constants": consts,
        "hyperparameters": vars(hp),
        "count_params": count_params,
        "n_molecules": len(offsets) - 1,
        "n_atoms": int(offsets[-1]),
        "inputs": list(x_keys),
        "output_width": int(arrays["output"].shape[1]),
        "layers": weighted,
        "weightless_layers": weightless,
        "construction_order": order,
        "model_layers": [layer.name for layer in model.layers],
    }
    if variant in K1_VARIANTS:
        info["M1_local"], info["M1_long"] = m1
        info["note"] = "forward only (no training/eval golden); K1 != N_max"
    return arrays, info


# --------------------------------------------------------------------------- training


def train_golden(
    ns: Any, folder: Path, work: Path, n_max: tuple[int, int], start: dict[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], dict[str, Any], Any, Path]:
    """One epoch of the unchanged ``start_training``; returns arrays, info, model, h5 path."""
    consts = configure_variant(ns, TRAIN_VARIANT, n_max)
    ns.Constants.Dataset.FOLDER = str(folder)
    hp = hyperparameters(*n_max, **TRAIN)
    holder: dict[str, Any] = {}

    def get_model(hparams: Any) -> Any:
        model, layers = build_model(ns, hparams)
        holder["model"], holder["layers"] = model, layers
        holder["start"] = weight_arrays(layers)
        return model

    run_cwd = work / "cwd"
    run_cwd.mkdir()
    original_get_model = ns.Training.get_model
    original_cwd = Path.cwd()
    log = io.StringIO()
    ns.Training.get_model = get_model
    try:
        os.chdir(run_cwd)
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            final_loss = ns.Training.start_training(hp)
    finally:
        os.chdir(original_cwd)
        ns.Training.get_model = original_get_model

    for key, value in start.items():
        if holder["start"][key].tobytes() != value.tobytes():
            raise RuntimeError(f"training start weight {key} differs from model_{TRAIN_VARIANT}")
    if set(holder["start"]) != set(start):
        raise RuntimeError("training model and model golden differ in their weights")

    runs = sorted((work / "Checkpoints").glob("Model-*"))
    if len(runs) != 1:
        raise RuntimeError(f"expected one legacy checkpoint dir, got {runs}")
    run = runs[0]
    csv_lines = (run / "training_log.csv").read_text().splitlines()
    header, rows = csv_lines[0].split(","), [line.split(",") for line in csv_lines[1:]]
    if len(rows) != TRAIN["epochs"]:
        raise RuntimeError(f"training_log.csv has {len(rows)} rows, expected {TRAIN['epochs']}")
    row = dict(zip(header, rows[-1], strict=True))
    logging = ns.Constants.Logging
    model = holder["model"]
    steps = int(model.optimizer.iterations.numpy())

    arrays: dict[str, np.ndarray] = {
        **weight_arrays(holder["layers"]),
        "energy_loss": np.float32(row[logging.train_energy_loss_key]),
        "val_energy_loss": np.float32(row[logging.val_energy_loss_key]),
        "final_loss": np.float32(final_loss),
        "steps": np.int64(steps),
        "epochs": np.int64(TRAIN["epochs"]),
    }
    h5 = run / "model_epoch_0.h5"
    info = {
        "start_weights": f"model_{TRAIN_VARIANT}.npz",
        "constants": consts,
        "hyperparameters": vars(hp),
        "steps": steps,
        "energy_loss": float(arrays["energy_loss"]),
        "val_energy_loss": float(arrays["val_energy_loss"]),
        "final_loss": float(arrays["final_loss"]),
        "final_loss_type": type(final_loss).__name__,
        "training_log": {"header": header, "rows": rows},
        "checkpoint_files": sorted(p.name for p in run.iterdir()),
        "dropout_active": False,
        "note": "model(...) is called without training=True in train_step (B02): dropout is off",
    }
    return arrays, info, model, h5


# --------------------------------------------------------------------------- evaluation


class _MeanRecorder:
    """NumPy proxy that records ``mean`` calls (installed as ``evaluate.np``)."""

    def __init__(self, module: Any) -> None:
        self._np = module
        self.calls: list[tuple[np.ndarray, Any]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._np, name)

    def mean(self, a: Any, *args: Any, **kwargs: Any) -> Any:
        result = self._np.mean(a, *args, **kwargs)
        self.calls.append((np.array(a, copy=True), result))
        return result


def in_memory_mae(ns: Any, model: Any, test_pkl: Path) -> tuple[np.float32, np.ndarray]:
    """``evaluate_model`` formula with the in-memory model: mean_k |sum_i E_i - E_k|."""
    tf = _tf(ns)
    losses = _keras(ns).losses
    with contextlib.redirect_stdout(io.StringIO()):
        test_ds, test_size = ns.Dataset.create_tf_dataset(str(test_pkl))

    @tf.function
    def test_step(inputs: Any, outputs: Any) -> Any:
        prediction = model([inputs[0], inputs[1], inputs[2], inputs[3]], training=False)
        return losses.MAE(outputs, tf.math.reduce_sum(prediction))

    per = np.zeros(test_size, dtype=np.float32)
    for i, (x, y) in enumerate(test_ds):
        per[i] = test_step(x, y).numpy()
    return np.mean(per), per


def eval_golden(ns: Any, h5: Path, test_pkl: Path, model: Any) -> dict[str, Any]:
    configure_variant(ns, EVAL_VARIANT, (model.inputs[0].shape[1], model.inputs[2].shape[1]))
    recorder = _MeanRecorder(ns.evaluate.np)
    out = io.StringIO()
    error = None
    ns.evaluate.np = recorder
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            ns.evaluate.evaluate_model(str(h5), str(test_pkl), use_long_range=True)
    except Exception as exc:  # noqa: BLE001 - fallback path is part of the contract
        error = f"{type(exc).__name__}: {exc}"
    finally:
        ns.evaluate.np = recorder._np

    mae_mem, per_mem = in_memory_mae(ns, model, test_pkl)
    result: dict[str, Any] = {
        "variant": EVAL_VARIANT,
        "checkpoint": "model_epoch_0.h5",
        "dataset": "test.pkl.gzip",
        "formula": "mean_k |sum_i E_i - E_k| over the test molecules (float32)",
        "mae_in_memory": float(mae_mem),
        "per_molecule_abs_error_in_memory": [float(v) for v in per_mem],
    }
    printed = FINAL_MAE.search(out.getvalue())
    if error is None and printed is not None and len(recorder.calls) == 1:
        per, mae = recorder.calls[0]
        if f"{mae:.5f}" != printed.group(1):
            raise RuntimeError(f"recorded MAE {mae} does not match printed {printed.group(1)}")
        result.update(
            {
                "path": "evaluate_model_h5",
                "mae": float(mae),
                "mae_printed": printed.group(1),
                "per_molecule_abs_error": [float(v) for v in per],
                "test_size": len(per),
                "h5_equals_in_memory": bool(
                    np.float32(mae).tobytes() == np.float32(mae_mem).tobytes()
                    and per.tobytes() == per_mem.tobytes()
                ),
            }
        )
    else:
        result.update(
            {
                "path": "in_memory_fallback",
                "fallback_reason": error or "no 'Final MAE loss' line / unexpected np.mean calls",
                "mae": float(mae_mem),
                "mae_printed": printed.group(1) if printed else None,
                "per_molecule_abs_error": [float(v) for v in per_mem],
                "test_size": len(per_mem),
            }
        )
    return result


# --------------------------------------------------------------------------- schedules


def schedules_golden(ns: Any) -> dict[str, Any]:
    keras = _keras(ns)
    lr = []
    for cfg in LR_SCHEDULES:
        sched = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=cfg["initial_learning_rate"],
            decay_steps=cfg["decay_steps"],
            decay_rate=cfg["decay_rate"],
            staircase=True,
        )
        lr.append(
            {
                **cfg,
                "staircase": True,
                "dtype": "float32",
                "values": [float(np.float32(sched(s).numpy())) for s in cfg["steps"]],
            }
        )
    weighting = []
    for case in WEIGHTINGS["cases"]:
        weighting.append(
            {
                **case,
                "decay_steps": WEIGHTINGS["decay_steps"],
                "decay_rate": WEIGHTINGS["decay_rate"],
                "steps": WEIGHTINGS["steps"],
                "values": [
                    float(
                        ns.Training.get_weighting_from_step(
                            s,
                            case["limit"],
                            case["start"],
                            WEIGHTINGS["decay_steps"],
                            WEIGHTINGS["decay_rate"],
                        )
                    )
                    for s in WEIGHTINGS["steps"]
                ],
            }
        )
    return {
        "learning_rate": lr,
        "weighting": weighting,
        "weighting_formula": "w_limit * (1 - f) + w_start * f, f = decay_rate**(step/decay_steps)",
    }


# --------------------------------------------------------------------------- generate


def versions(ns: Any) -> dict[str, str]:
    import pandas as pd

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "tensorflow": _tf(ns).__version__,
        "tf_keras": importlib.metadata.version("tf_keras"),
        "h5py": importlib.metadata.version("h5py"),
    }


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def generate(out: Path, ref: str, data_dir: Path | None = None) -> dict[str, Any]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    source = Path(data_dir or fixtures.fixtures_dir()) / SOURCE
    data = _load_npz(source)

    src = legacy.extract_legacy(ref)
    work = Path(tempfile.mkdtemp(prefix="lrmlp_model_golden_"))
    try:
        ns = legacy.load_legacy(src)
        folder, dfs = create_dataset(ns, source, work)
        keys = ns.Constants.Keys
        n_max = (
            int(dfs["train"][keys.LOCAL_DISTANCE_MATRIX_KEY].iloc[0].shape[1]),
            int(dfs["train"][keys.LONG_RANGE_DISTANCE_MATRIX_KEY].iloc[0].shape[1]),
        )
        inputs = molecule_inputs(ns, dfs["test"], data, N_MODEL_MOLECULES)

        manifest: dict[str, Any] = {
            "generator": "tests/golden/generate_model_golden.py",
            "generator_version": GENERATOR_VERSION,
            "legacy_ref": ref,
            "legacy_commit": legacy.commit_sha(ref),
            "versions": versions(ns),
            "dataset": {
                "name": "G-B",
                "source": SOURCE,
                "source_sha256": sha256(source),
                **DATASET,
                **SPLIT,
                "shuffle_buffer_size": SHUFFLE_BUFFER_SIZE,
                "splits": {s: len(df) for s, df in dfs.items()},
                "n_max_local": n_max[0],
                "n_max_long_range": n_max[1],
                "model_inputs": f"first {N_MODEL_MOLECULES} molecules of the test split",
            },
            "model": {
                **MODEL,
                "M1_local": n_max[0],
                "M1_long": n_max[1],
                "seed": SEED,
                "seeding": "tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(seed)",
                "weight_keys": "{Class}__{k:03d}__{j}: k = construction index per class, "
                "j = index in layer.get_weights()",
            },
            "variants": {},
        }

        start = None
        for variant in VARIANTS:
            arrays, info = model_golden(ns, variant, n_max, inputs)
            fixtures.save_npz(out / f"model_{variant}.npz", arrays)
            manifest["variants"][variant] = info
            if variant == TRAIN_VARIANT:
                start = {k: v for k, v in arrays.items() if WEIGHT_KEY.match(k)}
            print(
                f"model {variant}: count_params={info['count_params']} "
                f"sum_energy={arrays['sum_energy'].tolist()}",
                flush=True,
            )

        arrays, info, model, h5 = train_golden(ns, folder, work, n_max, start)
        fixtures.save_npz(out / f"train_{TRAIN_VARIANT}.npz", arrays)
        manifest["train"] = {TRAIN_VARIANT: info}
        print(
            f"train {TRAIN_VARIANT}: steps={info['steps']} energy_loss={info['energy_loss']} "
            f"val_energy_loss={info['val_energy_loss']} final_loss={info['final_loss']}",
            flush=True,
        )

        ev = eval_golden(ns, h5, folder / SPLIT_FILES["test"], model)
        _write_json(out / f"eval_{EVAL_VARIANT}.json", ev)
        manifest["eval"] = {EVAL_VARIANT: {k: ev[k] for k in ("path", "mae", "test_size")}}
        print(f"eval {EVAL_VARIANT}: path={ev['path']} mae={ev['mae']}", flush=True)

        _write_json(out / "schedules.json", schedules_golden(ns))
    finally:
        legacy.cleanup_legacy(src)
        import shutil

        shutil.rmtree(work, ignore_errors=True)

    files = [f"model_{v}.npz" for v in VARIANTS] + [
        f"train_{TRAIN_VARIANT}.npz",
        f"eval_{EVAL_VARIANT}.json",
        "schedules.json",
    ]
    manifest["files"] = {
        f"golden/{name}": {"sha256": sha256(out / name), "bytes": (out / name).stat().st_size}
        for name in files
    }
    _write_json(out / MANIFEST_NAME, manifest)
    return manifest


# --------------------------------------------------------------------------- check


def compare_npz(new: Path, committed: Path) -> list[str]:
    diffs = []
    a, b = _load_npz(new), _load_npz(committed)
    if list(a) != list(b):
        diffs.append(
            f"{committed.name}: keys differ: only new {sorted(set(a) - set(b))}, "
            f"only committed {sorted(set(b) - set(a))}"
        )
    for key in a:
        if key not in b:
            continue
        x, y = a[key], b[key]
        if x.dtype != y.dtype or x.shape != y.shape:
            diffs.append(
                f"{committed.name}:{key}: {x.dtype}{x.shape} != committed {y.dtype}{y.shape}"
            )
        elif x.tobytes() != y.tobytes():
            if np.issubdtype(x.dtype, np.number):
                dev = float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64))))
                diffs.append(f"{committed.name}:{key}: values differ, max |diff| = {dev:.3e}")
            else:
                diffs.append(f"{committed.name}:{key}: values differ")
    return diffs


def _strip(manifest: dict[str, Any]) -> dict[str, Any]:
    m = dict(manifest)
    m.pop("versions", None)
    m["files"] = sorted(manifest.get("files", {}))
    return m


def _dict_diff(a: Any, b: Any, path: str = "") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}.{k}: present only in {'committed' if k in b else 'new'}")
            else:
                out.extend(_dict_diff(a[k], b[k], f"{path}.{k}"))
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        out = []
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            out.extend(_dict_diff(x, y, f"{path}[{i}]"))
        return out
    return [] if a == b else [f"{path}: {a!r} != committed {b!r}"]


def check(committed_out: Path, ref: str) -> int:
    committed_out = Path(committed_out)
    manifest_path = committed_out / MANIFEST_NAME
    if not manifest_path.is_file():
        print(f"DIFF: {manifest_path} missing")
        return 1
    committed = json.loads(manifest_path.read_text())
    with tempfile.TemporaryDirectory(prefix="lrmlp_model_golden_check_") as td:
        new_out = Path(td) / "golden"
        new = generate(new_out, ref, data_dir=committed_out.parent)
        diffs = [f"manifest{d}" for d in _dict_diff(_strip(new), _strip(committed))]
        notes = []
        for rel in sorted(set(new["files"]) & set(committed["files"])):
            name = Path(rel).name
            committed_file = committed_out / name
            if not committed_file.is_file():
                diffs.append(f"{rel}: missing in {committed_out}")
                continue
            if name.endswith(".npz"):
                diffs.extend(compare_npz(new_out / name, committed_file))
            else:
                diffs.extend(
                    f"{name}{d}"
                    for d in _dict_diff(
                        json.loads((new_out / name).read_text()),
                        json.loads(committed_file.read_text()),
                    )
                )
            if new["files"][rel]["sha256"] != sha256(committed_file):
                notes.append(f"note: {rel}: file bytes differ")
        if new.get("versions") != committed.get("versions"):
            notes.append(
                f"note: versions differ: new {new.get('versions')} "
                f"vs committed {committed.get('versions')}"
            )
    for line in notes:
        print(line)
    if diffs:
        print("DIFF (not bit-identical; CPU nondeterminism or environment change):")
        for line in diffs:
            print("  " + line)
        return 1
    print("OK: identical")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="golden dir")
    parser.add_argument(
        "--check", action="store_true", help="regenerate in a temp dir and compare bit for bit"
    )
    parser.add_argument(
        "--ref", default=legacy.LEGACY_REF, help="legacy commit (default %(default)s)"
    )
    args = parser.parse_args(argv)
    t0 = time.perf_counter()
    if args.check:
        code = check(args.out, args.ref)
    else:
        generate(args.out, args.ref)
        print(f"wrote model golden fixtures to {args.out}")
        code = 0
    print(f"runtime: {time.perf_counter() - t0:.1f} s")
    return code


if __name__ == "__main__":
    sys.exit(main())
