"""Loaders and helpers for the golden fixtures (NumPy only, no dependency on ``lrmlp``).

File layout (all under ``tests/fixtures/``)::

    synthetic_small.npz             60 frames, aspirin-like (MD17 keys E, F, R, z)
    stachyose_like.npz              5 frames, 87 atoms
    edge_cases.npz                  <case>__R (N, 3), <case>__z (N,)
    golden/data_descriptors_<case>.npz
    golden/data_dataset_<GA|GB>.npz
    golden/manifest_data.json

Descriptor golden (``load_descriptor_golden``), per frame ``f``, all in **legacy
row order** (cell traversal order, see ``f{f}__order``):

    f{f}__order          (N,)      input atom index of each legacy row
    f{f}__positions      (N, 3)    input positions (input order)
    f{f}__numbers        (N,)      atomic numbers (input order)
    f{f}__local_rows     (sum k, 4) float64  [s, s*x/r, s*y/r, s*z/r]
    f{f}__local_counts   (N,)      k per atom
    f{f}__local_numbers  (sum k, 2) [Z_i, Z_j]
    f{f}__lr_rows        (sum m, 4) float64  [1/r, x/r^2, y/r^2, z/r^2]
    f{f}__lr_counts      (N,)      m per atom
    f{f}__lr_features    (sum m, 101) [Z_i, counts of Z = 1..100]
    f{f}__n_max          (2,)      (N_max_local, N_max_long_range) of the frame

plus scalars ``n_frames``, ``grid_size``, ``max_local_level``, ``cut_off``,
``inner_cut_off``, ``sort_matrix`` and ``frame_names``.

Dataset golden (``load_dataset_golden``), per split in ``train``/``val``/``test``:

    <split>__index                  (S,)  DataFrame index = frame index, stored order
    <split>__energy                 (S,)
    <split>__forces                 (A, 3)  input atom order per molecule
    <split>__atom_offsets           (S+1,)
    <split>__n_max                  (S, 2)  per-molecule (n_max_local, n_max_long_range)
    <split>__local_matrix           (A, n_loc, 4)   float32, legacy row order
    <split>__local_numbers          (A, n_loc, 2)   float32
    <split>__long_range_matrix      (A, n_lr, 4)    float32
    <split>__long_range_features    (A, n_lr, 101)  float32
    <split>__order                  (A,)  legacy atom order per molecule, concatenated

plus global ``n_max_local``, ``n_max_long_range``.
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)


def fixtures_dir() -> Path:
    """``tests/fixtures``."""
    return Path(__file__).resolve().parents[1] / "fixtures"


def golden_dir() -> Path:
    """``tests/fixtures/golden``."""
    return fixtures_dir() / "golden"


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def load_descriptor_golden(case: str) -> dict[str, np.ndarray]:
    """Load ``data_descriptors_<case>.npz`` as a flat ``{key: array}`` dict."""
    return _load(golden_dir() / f"data_descriptors_{case}.npz")


def load_dataset_golden(name: str) -> dict[str, np.ndarray]:
    """Load ``data_dataset_<name>.npz`` (``name`` is ``"GA"`` or ``"GB"``)."""
    return _load(golden_dir() / f"data_dataset_{name}.npz")


def load_manifest() -> dict[str, Any]:
    """Load ``golden/manifest_data.json``."""
    return json.loads((golden_dir() / "manifest_data.json").read_text())


def load_synthetic(name: str) -> dict[str, np.ndarray]:
    """Load ``<name>.npz`` from ``tests/fixtures`` (e.g. ``"synthetic_small"``)."""
    return _load(fixtures_dir() / f"{name}.npz")


def load_edge_cases() -> dict[str, dict[str, np.ndarray]]:
    """Load ``edge_cases.npz`` as ``{case: {"R": (N, 3), "z": (N,)}}``."""
    flat = _load(fixtures_dir() / "edge_cases.npz")
    cases: dict[str, dict[str, np.ndarray]] = {}
    for key, value in flat.items():
        case, field = key.rsplit("__", 1)
        cases.setdefault(case, {})[field] = value
    return cases


def frame(golden: dict[str, np.ndarray], f: int) -> dict[str, np.ndarray]:
    """All ``f{f}__*`` entries of a descriptor golden, prefix stripped."""
    prefix = f"f{f}__"
    return {k[len(prefix) :]: v for k, v in golden.items() if k.startswith(prefix)}


def split_ragged(rows: np.ndarray, counts: Sequence[int] | np.ndarray) -> list[np.ndarray]:
    """Split concatenated rows into one array per atom (``len(counts)`` parts)."""
    counts = np.asarray(counts, dtype=np.int64)
    if int(counts.sum()) != len(rows):
        raise ValueError(f"counts sum to {int(counts.sum())}, but there are {len(rows)} rows")
    return np.split(rows, np.cumsum(counts)[:-1])


def to_input_order(per_atom_list: Sequence[Any], order: Sequence[int] | np.ndarray) -> list:
    """Reorder per-atom data from legacy row order to input atom order.

    ``order[k]`` is the input index of legacy row ``k``, so the result satisfies
    ``result[order[k]] == per_atom_list[k]``.
    """
    order = np.asarray(order, dtype=np.int64)
    if len(order) != len(per_atom_list):
        raise ValueError("order and per_atom_list differ in length")
    result: list = [None] * len(order)
    for k, i in enumerate(order):
        result[int(i)] = per_atom_list[k]
    return result


def save_npz(path: str | Path, arrays: dict[str, np.ndarray]) -> None:
    """Write a compressed ``.npz`` byte-reproducibly.

    Same container as ``np.savez_compressed`` (zip of ``.npy``, deflate), readable
    with ``np.load``, but with a fixed zip timestamp so identical arrays give
    identical bytes (and sha256). ``np.savez_compressed`` stamps the current time.
    """
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for key, value in arrays.items():
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.asanyarray(value), allow_pickle=False)
            info = zipfile.ZipInfo(f"{key}.npy", date_time=_ZIP_EPOCH)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, buf.getvalue())


# --- independent reference of the legacy assignment (used by tests and the manifest) ---


def cell_indices(positions: np.ndarray, grid_size: float) -> np.ndarray:
    """Legacy grid index ``floor((R - R.min(axis=0)) / grid_size)``, shape (N, 3)."""
    positions = np.asarray(positions, dtype=np.float64)
    return np.floor((positions - positions.min(axis=0)) / grid_size).astype(np.int32)


def legacy_traversal_order(positions: np.ndarray, grid_size: float) -> np.ndarray:
    """Legacy row order: cells in lexicographic index order, atoms in input order within a cell."""
    idx = cell_indices(positions, grid_size)
    return np.lexsort((np.arange(len(idx)), idx[:, 2], idx[:, 1], idx[:, 0]))


def legacy_pair_classes(
    positions: np.ndarray, grid_size: float, max_local_level: int, cut_off: float
) -> dict[str, np.ndarray]:
    """Classify every ordered pair (i, j), i != j, as the legacy code treats it.

    Returns boolean (N, N) masks in input order:

    ``local``          level <= l_max + 1 and r <= r_c   -> local row
    ``dropped``        level <= l_max and r > r_c        -> lost (A02)
    ``reduced``        level == l_max + 1 and r > r_c    -> in the reduced long-range cell
    ``centroid``       level >= l_max + 2                -> only via the cell centroid
    ``centroid_in_rc`` level >= l_max + 2 and r <= r_c   -> A01
    """
    positions = np.asarray(positions, dtype=np.float64)
    idx = cell_indices(positions, grid_size)
    level = np.max(np.abs(idx[:, None, :] - idx[None, :, :]), axis=-1)
    diff = positions[:, None, :] - positions[None, :, :]
    r = np.linalg.norm(diff, ord=2, axis=-1)
    off = ~np.eye(len(positions), dtype=bool)
    inside = r <= cut_off
    return {
        "level": level,
        "local": off & (level <= max_local_level + 1) & inside,
        "dropped": off & (level <= max_local_level) & ~inside,
        "reduced": off & (level == max_local_level + 1) & ~inside,
        "centroid": off & (level >= max_local_level + 2),
        "centroid_in_rc": off & (level >= max_local_level + 2) & inside,
    }


def legacy_expected_counts(
    positions: np.ndarray, grid_size: float, max_local_level: int, cut_off: float
) -> tuple[np.ndarray, np.ndarray]:
    """Expected (local, long-range) row counts per atom in input order.

    Long-range rows: one per non-empty cell on level >= l_max + 2 plus one per
    cell on level l_max + 1 that keeps at least one atom beyond r_c.
    """
    cls = legacy_pair_classes(positions, grid_size, max_local_level, cut_off)
    idx = cell_indices(positions, grid_size)
    _, cell_of = np.unique(idx, axis=0, return_inverse=True)
    cell_of = cell_of.reshape(-1)
    n = len(positions)
    lr = np.zeros(n, dtype=np.int64)
    for i in range(n):
        cells = set()
        for j in range(n):
            if cls["centroid"][i, j] or cls["reduced"][i, j]:
                cells.add(int(cell_of[j]))
        lr[i] = len(cells)
    return cls["local"].sum(axis=1).astype(np.int64), lr


# --- model, training, schedule and evaluation golden (T03, generate_model_golden.py) ---
#
# model_<variant>.npz (variant in lr_energy, local_energy, lr_forces), model M-1 on G-B:
#
#     {Class}__{k:03d}__{j}       initial weights; k = construction index within the layer
#                                 class (= Keras 2 name suffix), j = index in get_weights()
#     count_params, seed, n_max_local, n_max_long_range, use_long_range, predict_only_energy
#     index (5,), energy (5,), atom_offsets (6,), order (A,)   first 5 test molecules
#     x__local_matrix, x__local_numbers [, x__long_range_matrix, x__long_range_features]
#                                 padded float32 inputs in legacy row order
#     output (A, 1 | 4)           model(x, training=False) per molecule, concatenated
#     sum_energy (5,)             reduce_sum(output[:, 0]) per molecule
#     sum_all (5,)                reduce_sum(output) per molecule (evaluate.py, B05)
#
# train_lr_energy.npz: weights after one epoch ({Class}__{k:03d}__{j}), energy_loss,
# val_energy_loss, final_loss, steps, epochs. Start weights = model_lr_energy.npz.
# schedules.json, eval_lr_energy.json and manifest_model.json are plain JSON.

_WEIGHT_KEY = re.compile(r"^([A-Za-z][A-Za-z0-9]*)__(\d{3})__(\d+)$")


def load_model_manifest() -> dict[str, Any]:
    """Load ``golden/manifest_model.json``."""
    return json.loads((golden_dir() / "manifest_model.json").read_text())


def weights_by_class(arrays: dict[str, np.ndarray]) -> dict[str, list[list[np.ndarray]]]:
    """Group ``{Class}__{k:03d}__{j}`` entries as ``{Class: [[w_0, w_1, ...] per layer k]}``.

    Layers are in construction order per class (``k`` = 0, 1, ...), the inner list
    follows ``layer.get_weights()``.
    """
    grouped: dict[str, dict[int, dict[int, np.ndarray]]] = {}
    for key, value in arrays.items():
        m = _WEIGHT_KEY.match(key)
        if m:
            layer = grouped.setdefault(m.group(1), {}).setdefault(int(m.group(2)), {})
            layer[int(m.group(3))] = value
    out: dict[str, list[list[np.ndarray]]] = {}
    for cls, layers in grouped.items():
        if sorted(layers) != list(range(len(layers))):
            raise ValueError(f"{cls}: layer indices are not contiguous: {sorted(layers)}")
        out[cls] = []
        for k in range(len(layers)):
            ws = layers[k]
            if sorted(ws) != list(range(len(ws))):
                raise ValueError(f"{cls}[{k}]: weight indices are not contiguous")
            out[cls].append([ws[j] for j in range(len(ws))])
    return out


def _split_weights(raw: dict[str, np.ndarray]) -> dict[str, Any]:
    rest: dict[str, Any] = {k: v for k, v in raw.items() if not _WEIGHT_KEY.match(k)}
    rest["weights_by_class"] = weights_by_class(raw)
    return rest


def load_model_golden(variant: str) -> dict[str, Any]:
    """Load ``model_<variant>.npz``.

    Returns all non-weight arrays (see the key list above) plus ``weights_by_class``
    and, from the manifest, ``layers`` (weighted layer table), ``weightless_layers``
    (classes without weights in construction order) and ``construction_order``.
    """
    out = _split_weights(_load(golden_dir() / f"model_{variant}.npz"))
    info = load_model_manifest()["variants"][variant]
    for key in ("layers", "weightless_layers", "construction_order"):
        out[key] = info[key]
    return out


def load_train_golden(variant: str) -> dict[str, Any]:
    """Load ``train_<variant>.npz``.

    ``weights_by_class`` holds the weights after training; ``start_weights_by_class``
    the initial weights from the referenced model golden.
    """
    out = _split_weights(_load(golden_dir() / f"train_{variant}.npz"))
    info = load_model_manifest()["train"][variant]
    start = _load(golden_dir() / info["start_weights"])
    out["start_weights_by_class"] = weights_by_class(start)
    out["info"] = info
    return out


def load_schedules_golden() -> dict[str, Any]:
    """Load ``golden/schedules.json`` (learning-rate and loss-weighting schedules)."""
    return json.loads((golden_dir() / "schedules.json").read_text())


def load_eval_golden(variant: str) -> dict[str, Any]:
    """Load ``golden/eval_<variant>.json`` (test MAE and per-molecule absolute errors)."""
    return json.loads((golden_dir() / f"eval_{variant}.json").read_text())
