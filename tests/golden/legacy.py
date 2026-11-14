"""Access to the unchanged legacy code (``Neural-Network/`` at a fixed commit).

The legacy modules are never imported from the working tree. They are exported
with ``git archive <ref> Neural-Network`` into a temporary directory and imported
from there. Requires the golden environment (TF 2.21 + tf_keras) and
``TF_USE_LEGACY_KERAS=1``; run with ``PYTHONDONTWRITEBYTECODE=1``.

Interface (also used by T03)
----------------------------
``extract_legacy(ref="9413bfd") -> Path``
    Exports ``Neural-Network/`` and returns ``<tmpdir>/Neural-Network``. The caller
    owns the temporary directory; remove it with ``cleanup_legacy(src)``.

``load_legacy(src) -> SimpleNamespace(Constants, Dataset, Model, Training, evaluate)``
    Imports the five legacy modules from ``src`` (fresh, replacing any cached
    modules of the same names). Raises ``RuntimeError`` unless
    ``TF_USE_LEGACY_KERAS=1``. ``ns.Dataset`` is the *module* ``Dataset.py``; the
    constants class is ``ns.Constants.Dataset``.

``configure(ns, *, grid_size, max_local_level, ...) -> dict``
    Patches the frozen-dataclass constants at class level (the legacy modules
    import the same class objects, so the patch is seen everywhere) including the
    derived values ``CUT_OFF``, ``INNER_CUT_OFF`` and ``Model.input_shape_*``.
    Returns the effective constants as a plain dict.

``legacy_atom_order(grid, positions) -> np.ndarray``
    For a grid from ``ns.Dataset.create_grid``: the input index of each legacy
    row (cell traversal order), found by exact position equality.

``commit_sha(ref) -> str``
    Full commit SHA of ``ref``.
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_REF = "9413bfd"
LEGACY_MODULES = ("Constants", "Dataset", "Model", "Training", "evaluate")

# Fixed constants of all golden configurations.
MAX_ATOM_ELEMENTS = 100
USE_CHARGE_NUMBER = True
INNER_CUT_OFF_OFFSET = 0.2


def commit_sha(ref: str = LEGACY_REF) -> str:
    """Full SHA of ``ref`` in this repository."""
    out = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "--verify", f"{ref}^{{commit}}"],
        check=True,
        capture_output=True,
        text=True,
    )
    return out.stdout.strip()


def extract_legacy(ref: str = LEGACY_REF) -> Path:
    """Export ``Neural-Network/`` at ``ref`` into a new temp dir; return the module dir."""
    tmp = Path(tempfile.mkdtemp(prefix="lrmlp_legacy_"))
    archive = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "archive", ref, "Neural-Network"],
        check=True,
        capture_output=True,
    )
    subprocess.run(["tar", "-x", "-C", str(tmp)], input=archive.stdout, check=True)
    src = tmp / "Neural-Network"
    if not (src / "Dataset.py").is_file():
        raise RuntimeError(f"git archive {ref} did not produce Neural-Network/Dataset.py")
    return src


def cleanup_legacy(src: Path) -> None:
    """Remove the temp dir created by ``extract_legacy`` and unload the legacy modules."""
    src = Path(src)
    for name in LEGACY_MODULES:
        mod = sys.modules.get(name)
        if mod is not None and Path(getattr(mod, "__file__", "") or "").parent == src:
            del sys.modules[name]
    if str(src) in sys.path:
        sys.path.remove(str(src))
    if src.parent.name.startswith("lrmlp_legacy_"):
        shutil.rmtree(src.parent, ignore_errors=True)


def load_legacy(src: Path) -> SimpleNamespace:
    """Import the legacy modules from ``src`` (a dir from ``extract_legacy``)."""
    if os.environ.get("TF_USE_LEGACY_KERAS") != "1":
        raise RuntimeError(
            "the legacy code needs Keras 2: set TF_USE_LEGACY_KERAS=1 "
            "(and use the golden env .overhaul/env/venv-py312-tf221)"
        )
    src = Path(src).resolve()
    if src.resolve().is_relative_to((REPO_ROOT / "Neural-Network").resolve()):
        raise RuntimeError("refusing to import legacy code from the working tree")
    sys.dont_write_bytecode = True
    os.environ.setdefault("MPLBACKEND", "Agg")
    for name in LEGACY_MODULES:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(src))
    modules = {name: importlib.import_module(name) for name in LEGACY_MODULES}
    for name, mod in modules.items():
        if Path(mod.__file__).resolve().parent != src:
            raise RuntimeError(f"{name} was imported from {mod.__file__}, not from {src}")
    return SimpleNamespace(**modules)


def configure(
    ns: SimpleNamespace,
    *,
    grid_size: float,
    max_local_level: int,
    sort_matrix: bool = False,
    n_max_local: int | None = None,
    n_max_long_range: int | None = None,
    predict_only_energy: bool = True,
    use_long_range: bool = True,
    shuffle_buffer: int = 1000,
) -> dict[str, Any]:
    """Patch the legacy constants on class level and return the effective values.

    Derived values use the exact legacy expressions (``Constants.py:34-36``)::

        CUT_OFF = (MAX_LOCAL_LEVEL + 1 + 0.5**0.5) * GRID_SIZE
        INNER_CUT_OFF = CUT_OFF - 0.2
    """
    D = ns.Constants.Dataset
    M = ns.Constants.Model
    D.GRID_SIZE = grid_size
    D.MAX_LOCAL_LEVEL = max_local_level
    D.CUT_OFF = (D.MAX_LOCAL_LEVEL + 1 + 0.5**0.5) * D.GRID_SIZE
    D.INNER_CUT_OFF = D.CUT_OFF - INNER_CUT_OFF_OFFSET
    D.MAX_ATOM_ELEMENTS = MAX_ATOM_ELEMENTS
    D.use_charge_number = USE_CHARGE_NUMBER
    D.sort_matrix = sort_matrix
    D.SHUFFLE_BUFFER_SIZE = shuffle_buffer
    if n_max_local is not None:
        M.n_max_local = n_max_local
    if n_max_long_range is not None:
        M.n_max_long_range = n_max_long_range
    M.input_shape_local_matrix = (M.n_max_local, 4)
    M.input_shape_atomic_numbers = (M.n_max_local, 2)
    M.input_shape_long_range_matrix = (M.n_max_long_range, 4)
    M.input_shape_long_range_atomic_features = (M.n_max_long_range, D.MAX_ATOM_ELEMENTS + 1)
    M.predict_only_energy = predict_only_energy
    M.use_long_range = use_long_range
    return {
        "GRID_SIZE": D.GRID_SIZE,
        "MAX_LOCAL_LEVEL": D.MAX_LOCAL_LEVEL,
        "CUT_OFF": D.CUT_OFF,
        "INNER_CUT_OFF": D.INNER_CUT_OFF,
        "MAX_ATOM_ELEMENTS": D.MAX_ATOM_ELEMENTS,
        "use_charge_number": D.use_charge_number,
        "sort_matrix": D.sort_matrix,
        "SHUFFLE_BUFFER_SIZE": D.SHUFFLE_BUFFER_SIZE,
        "n_max_local": M.n_max_local,
        "n_max_long_range": M.n_max_long_range,
        "predict_only_energy": M.predict_only_energy,
        "use_long_range": M.use_long_range,
    }


def legacy_atom_order(grid: list, positions: np.ndarray) -> np.ndarray:
    """Input index of each legacy row, by exact position equality.

    ``create_matrices`` iterates ``for cell in grid: for atom in cell.positions``,
    so its rows are in that order. Positions must be unique.
    """
    legacy = np.asarray([p for cell in grid for p in cell.positions], dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    if legacy.shape != positions.shape:
        raise ValueError(f"grid holds {legacy.shape} positions, input has {positions.shape}")
    eq = np.all(legacy[:, None, :] == positions[None, :, :], axis=-1)
    hits = eq.sum(axis=1)
    if not np.all(hits == 1):
        raise ValueError("positions are not unique; legacy order is ambiguous")
    order = eq.argmax(axis=1).astype(np.int64)
    if not np.array_equal(np.sort(order), np.arange(len(order))):
        raise ValueError("legacy order is not a permutation")
    return order
