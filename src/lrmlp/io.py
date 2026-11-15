"""Reading raw trajectories in the sGDML / MD17 ``.npz`` layout.

This module replaces the raw-data part of ``create_dataset`` in the legacy
``Neural-Network/Dataset.py`` (commit ``9413bfd``, ``Dataset.py:413-422``).

File layout
-----------
An ``.npz`` archive with the keys of the sGDML datasets [16] (thesis Sec. 4.3.1):

``E``
    Energies in kcal/mol, shape (S,) or (S, 1); always returned flat.
``F``
    Forces in kcal/(mol·Å), shape (S, N, 3).
``R``
    Positions in Å, shape (S, N, 3).
``z``
    Atomic numbers, shape (N,), the same for every frame.

Further keys (``name``, ``theory``, ``md5``, ...) are ignored. The archive is
opened once, inside a context manager, and every array is read exactly once.
Pickled objects are never loaded.

The module depends on NumPy only.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

__all__ = ["REQUIRED_KEYS", "Trajectory", "load_trajectory"]

#: Keys an input archive must contain.
REQUIRED_KEYS = ("E", "F", "R", "z")


@dataclass(frozen=True, slots=True, eq=False)
class Trajectory:
    """Frames of one molecule read from an sGDML / MD17 ``.npz`` file.

    Attributes
    ----------
    energies : ndarray of float64, shape (S,)
        Total energies in kcal/mol.
    forces : ndarray of float64, shape (S, N, 3)
        Forces in kcal/(mol·Å), atoms in input order.
    positions : ndarray of float64, shape (S, N, 3)
        Positions in Å, atoms in input order.
    numbers : ndarray of int64, shape (N,)
        Atomic numbers.
    n_available : int
        Number of frames in the file (S ≤ ``n_available``).
    path : Path or None
        File the frames were read from.
    """

    energies: npt.NDArray[np.float64]
    forces: npt.NDArray[np.float64]
    positions: npt.NDArray[np.float64]
    numbers: npt.NDArray[np.int64]
    n_available: int
    path: Path | None = None

    @property
    def n_frames(self) -> int:
        """Number S of frames."""
        return int(self.energies.shape[0])

    @property
    def n_atoms(self) -> int:
        """Number N of atoms per frame."""
        return int(self.numbers.shape[0])


def _leading(array: np.ndarray, size: int) -> np.ndarray:
    """First ``size`` frames of ``array`` as a contiguous float64 copy."""
    return np.ascontiguousarray(array[:size], dtype=np.float64)


def load_trajectory(path: str | Path, n_samples: int | None = None) -> Trajectory:
    """Read energies, forces, positions and atomic numbers from an ``.npz`` file.

    Parameters
    ----------
    path : str or Path
        sGDML / MD17 archive with the keys ``E``, ``F``, ``R`` and ``z`` (see the
        module docstring).
    n_samples : int or None
        Read only the first ``n_samples`` frames. A file with fewer frames gives
        all of them (the legacy code slices ``[:n]``). ``None`` reads every frame.

    Returns
    -------
    Trajectory
        The frames; ``E`` of shape (S, 1) is flattened to (S,).

    Raises
    ------
    ValueError
        If keys are missing (all missing keys are listed), shapes are
        inconsistent, the atomic numbers are not positive integers or
        ``n_samples`` is not a positive integer.
    FileNotFoundError
        If ``path`` does not exist.
    """
    path = Path(path)
    if n_samples is not None:
        if isinstance(n_samples, bool):
            raise ValueError(f"n_samples must be a positive integer or None, got {n_samples!r}")
        n_samples = operator.index(n_samples)
        if n_samples <= 0:
            raise ValueError(f"n_samples must be a positive integer or None, got {n_samples!r}")

    with np.load(path, allow_pickle=False) as data:
        missing = [key for key in REQUIRED_KEYS if key not in data.files]
        if missing:
            raise ValueError(
                f"{path}: missing key(s) {missing}; an sGDML/MD17 file needs {list(REQUIRED_KEYS)}"
            )
        raw = {key: data[key] for key in REQUIRED_KEYS}  # every array read exactly once

    energies, forces, positions, z = raw["E"], raw["F"], raw["R"], raw["z"]
    if energies.ndim == 2 and energies.shape[1] == 1:
        energies = energies[:, 0]
    if energies.ndim != 1:
        raise ValueError(f"{path}: E must have shape (S,) or (S, 1), got {raw['E'].shape}")
    if z.ndim != 1:
        raise ValueError(f"{path}: z must have shape (N,), got {z.shape}")
    n_frames, n_atoms = energies.shape[0], z.shape[0]
    for key, array in (("F", forces), ("R", positions)):
        if array.shape != (n_frames, n_atoms, 3):
            raise ValueError(
                f"{path}: {key} must have shape (S, N, 3) = ({n_frames}, {n_atoms}, 3), "
                f"got {array.shape}"
            )
    if not (np.issubdtype(z.dtype, np.integer) or np.all(np.mod(z, 1) == 0)):
        raise ValueError(f"{path}: z must hold integer atomic numbers")
    numbers = z.astype(np.int64)
    if np.any(numbers < 1):
        raise ValueError(f"{path}: atomic numbers must be >= 1")

    size = n_frames if n_samples is None else min(n_samples, n_frames)
    return Trajectory(
        energies=_leading(energies, size),
        forces=_leading(forces, size),
        positions=_leading(positions, size),
        numbers=numbers,
        n_available=int(n_frames),
        path=path,
    )
