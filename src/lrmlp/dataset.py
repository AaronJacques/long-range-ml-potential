"""Preprocessed descriptor datasets: building, storage format and tf.data pipelines.

This module replaces ``create_dataset``, ``create_tf_dataset``, ``get_generator``
and ``get_tf_dataset_from_generator`` of the legacy ``Neural-Network/Dataset.py``
(commit ``9413bfd``, ``Dataset.py:407-661``). pandas is not used.

Building (:func:`build_dataset`)
--------------------------------
1. The first ``data.n_samples_per_source`` frames of every source are read
   (:func:`lrmlp.io.load_trajectory`, ``"all"`` = every frame) and concatenated
   in source order. The global frame index of frame f of source s is
   ``offset_s + f`` with ``offset_s`` the number of frames taken from the
   sources before s (the legacy ``pd.concat(..., ignore_index=True)``).
2. The descriptors of every frame (:func:`lrmlp.descriptors.compute_descriptors`)
   are computed in worker processes, in chunks of consecutive frames; the
   result does not depend on the number of workers. The workers are started
   with ``"spawn"``, so scripts calling :func:`build_dataset` with more than one
   worker need an ``if __name__ == "__main__":`` guard.
3. N_max^local and N_max^long are the maxima over **all** frames, before the
   split (legacy behaviour, p. 44).
4. The frames are split with :func:`split_indices` and every split is written
   in the order described below.

Split (:func:`split_indices`)
-----------------------------
Exactly the pandas expressions of the legacy code (``Dataset.py:475-477``)::

    train = df.sample(frac=1 - (v + t), random_state=seed)
    val   = df.drop(train.index).sample(frac=v / (v + t), random_state=seed)
    test  = df.drop(train.index).drop(val.index)

``df.sample(frac=f, random_state=seed)`` on n rows takes
``size = round(f * n)`` (Python ``round``, half to even) rows at the positions
``numpy.random.RandomState(seed).permutation(n)[:size]``. The validation sample is
drawn from the remaining frames in ascending order, with a fresh generator.
``train`` and ``val`` keep the sampling order, ``test`` is ascending. For
v + t = 0 validation and test are empty (the legacy code divides by zero).

Storage format (``format_version`` 1)
-------------------------------------
A directory with ``metadata.json`` and one subdirectory per split
(``train/``, ``val/``, ``test/``) holding uncompressed ``.npy`` files that are
memory-mapped on loading. With S molecules, A atoms in total (molecules
concatenated in split order), N_loc = ``n_max_local``, N_lr =
``n_max_long_range`` and n_sp = number of species:

======================== ================== ========= ================================
file                     shape              dtype     content
======================== ================== ========= ================================
``energies.npy``         (S,)               float64   total energy, kcal/mol
``atom_offsets.npy``     (S + 1,)           int64     atoms of molecule k:
                                                      ``atom_offsets[k]:atom_offsets[k+1]``
``sample_index.npy``     (S,)               int64     global frame index (see above)
``atomic_numbers.npy``   (A,)               uint8     z_i
``forces.npy``           (A, 3)             float32   force on atom i, kcal/(mol·Å)
``local_matrix.npy``     (A, N_loc, 4)      float32   R̃_local (Eq. 4.4-4.6)
``local_numbers.npy``    (A, N_loc, 2)      uint8     Z_i (Eq. 4.15)
``long_range_matrix.npy`` (A, N_lr, 4)      float32   R̃_long (Eq. 4.16-4.18)
``long_range_counts.npy`` (A, N_lr, n_sp)   uint8     f_Z of the species ``species``
======================== ================== ========= ================================

Atoms are **always in input order**: row a of molecule k and ``forces[a]``
belong to atom a of the frame (fixes B01). The rows of an atom follow
:func:`lrmlp.descriptors.compute_descriptors`; padding rows are zero.

The 101-column F_i (Eq. 4.19-4.20) is not stored: for Stachyose it would take
≈ 27 272·87·28·101·4 B ≈ 26.8 GB. It is rebuilt from ``atomic_numbers`` and
``long_range_counts``: column 0 is z_i on every real row, column Z is the count
of element Z (``species[s]`` for count column s), all other columns are zero.
A row is real if its counts are not all zero; every long-range cell holds at
least one atom, so this is exact (testing ``long_range_matrix[..., 0] > 0``
would miss a cell whose centre of mass coincides with the atom, which the
legacy code encodes as ``[0, 0, 0, 0]``). The result equals the
``long_range_features`` of :func:`lrmlp.descriptors.pad_descriptors`.

``metadata.json`` holds ``format_version``, ``lrmlp_version``, ``created``
(UTC, ISO 8601), ``name``, ``sources`` (path, sha256, ``n_frames_available``,
``n_frames_used``, ``frame_offset``, ``n_atoms``), ``n_samples_per_source``, the
complete ``grid`` configuration with ``cutoff`` and ``inner_cutoff``,
``max_atomic_number``, ``n_max_local``, ``n_max_long_range``, ``species``
(sorted atomic numbers), ``splits`` (molecules and atoms per split),
``val_fraction``, ``test_fraction``, ``split_seed``, ``units`` and ``arrays``
(dtype per file).

tf.data (:func:`make_tf_dataset`, :func:`make_batched_tf_dataset`)
------------------------------------------------------------------
Elements are read from the memory-mapped arrays by molecule index; nothing is
embedded as a graph constant except the (n_sp × max_atomic_number) expansion
matrix of F_i. Shuffling permutes molecule indices (seeded,
``reshuffle_each_iteration=True``); evaluation pipelines are never shuffled
(B19). The training pipeline yields one molecule per element (|B| = 1, p. 51).
TensorFlow is imported only by these two functions.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import logging
import math
import operator
import os
import secrets
import shutil
import sys
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from multiprocessing import get_context
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from lrmlp import __version__
from lrmlp.config import ExperimentConfig, GridConfig
from lrmlp.descriptors import compute_descriptors
from lrmlp.io import load_trajectory

if TYPE_CHECKING:
    import tensorflow as tf

__all__ = [
    "FORMAT_VERSION",
    "SPLITS",
    "DescriptorDataset",
    "DescriptorSplit",
    "build_dataset",
    "load_dataset",
    "make_batched_tf_dataset",
    "make_tf_dataset",
    "split_indices",
]

logger = logging.getLogger(__name__)

#: Version of the on-disk layout described in the module docstring.
FORMAT_VERSION = 1
#: Split names in file order.
SPLITS = ("train", "val", "test")
#: Name of the metadata file.
METADATA_FILE = "metadata.json"
#: Physical units of the stored quantities.
UNITS = {"energy": "kcal/mol", "forces": "kcal/(mol*Angstrom)", "positions": "Angstrom"}
#: dtype of every stored array.
ARRAY_DTYPES: dict[str, np.dtype[Any]] = {
    "energies": np.dtype(np.float64),
    "atom_offsets": np.dtype(np.int64),
    "sample_index": np.dtype(np.int64),
    "atomic_numbers": np.dtype(np.uint8),
    "forces": np.dtype(np.float32),
    "local_matrix": np.dtype(np.float32),
    "local_numbers": np.dtype(np.uint8),
    "long_range_matrix": np.dtype(np.float32),
    "long_range_counts": np.dtype(np.uint8),
}
#: Model input names (``lrmlp.model.build_model``).
INPUT_KEYS = ("local_matrix", "local_numbers", "long_range_matrix", "long_range_features")

_UINT8_MAX = int(np.iinfo(np.uint8).max)
#: Molecules written per block in the second build pass (bounds the temporary memory).
_WRITE_BLOCK = 1024
#: Upper bound on the frames per worker task.
_MAX_CHUNK = 256

IntArray = npt.NDArray[np.int64]


# --------------------------------------------------------------------------- split


def _sample_positions(n: int, frac: float, seed: int) -> IntArray:
    """Positions drawn by ``df.sample(frac=frac, random_state=seed)`` on ``n`` rows."""
    size = round(frac * n)
    return np.random.RandomState(seed).permutation(n)[:size].astype(np.int64)


def split_indices(
    n: int, val_fraction: float, test_fraction: float, seed: int
) -> tuple[IntArray, IntArray, IntArray]:
    """Split ``n`` frames like the legacy pandas code, without pandas.

    Parameters
    ----------
    n : int
        Number of frames (indices ``0 … n-1``).
    val_fraction, test_fraction : float
        Fractions v and t of the validation and test split (v, t ≥ 0, v + t < 1).
    seed : int
        ``random_state`` of both ``df.sample`` calls.

    Returns
    -------
    train, val, test : ndarray of int64
        Frame indices; ``train`` and ``val`` in sampling order, ``test``
        ascending. The rule is described in the module docstring.

    Raises
    ------
    ValueError
        If ``n`` is negative or the fractions are invalid.
    """
    n = operator.index(n)
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        raise ValueError(
            "fractions must satisfy v, t >= 0 and v + t < 1, got "
            f"v = {val_fraction}, t = {test_fraction}"
        )
    total = val_fraction + test_fraction
    train = _sample_positions(n, 1 - (val_fraction + test_fraction), seed)
    taken = np.zeros(n, dtype=bool)
    taken[train] = True
    rest = np.flatnonzero(~taken).astype(np.int64)
    if total == 0:
        val = np.zeros(0, dtype=np.int64)
    else:
        val = rest[
            _sample_positions(rest.shape[0], val_fraction / (val_fraction + test_fraction), seed)
        ]
    taken[val] = True
    test = np.flatnonzero(~taken).astype(np.int64)
    return train, val, test


# --------------------------------------------------------------------------- workers


@dataclass(frozen=True, slots=True)
class _ChunkTask:
    positions: npt.NDArray[np.float64]  # (c, N, 3)
    numbers: IntArray  # (N,)
    grid: GridConfig
    species: IntArray  # (n_species,)
    source: str | None  # trajectory path for error messages; None: "frame f: ..." only
    first_frame: int  # frame index of positions[0] within the source
    frame_offset: int  # global frame index of frame 0 of the source


@dataclass(frozen=True, slots=True)
class _Ragged:
    """Compact ragged descriptors of consecutive atoms (one or more frames)."""

    local_matrix: npt.NDArray[np.float32]  # (K, 4)
    local_numbers: npt.NDArray[np.uint8]  # (K, 2)
    n_local: IntArray  # (atoms,)
    long_range_matrix: npt.NDArray[np.float32]  # (M, 4)
    long_range_counts: npt.NDArray[np.uint8]  # (M, n_species)
    n_long: IntArray  # (atoms,)

    @classmethod
    def concatenate(cls, parts: list[_Ragged], n_species: int) -> _Ragged:
        """Concatenate ``parts`` field by field; **empties** ``parts``.

        Every field is copied into one preallocated array and the source arrays
        are released as soon as they are copied, so the ragged data is not held
        twice (the result equals :func:`numpy.concatenate` of the parts).
        """
        if not parts:
            return cls(
                np.zeros((0, 4), np.float32),
                np.zeros((0, 2), np.uint8),
                np.zeros(0, np.int64),
                np.zeros((0, 4), np.float32),
                np.zeros((0, n_species), np.uint8),
                np.zeros(0, np.int64),
            )
        names = [f.name for f in dataclasses.fields(cls)]
        columns: dict[str, list[np.ndarray | None]] = {
            name: [getattr(p, name) for p in parts] for name in names
        }
        parts.clear()
        merged = {}
        for name in names:
            column = columns.pop(name)
            first = column[0]
            assert first is not None
            total = sum(a.shape[0] for a in column if a is not None)
            out = np.empty((total, *first.shape[1:]), dtype=first.dtype)
            del first
            lo = 0
            for k, array in enumerate(column):
                assert array is not None
                out[lo : lo + array.shape[0]] = array
                lo += array.shape[0]
                column[k] = None
                del array
            merged[name] = out
        return cls(**merged)


def _chunk_descriptors(task: _ChunkTask) -> _Ragged:
    """Descriptors of the frames of one task (runs in a worker process).

    A :class:`ValueError` of a frame is re-raised with the source path and the
    frame index (within the source and global); without a source (``source=None``,
    :meth:`lrmlp.potential.Potential.predict`) as ``frame f: ...``, the message of
    the serial path.
    """
    parts = []
    for k, positions in enumerate(task.positions):
        try:
            desc = compute_descriptors(positions, task.numbers, task.grid, species=task.species)
            counts = desc.long_range_counts
            if counts.size and counts.max() > _UINT8_MAX:
                raise ValueError(
                    f"a long-range cell holds more than {_UINT8_MAX} atoms of one element"
                )
        except ValueError as exc:
            frame = task.first_frame + k
            if task.source is None:
                raise ValueError(f"frame {frame}: {exc}") from exc
            raise ValueError(
                f"{task.source}: frame {frame} (global frame {task.frame_offset + frame}): {exc}"
            ) from exc
        parts.append(
            _Ragged(
                local_matrix=desc.local_matrix.astype(np.float32),
                local_numbers=desc.local_numbers.astype(np.uint8),
                n_local=desc.n_local,
                long_range_matrix=desc.long_range_matrix.astype(np.float32),
                long_range_counts=counts.astype(np.uint8),
                n_long=desc.n_long,
            )
        )
    return _Ragged.concatenate(parts, task.species.shape[0])


def _resolve_workers(n_workers: int | None, default: int) -> int:
    n = default if n_workers is None else operator.index(n_workers)
    if n < 0:
        raise ValueError(f"n_workers must be non-negative, got {n}")
    return n if n > 0 else (os.cpu_count() or 1)


@contextlib.contextmanager
def _importable_main() -> Iterator[None]:
    """Hide a ``__main__.__file__`` that is not a file (``python - < script``) from "spawn".

    Spawned workers re-run the parent's main script from ``__main__.__file__``; for
    a script read from stdin that is ``"<stdin>"`` and the workers die. The workers
    only need :mod:`lrmlp.dataset`, so the attribute is removed while they start.
    """
    main = sys.modules.get("__main__")
    path = getattr(main, "__file__", None)
    if main is None or path is None or getattr(main, "__spec__", None) is not None:
        yield
        return
    if os.path.isfile(path):
        yield
        return
    del main.__file__
    try:
        yield
    finally:
        main.__file__ = path


#: Thread-pool variables of the BLAS/OpenMP libraries that NumPy may load; set to 1
#: in the descriptor worker processes.
WORKER_THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


class _ChildThreadEnv:
    """Reference-counted, thread-safe override of :data:`WORKER_THREAD_ENV`.

    Several threads may start worker pools at the same time (e.g. two
    ``build_dataset`` or ``Potential.predict(n_workers > 1)`` calls). The first
    entry saves the parent's values and sets the variables to ``"1"``; later
    entries only count. The parent's values are restored when the last block
    exits, so an overlapping block never saves the ``"1"`` of another one as the
    "previous" value.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._users = 0
        self._saved: dict[str, str | None] = {}

    def enter(self) -> None:
        with self._lock:
            if self._users == 0:
                self._saved = {name: os.environ.get(name) for name in WORKER_THREAD_ENV}
                os.environ.update(dict.fromkeys(WORKER_THREAD_ENV, "1"))
            self._users += 1

    def exit(self) -> None:
        with self._lock:
            self._users -= 1
            if self._users > 0:
                return
            for name, value in self._saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
            self._saved = {}


_CHILD_THREAD_ENV = _ChildThreadEnv()


@contextlib.contextmanager
def _single_threaded_children() -> Iterator[None]:
    """Let child processes started in this block use one BLAS/OpenMP thread each.

    BLAS libraries read their thread count once, when NumPy loads them. A spawned
    worker imports NumPy before any pool initializer runs (it re-imports the
    parent's main module first), so the variables of :data:`WORKER_THREAD_ENV` are
    set in ``os.environ`` while the workers are started and inherited by them. The
    parent's already loaded libraries are not affected, and the previous values
    are restored when the last of several overlapping blocks (other threads)
    exits. Without this, every worker starts a thread pool with one thread per CPU
    and N workers oversubscribe the CPU (``docs/benchmarks.md``).
    """
    _CHILD_THREAD_ENV.enter()
    try:
        yield
    finally:
        _CHILD_THREAD_ENV.exit()


def _worker_map[T, R](fn: Callable[[T], R], items: Sequence[T], n_workers: int) -> list[R]:
    """``[fn(x) for x in items]`` in a "spawn" process pool of ``n_workers`` workers.

    The workers use one BLAS/OpenMP thread each (:func:`_single_threaded_children`).
    """
    # "spawn" is safe even if the parent has already imported TensorFlow.
    with (
        _importable_main(),
        ProcessPoolExecutor(max_workers=n_workers, mp_context=get_context("spawn")) as pool,
    ):
        # The executor starts its workers while the tasks are submitted, i.e. inside
        # Executor.map, which submits every task before it returns.
        with _single_threaded_children():
            results = pool.map(fn, items)
        return list(results)


def _run_tasks(tasks: list[_ChunkTask], n_workers: int) -> list[_Ragged]:
    """Run the tasks in order, in-process for one worker, else in a process pool.

    The workers use one BLAS/OpenMP thread each (:func:`_single_threaded_children`);
    the serial path runs in the calling process with its thread settings.
    """
    n_workers = min(n_workers, len(tasks))
    if n_workers <= 1:
        return [_chunk_descriptors(t) for t in tasks]
    return _worker_map(_chunk_descriptors, tasks, n_workers)


def _make_build_dir(target: Path) -> Path:
    """Create an empty, uniquely named sibling of ``target`` for building.

    Unlike :func:`tempfile.mkdtemp` (always ``0o700``) the directory is created
    with mode ``0o777``, so the operating system applies the current umask and
    the renamed dataset directory gets the permissions of a plain ``mkdir``.
    """
    while True:
        tmp = target.with_name(f".{target.name}.building-{secrets.token_hex(8)}")
        try:
            tmp.mkdir(mode=0o777)
        except FileExistsError:
            continue
        return tmp


# --------------------------------------------------------------------------- building helpers


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _offsets(counts: npt.NDArray[np.integer]) -> IntArray:
    return np.concatenate([[0], np.cumsum(counts, dtype=np.int64)]).astype(np.int64)


def _ranges(starts: IntArray, counts: IntArray) -> tuple[IntArray, IntArray, IntArray]:
    """Expand ``[starts[i], starts[i] + counts[i])`` to (owner i, slot, source index)."""
    owner = np.repeat(np.arange(counts.shape[0], dtype=np.int64), counts)
    slot = np.arange(owner.shape[0], dtype=np.int64) - np.repeat(_offsets(counts)[:-1], counts)
    return owner, slot, np.repeat(starts, counts) + slot


def _create_array(path: Path, dtype: np.dtype[Any], shape: tuple[int, ...]) -> np.ndarray:
    """Writable ``.npy`` file (memory-mapped unless empty, which cannot be mapped)."""
    if math.prod(shape) == 0:
        array = np.zeros(shape, dtype=dtype)
        np.save(path, array)
        return array
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def _write_split(
    directory: Path,
    frames: IntArray,
    *,
    energies: npt.NDArray[np.float64],
    frame_atoms: IntArray,
    atom_numbers: npt.NDArray[np.uint8],
    atom_forces: npt.NDArray[np.float32],
    ragged: _Ragged,
    n_max_local: int,
    n_max_long_range: int,
) -> dict[str, int]:
    """Write the molecules ``frames`` (global frame indices, stored order) to ``directory``."""
    directory.mkdir()
    frame_offsets = _offsets(frame_atoms)
    local_offsets = _offsets(ragged.n_local)
    long_offsets = _offsets(ragged.n_long)
    counts = frame_atoms[frames]
    atom_offsets = _offsets(counts)
    n_atoms = int(atom_offsets[-1])
    n_species = ragged.long_range_counts.shape[1]

    np.save(directory / "energies.npy", energies[frames].astype(np.float64))
    np.save(directory / "atom_offsets.npy", atom_offsets)
    np.save(directory / "sample_index.npy", frames.astype(np.int64))
    _, _, atoms = _ranges(frame_offsets[frames], counts)
    np.save(directory / "atomic_numbers.npy", atom_numbers[atoms])
    np.save(directory / "forces.npy", atom_forces[atoms])

    out = {
        "local_matrix": _create_array(
            directory / "local_matrix.npy", ARRAY_DTYPES["local_matrix"], (n_atoms, n_max_local, 4)
        ),
        "local_numbers": _create_array(
            directory / "local_numbers.npy",
            ARRAY_DTYPES["local_numbers"],
            (n_atoms, n_max_local, 2),
        ),
        "long_range_matrix": _create_array(
            directory / "long_range_matrix.npy",
            ARRAY_DTYPES["long_range_matrix"],
            (n_atoms, n_max_long_range, 4),
        ),
        "long_range_counts": _create_array(
            directory / "long_range_counts.npy",
            ARRAY_DTYPES["long_range_counts"],
            (n_atoms, n_max_long_range, n_species),
        ),
    }
    for lo in range(0, frames.shape[0], _WRITE_BLOCK):
        hi = min(lo + _WRITE_BLOCK, frames.shape[0])
        a_lo, a_hi = int(atom_offsets[lo]), int(atom_offsets[hi])
        if a_hi == a_lo:
            continue
        block_atoms = atoms[a_lo:a_hi]
        owner, slot, src = _ranges(local_offsets[block_atoms], ragged.n_local[block_atoms])
        local = np.zeros((a_hi - a_lo, n_max_local, 4), dtype=np.float32)
        local_z = np.zeros((a_hi - a_lo, n_max_local, 2), dtype=np.uint8)
        local[owner, slot] = ragged.local_matrix[src]
        local_z[owner, slot] = ragged.local_numbers[src]
        owner, slot, src = _ranges(long_offsets[block_atoms], ragged.n_long[block_atoms])
        long_range = np.zeros((a_hi - a_lo, n_max_long_range, 4), dtype=np.float32)
        long_counts = np.zeros((a_hi - a_lo, n_max_long_range, n_species), dtype=np.uint8)
        long_range[owner, slot] = ragged.long_range_matrix[src]
        long_counts[owner, slot] = ragged.long_range_counts[src]
        out["local_matrix"][a_lo:a_hi] = local
        out["local_numbers"][a_lo:a_hi] = local_z
        out["long_range_matrix"][a_lo:a_hi] = long_range
        out["long_range_counts"][a_lo:a_hi] = long_counts
    for array in out.values():
        if isinstance(array, np.memmap):
            array.flush()
    del out
    return {"n_molecules": int(frames.shape[0]), "n_atoms": n_atoms}


def _check_target(target: Path, overwrite: bool) -> None:
    if not target.exists():
        return
    if not overwrite:
        raise FileExistsError(f"dataset directory {target} already exists (use overwrite=True)")
    if not target.is_dir() or not ((target / METADATA_FILE).is_file() or not any(target.iterdir())):
        raise FileExistsError(
            f"{target} exists and is not a dataset directory; refusing to overwrite it"
        )


# --------------------------------------------------------------------------- build


def build_dataset(
    config: ExperimentConfig,
    output_dir: Path | None = None,
    n_workers: int | None = None,
    *,
    overwrite: bool = False,
) -> Path:
    """Compute the descriptors of all frames, split them and write the dataset.

    Parameters
    ----------
    config : ExperimentConfig
        ``config.data`` selects sources, frames and split, ``config.grid`` the
        descriptors.
    output_dir : Path or None
        Target directory; default ``config.data.output_dir``.
    n_workers : int or None
        Worker processes; default ``config.data.n_workers``, 0 = all CPUs, 1 =
        no worker processes. The result does not depend on it. More than one
        worker is started with the ``"spawn"`` method on every platform, so a
        script that calls this function must guard the call with
        ``if __name__ == "__main__":`` (otherwise every worker re-runs the script).
    overwrite : bool
        Replace an existing dataset directory. Without it an existing target is an
        error (legacy behaviour).

    Returns
    -------
    Path
        The dataset directory (layout in the module docstring).

    Raises
    ------
    FileExistsError
        If the target exists and ``overwrite`` is false, or it exists but is
        neither a dataset directory nor empty.
    ValueError
        On invalid input data (see :func:`lrmlp.io.load_trajectory` and
        :func:`lrmlp.descriptors.compute_descriptors`); errors of a frame name the
        source and the frame index.

    Notes
    -----
    The dataset is written to a temporary sibling directory and renamed at the
    end, so an interrupted build leaves no partial dataset behind. The ragged
    descriptors of all frames are held in memory between the two passes (compact
    float32 / uint8, roughly the size of the written files); they are merged
    without holding a second copy.
    """
    data, grid = config.data, config.grid
    target = Path(output_dir if output_dir is not None else data.output_dir)  # type: ignore[arg-type]
    _check_target(target, overwrite)
    workers = _resolve_workers(n_workers, data.n_workers)
    n_samples = None if data.n_samples_per_source == "all" else int(data.n_samples_per_source)

    trajectories = [load_trajectory(src, n_samples) for src in data.sources]
    species = np.unique(np.concatenate([t.numbers for t in trajectories])).astype(np.int64)
    if species.size and species.max() > min(grid.max_atomic_number, _UINT8_MAX):
        raise ValueError(
            f"atomic number {int(species.max())} exceeds grid.max_atomic_number = "
            f"{grid.max_atomic_number} (or {_UINT8_MAX})"
        )

    tasks: list[_ChunkTask] = []
    total_frames = sum(t.n_frames for t in trajectories)
    chunk = max(1, min(_MAX_CHUNK, math.ceil(total_frames / (4 * workers))))
    offset = 0
    for src, t in zip(data.sources, trajectories, strict=True):
        for lo in range(0, t.n_frames, chunk):
            tasks.append(
                _ChunkTask(
                    t.positions[lo : lo + chunk], t.numbers, grid, species, str(src), lo, offset
                )
            )
        offset += t.n_frames
    ragged = _Ragged.concatenate(_run_tasks(tasks, workers), species.shape[0])
    n_max_local = int(ragged.n_local.max()) if ragged.n_local.size else 0
    n_max_long_range = int(ragged.n_long.max()) if ragged.n_long.size else 0

    energies = np.concatenate([t.energies for t in trajectories]).astype(np.float64)
    frame_atoms = np.concatenate(
        [np.full(t.n_frames, t.n_atoms, dtype=np.int64) for t in trajectories]
    )
    atom_numbers = np.concatenate([np.tile(t.numbers, t.n_frames) for t in trajectories]).astype(
        np.uint8
    )
    atom_forces = np.concatenate([t.forces.reshape(-1, 3) for t in trajectories]).astype(np.float32)
    split = dict(
        zip(
            SPLITS,
            split_indices(total_frames, data.val_fraction, data.test_fraction, data.split_seed),
            strict=True,
        )
    )

    sources = []
    offset = 0
    for src, t in zip(data.sources, trajectories, strict=True):
        sources.append(
            {
                "path": str(src),
                "sha256": _sha256(Path(src)),
                "n_frames_available": t.n_available,
                "n_frames_used": t.n_frames,
                "frame_offset": offset,
                "n_atoms": t.n_atoms,
            }
        )
        offset += t.n_frames

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = _make_build_dir(target)
    try:
        split_info = {
            name: _write_split(
                tmp / name,
                split[name],
                energies=energies,
                frame_atoms=frame_atoms,
                atom_numbers=atom_numbers,
                atom_forces=atom_forces,
                ragged=ragged,
                n_max_local=n_max_local,
                n_max_long_range=n_max_long_range,
            )
            for name in SPLITS
        }
        grid_dict = config.to_dict()["grid"]
        metadata = {
            "format_version": FORMAT_VERSION,
            "lrmlp_version": __version__,
            "created": datetime.now(UTC).isoformat(timespec="seconds"),
            "name": config.name,
            "sources": sources,
            "n_samples_per_source": data.n_samples_per_source,
            "grid": {**grid_dict, "cutoff": grid.cutoff, "inner_cutoff": grid.inner_cutoff},
            "max_atomic_number": grid.max_atomic_number,
            "n_max_local": n_max_local,
            "n_max_long_range": n_max_long_range,
            "species": [int(z) for z in species],
            "n_samples": total_frames,
            "splits": split_info,
            "val_fraction": data.val_fraction,
            "test_fraction": data.test_fraction,
            "split_seed": data.split_seed,
            "units": UNITS,
            "arrays": {name: dtype.name for name, dtype in ARRAY_DTYPES.items()},
        }
        (tmp / METADATA_FILE).write_text(json.dumps(metadata, indent=2) + "\n")
        _check_target(target, overwrite)
        if target.exists():
            shutil.rmtree(target)
        os.replace(tmp, target)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    logger.info(
        "built dataset %s: %d frames, n_max = (%d, %d)",
        target,
        total_frames,
        n_max_local,
        n_max_long_range,
    )
    return target


# --------------------------------------------------------------------------- loading


def _load_array(path: Path) -> np.ndarray:
    """Memory-map a ``.npy`` file (empty arrays cannot be mapped and are read)."""
    try:
        return np.load(path, mmap_mode="r", allow_pickle=False)
    except ValueError:  # mmap of a zero-length data segment
        return np.load(path, allow_pickle=False)


def _feature_expansion(species: IntArray, max_atomic_number: int) -> npt.NDArray[np.float32]:
    """(n_species, max_atomic_number) 0/1 matrix mapping count column s to element Z - 1."""
    expansion = np.zeros((species.shape[0], max_atomic_number), dtype=np.float32)
    expansion[np.arange(species.shape[0]), species - 1] = 1
    return expansion


class DescriptorSplit:
    """One split of a preprocessed dataset (read-only, memory-mapped).

    Parameters
    ----------
    path : Path
        Split directory (``train/``, ``val/`` or ``test/``).
    name : str
        Split name.
    species : sequence of int
        Sorted atomic numbers of the count columns.
    max_atomic_number : int
        m_max; F_i has ``max_atomic_number + 1`` columns.

    Attributes
    ----------
    arrays : dict of str to ndarray
        The stored arrays (see the module docstring), memory-mapped.
    """

    def __init__(
        self, path: Path, name: str, species: Sequence[int], max_atomic_number: int
    ) -> None:
        self.path = Path(path)
        self.name = name
        self.species = np.asarray(species, dtype=np.int64)
        self.max_atomic_number = int(max_atomic_number)
        missing = [k for k in ARRAY_DTYPES if not (self.path / f"{k}.npy").is_file()]
        if missing:
            raise ValueError(f"{self.path}: missing array file(s) {missing}")
        self.arrays: dict[str, np.ndarray] = {
            k: _load_array(self.path / f"{k}.npy") for k in ARRAY_DTYPES
        }
        for key, dtype in ARRAY_DTYPES.items():
            if self.arrays[key].dtype != dtype:
                raise ValueError(
                    f"{self.path}/{key}.npy: dtype {self.arrays[key].dtype} != {dtype}"
                )
        n_atoms = int(self.atom_offsets[-1])
        if self.atom_offsets.shape != (len(self) + 1,) or self.arrays["forces"].shape != (
            n_atoms,
            3,
        ):
            raise ValueError(f"{self.path}: inconsistent array shapes")

    def __len__(self) -> int:
        """Return the number of molecules."""
        return int(self.arrays["energies"].shape[0])

    def __repr__(self) -> str:
        """Short description."""
        return f"DescriptorSplit({self.name!r}, molecules={len(self)}, atoms={self.n_atoms})"

    @property
    def atom_offsets(self) -> IntArray:
        """Atom ranges of the molecules, shape (S + 1,)."""
        return self.arrays["atom_offsets"]

    @property
    def n_atoms(self) -> int:
        """Total number A of atoms."""
        return int(self.atom_offsets[-1])

    @property
    def n_max_local(self) -> int:
        """Padded number of local rows N_max^local."""
        return int(self.arrays["local_matrix"].shape[1])

    @property
    def n_max_long_range(self) -> int:
        """Padded number of long-range rows N_max^long."""
        return int(self.arrays["long_range_matrix"].shape[1])

    @property
    def energies(self) -> npt.NDArray[np.float64]:
        """Total energies in kcal/mol, shape (S,)."""
        return self.arrays["energies"]

    @property
    def sample_index(self) -> IntArray:
        """Global frame index of every molecule, shape (S,)."""
        return self.arrays["sample_index"]

    def long_range_features(self, lo: int, hi: int) -> npt.NDArray[np.float32]:
        """Rebuild F_i (Eq. 4.19-4.20) of the atoms ``lo:hi``.

        Returns
        -------
        ndarray of float32, shape (hi - lo, N_max^long, max_atomic_number + 1)
            Column 0 = z_i on real rows, column Z = count of element Z; identical
            to the ``long_range_features`` of :func:`lrmlp.descriptors.pad_descriptors`.
        """
        counts = np.asarray(self.arrays["long_range_counts"][lo:hi])
        z = np.asarray(self.arrays["atomic_numbers"][lo:hi])
        out = np.zeros((*counts.shape[:2], self.max_atomic_number + 1), dtype=np.float32)
        out[..., self.species] = counts
        out[..., 0] = np.where(counts.any(axis=-1), z[:, None], 0)
        return out

    def molecule(self, k: int) -> dict[str, np.ndarray]:
        """Return the padded model inputs and labels of molecule ``k`` (stored order).

        Returns
        -------
        dict of str to ndarray
            ``local_matrix`` (N, N_loc, 4) float32, ``local_numbers`` (N, N_loc, 2)
            uint8, ``long_range_matrix`` (N, N_lr, 4) float32,
            ``long_range_features`` (N, N_lr, max_atomic_number + 1) float32 (as
            :func:`lrmlp.descriptors.pad_descriptors`), ``atomic_numbers`` (N,)
            uint8, ``energy`` () float64 and ``forces`` (N, 3) float32. Atoms are
            in input order.

        Raises
        ------
        IndexError
            If ``k`` is out of range.
        """
        n = len(self)
        k = operator.index(k)
        if not -n <= k < n:
            raise IndexError(f"molecule index {k} out of range for {n} molecules")
        k %= n
        lo, hi = int(self.atom_offsets[k]), int(self.atom_offsets[k + 1])
        a = self.arrays
        return {
            "local_matrix": np.array(a["local_matrix"][lo:hi]),
            "local_numbers": np.array(a["local_numbers"][lo:hi]),
            "long_range_matrix": np.array(a["long_range_matrix"][lo:hi]),
            "long_range_features": self.long_range_features(lo, hi),
            "atomic_numbers": np.array(a["atomic_numbers"][lo:hi]),
            "energy": np.array(a["energies"][k]),
            "forces": np.array(a["forces"][lo:hi]),
        }

    def __iter__(self) -> Iterator[dict[str, np.ndarray]]:
        """Iterate over :meth:`molecule` in stored order."""
        for k in range(len(self)):
            yield self.molecule(k)


class DescriptorDataset:
    """A preprocessed dataset directory (see :func:`load_dataset`).

    Attributes
    ----------
    path : Path
        Dataset directory.
    metadata : dict
        Content of ``metadata.json``.
    splits : dict of str to DescriptorSplit
        ``"train"``, ``"val"`` and ``"test"``.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        meta_path = self.path / METADATA_FILE
        if not meta_path.is_file():
            raise FileNotFoundError(f"{meta_path} not found; not a dataset directory")
        self.metadata: dict[str, Any] = json.loads(meta_path.read_text())
        version = self.metadata.get("format_version")
        if version != FORMAT_VERSION:
            raise ValueError(
                f"{self.path}: format_version {version!r} is not supported "
                f"(expected {FORMAT_VERSION}); rebuild the dataset"
            )
        self.splits: dict[str, DescriptorSplit] = {
            name: DescriptorSplit(
                self.path / name,
                name,
                self.metadata["species"],
                self.metadata["max_atomic_number"],
            )
            for name in SPLITS
        }
        for name, split in self.splits.items():
            if (split.n_max_local, split.n_max_long_range) != (
                self.n_max_local,
                self.n_max_long_range,
            ):
                raise ValueError(f"{self.path}/{name}: padding differs from metadata.json")

    def __getitem__(self, name: str) -> DescriptorSplit:
        """Split ``name``."""
        return self.splits[name]

    def __repr__(self) -> str:
        """Short description."""
        sizes = {name: len(s) for name, s in self.splits.items()}
        return f"DescriptorDataset({str(self.path)!r}, {sizes})"

    @property
    def n_max_local(self) -> int:
        """N_max^local of the dataset."""
        return int(self.metadata["n_max_local"])

    @property
    def n_max_long_range(self) -> int:
        """N_max^long of the dataset."""
        return int(self.metadata["n_max_long_range"])

    @property
    def grid(self) -> GridConfig:
        """Grid configuration the descriptors were built with."""
        fields = {f.name for f in dataclasses.fields(GridConfig)}
        return GridConfig(**{k: v for k, v in self.metadata["grid"].items() if k in fields})


def load_dataset(path: str | Path) -> DescriptorDataset:
    """Open a dataset written by :func:`build_dataset`.

    Parameters
    ----------
    path : str or Path
        Dataset directory.

    Returns
    -------
    DescriptorDataset
        Metadata and memory-mapped splits.

    Raises
    ------
    FileNotFoundError
        If ``metadata.json`` is missing.
    ValueError
        If ``format_version`` is unsupported or arrays are missing or inconsistent.
    """
    return DescriptorDataset(Path(path))


# --------------------------------------------------------------------------- tf.data


def _atom_block_fn(
    split: DescriptorSplit, include_long_range: bool
) -> tuple[Any, list[Any], list[str]]:
    """NumPy reader of the stored arrays of one atom range ``[lo, hi)``."""
    import tensorflow as tf

    keys = ["local_matrix", "local_numbers", "forces", "atomic_numbers"]
    if include_long_range:
        keys += ["long_range_matrix", "long_range_counts"]
    arrays = [split.arrays[k] for k in keys]

    def read(lo: np.ndarray, hi: np.ndarray) -> list[np.ndarray]:
        return [np.asarray(a[int(lo) : int(hi)]) for a in arrays]

    return read, [tf.as_dtype(ARRAY_DTYPES[k]) for k in keys], keys


def _reconstruct(
    split: DescriptorSplit, include_long_range: bool, raw: Mapping[str, Any]
) -> dict[str, Any]:
    """Model inputs from the stored arrays of an atom block (TensorFlow ops)."""
    import tensorflow as tf

    out = {
        "local_matrix": tf.ensure_shape(raw["local_matrix"], (None, split.n_max_local, 4)),
        "local_numbers": tf.cast(
            tf.ensure_shape(raw["local_numbers"], (None, split.n_max_local, 2)), tf.float32
        ),
    }
    if include_long_range:
        n_lr, n_sp = split.n_max_long_range, int(split.species.shape[0])
        counts = tf.cast(tf.ensure_shape(raw["long_range_counts"], (None, n_lr, n_sp)), tf.float32)
        expansion = tf.constant(_feature_expansion(split.species, split.max_atomic_number))
        valid = tf.cast(tf.reduce_any(counts > 0, axis=-1), tf.float32)
        z = tf.cast(tf.ensure_shape(raw["atomic_numbers"], (None,)), tf.float32)
        out["long_range_matrix"] = tf.ensure_shape(raw["long_range_matrix"], (None, n_lr, 4))
        out["long_range_features"] = tf.concat(
            [(z[:, None] * valid)[..., None], tf.tensordot(counts, expansion, axes=1)], axis=-1
        )
    return out


def make_tf_dataset(
    split: DescriptorSplit,
    *,
    shuffle: bool,
    shuffle_buffer: int,
    seed: int | None,
    include_long_range: bool = True,
) -> tf.data.Dataset:
    """Training / validation pipeline with one molecule per element (|B| = 1, p. 51).

    Parameters
    ----------
    split : DescriptorSplit
        Source split.
    shuffle : bool
        Shuffle the molecule order. Use ``False`` for validation and test (B19).
    shuffle_buffer : int
        Shuffle buffer in molecules (legacy: 1000); ≤ 1 keeps the stored order.
    seed : int or None
        Shuffle seed (B11). The order differs from epoch to epoch
        (``reshuffle_each_iteration=True``) but is reproducible for a given seed.
    include_long_range : bool
        Include ``long_range_matrix`` and ``long_range_features`` (set ``False``
        for ``model.use_long_range = False``, Fig. 4.6).

    Returns
    -------
    tf.data.Dataset
        Elements ``(inputs, targets)`` with ``inputs`` = {``local_matrix`` (N,
        N_loc, 4), ``local_numbers`` (N, N_loc, 2), ``long_range_matrix`` (N, N_lr,
        4), ``long_range_features`` (N, N_lr, m_max + 1)} and ``targets`` =
        {``energy`` (1,), ``forces`` (N, 3)}, all float32, N = ``None`` in the
        ``element_spec``. Without shuffling the order is the stored order.
    """
    import tensorflow as tf

    offsets = tf.constant(np.asarray(split.atom_offsets, dtype=np.int64))
    energies = tf.constant(np.asarray(split.energies, dtype=np.float32))
    read, dtypes, keys = _atom_block_fn(split, include_long_range)

    def load(k: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        lo, hi = offsets[k], offsets[k + 1]
        raw = dict(
            zip(keys, tf.numpy_function(read, [lo, hi], dtypes, stateful=False), strict=True)
        )
        inputs = _reconstruct(split, include_long_range, raw)
        targets = {
            "energy": tf.reshape(energies[k], (1,)),
            "forces": tf.ensure_shape(raw["forces"], (None, 3)),
        }
        return inputs, targets

    ds = tf.data.Dataset.range(len(split))
    if shuffle and shuffle_buffer > 1:
        ds = ds.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    return ds.prefetch(tf.data.AUTOTUNE)


def make_batched_tf_dataset(
    split: DescriptorSplit, molecules_per_batch: int, *, include_long_range: bool = True
) -> tf.data.Dataset:
    """Build the evaluation / inference pipeline (several molecules per element, unshuffled).

    Parameters
    ----------
    split : DescriptorSplit
        Source split.
    molecules_per_batch : int
        Molecules B per element (the last element may hold fewer).
    include_long_range : bool
        Include the long-range inputs (see :func:`make_tf_dataset`).

    Returns
    -------
    tf.data.Dataset
        Elements ``(inputs, targets)``: the inputs of :func:`make_tf_dataset` with
        the atoms of B consecutive molecules concatenated (A atoms) plus
        ``segment_ids`` (A,) int32 (molecule of each atom, 0 … B-1), and targets
        ``energy`` (B,) and ``forces`` (A, 3). The molecule energy is
        ``unsorted_segment_sum(atomic_energy, segment_ids, B)``. Stored order (B19).

    Raises
    ------
    ValueError
        If ``molecules_per_batch`` is not positive.
    """
    import tensorflow as tf

    molecules_per_batch = operator.index(molecules_per_batch)
    if molecules_per_batch < 1:
        raise ValueError(f"molecules_per_batch must be positive, got {molecules_per_batch}")
    offsets_np = np.asarray(split.atom_offsets, dtype=np.int64)
    offsets = tf.constant(offsets_np)
    energies = tf.constant(np.asarray(split.energies, dtype=np.float32))
    n = len(split)
    read, dtypes, keys = _atom_block_fn(split, include_long_range)

    def load(b: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        first = b * molecules_per_batch
        last = tf.minimum(first + molecules_per_batch, n)
        lo, hi = offsets[first], offsets[last]
        raw = dict(
            zip(keys, tf.numpy_function(read, [lo, hi], dtypes, stateful=False), strict=True)
        )
        inputs = _reconstruct(split, include_long_range, raw)
        counts = offsets[first + 1 : last + 1] - offsets[first:last]
        inputs["segment_ids"] = tf.repeat(
            tf.range(last - first, dtype=tf.int32), tf.cast(counts, tf.int32)
        )
        targets = {
            "energy": energies[first:last],
            "forces": tf.ensure_shape(raw["forces"], (None, 3)),
        }
        return inputs, targets

    n_batches = -(-n // molecules_per_batch)
    ds = tf.data.Dataset.range(n_batches)
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    return ds.prefetch(tf.data.AUTOTUNE)
