"""Hierarchical cell grid and the local / long-range neighbour partition.

This module replaces ``GridCell``, ``create_grid`` and the assignment part of
``create_matrices`` of the legacy ``Neural-Network/Dataset.py`` (commit ``9413bfd``)
with vectorised NumPy code. Equation and figure numbers refer to the thesis
(Sec. 4.2, pp. 42-43).

Grid (Eq. 4.11, 4.13)
    The grid is axis-aligned with origin g_000 = component-wise minimum of the
    positions and cell edge Δg. Atom i lies in cell ``floor((r_i - g_000) / Δg)``.
    The level of a cell c seen from atom i is the Chebyshev distance between the
    cell index of i and c; level 0 is the own cell.

Cutoff (Eq. 4.14)
    r_c = (l_max + 1 + 1/√2)·Δg.

Two assignment modes (``GridConfig.assignment``) decide for every ordered pair
(i, j), i ≠ j, whether atom j is a local neighbour of i, contributes to a
long-range cell of i, or is dropped:

``"thesis"`` (default, p. 43 and Fig. 4.1)
    The local environment is N_rc(i) = {j ≠ i : r_ij ≤ r_c}; "the grid is not
    relevant for this step". All remaining atoms stay in the *reduced grid*:
    every cell with at least one remaining atom becomes one long-range row, with
    the Z-weighted centre of mass (Eq. 4.12) and the element counts of the
    remaining atoms only. No atom is lost.

``"legacy"`` (``Dataset.py:310-356``)
    For atom i and a cell c on level L:

    * L ≥ l_max + 2: the *whole* cell is long range, including atoms with
      r_ij ≤ r_c (deviation A01).
    * L ≤ l_max + 1: atoms with r_ij ≤ r_c are local. Atoms with r_ij > r_c form
      the reduced long-range cell if L = l_max + 1 and are *dropped* (neither
      local nor long range) if L ≤ l_max (deviation A02).

Geometry note
    The thesis (p. 42) states that all cells up to level l_max lie within r_c.
    In 3D this does not hold: an atom in a cell on level ≤ l_max (a cell of G_i)
    can be up to √3·(l_max + 1)·Δg away (≈ 12.1 Å for Δg = 1 Å, l_max = 6), which
    is larger than r_c ≈ 7.71 Å. This is the origin of A02; the ``"thesis"`` mode does not
    depend on the assumption.

Differences to the legacy code that do not change results for valid input: the
self pair is excluded by index instead of by position equality (A14), and two
distinct atoms at distance 0 raise :class:`ValueError` (B12). The alternative
centre of mass weighted by main group numbers (``Dataset.py:104-112``) is gone
(A11).

All arithmetic is float64. The module depends on NumPy only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from lrmlp.config import GridConfig

__all__ = [
    "CellAggregates",
    "CellAssignment",
    "Partition",
    "assign_cells",
    "cell_aggregates",
    "cutoff_radius",
    "partition",
]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
Int32Array = npt.NDArray[np.int32]
BoolArray = npt.NDArray[np.bool_]


# --------------------------------------------------------------------------- data classes


@dataclass(frozen=True, slots=True, eq=False)
class CellAssignment:
    """Assignment of atoms to the cells of the grid (Eq. 4.11).

    Attributes
    ----------
    origin : ndarray of float64, shape (3,)
        Grid origin g_000, the component-wise minimum of the positions.
    cell_index : ndarray of int32, shape (N, 3)
        Integer cell index ``floor((r_i - g_000) / Δg)`` of every atom.
    cells : ndarray of int32, shape (M, 3)
        The distinct occupied cell indices in lexicographic order, as returned
        by ``np.unique(cell_index, axis=0)`` (legacy cell order).
    cell_of_atom : ndarray of int64, shape (N,)
        Row of ``cells`` that contains atom i, ``cells[cell_of_atom] == cell_index``.
    """

    origin: FloatArray
    cell_index: Int32Array
    cells: Int32Array
    cell_of_atom: IntArray

    @property
    def n_cells(self) -> int:
        """Number M of occupied cells."""
        return int(self.cells.shape[0])


@dataclass(frozen=True, slots=True, eq=False)
class Partition:
    """Local / long-range partition of all ordered atom pairs of one molecule.

    Every ``(N, N)`` mask is indexed ``[i, j]`` = "atom j as seen from atom i",
    atoms in input order. The diagonal is ``False`` in every mask. In the
    ``"thesis"`` mode ``local_mask | long_range_mask | eye`` is all ``True``; in
    the ``"legacy"`` mode ``dropped_mask`` holds the remaining pairs (A02).

    Attributes
    ----------
    distances : ndarray of float64, shape (N, N)
        Pair distances r_ij = ‖r_i − r_j‖.
    cell_index : ndarray of int32, shape (N, 3)
        Cell index of every atom (Eq. 4.11), see :class:`CellAssignment`.
    cell_of_atom : ndarray of int64, shape (N,)
        Row of ``cells`` that contains atom i.
    cells : ndarray of int32, shape (M, 3)
        Occupied cells in lexicographic order.
    cell_levels : ndarray of int64, shape (N, M)
        Level of cell c seen from atom i: Chebyshev distance of the cell indices
        (Eq. 4.13); 0 for the own cell.
    local_mask : ndarray of bool, shape (N, N)
        Atom j is a local neighbour of atom i (a row of R̃_i and Z_i, Eq. 4.15).
    long_range_mask : ndarray of bool, shape (N, N)
        Atom j contributes to the long-range cell ``cell_of_atom[j]`` of atom i
        (reduced grid, Fig. 4.1).
    dropped_mask : ndarray of bool, shape (N, N)
        Atom j is neither local nor long range for atom i. Only non-zero in the
        ``"legacy"`` mode (A02).
    cutoff : float
        Cutoff radius r_c used for the partition (Eq. 4.14).
    assignment : {"thesis", "legacy"}
        Assignment mode that produced the partition.
    """

    distances: FloatArray
    cell_index: Int32Array
    cell_of_atom: IntArray
    cells: Int32Array
    cell_levels: IntArray
    local_mask: BoolArray
    long_range_mask: BoolArray
    dropped_mask: BoolArray
    cutoff: float
    assignment: Literal["thesis", "legacy"]

    @property
    def n_atoms(self) -> int:
        """Number N of atoms."""
        return int(self.distances.shape[0])

    @property
    def n_cells(self) -> int:
        """Number M of occupied cells."""
        return int(self.cells.shape[0])

    @property
    def pair_levels(self) -> IntArray:
        """Level of the cell of atom j seen from atom i, shape (N, N)."""
        return self.cell_levels[:, self.cell_of_atom]


@dataclass(frozen=True, slots=True, eq=False)
class CellAggregates:
    """Long-range cell aggregates per atom (reduced grid, Eq. 4.12, p. 42).

    Built only from the atoms of :attr:`Partition.long_range_mask`. Entries of
    cells that are not present for an atom are zero.

    Attributes
    ----------
    present : ndarray of bool, shape (N, M)
        Cell c is a long-range cell of atom i, i.e. at least one atom of c is in
        ``long_range_mask[i]``. ``present[i].sum()`` is the number of long-range
        rows of atom i.
    center_of_mass : ndarray of float64, shape (N, M, 3)
        Z-weighted centre of mass Σ z_j r_j / Σ z_j over the long-range atoms j
        of cell c seen from atom i (Eq. 4.12, ``Dataset.py:96-103``).
    counts : ndarray of int64, shape (N, M, max_atomic_number)
        Element counts f_k of those atoms; index Z − 1 holds element Z, atoms
        with Z > ``max_atomic_number`` are not counted (``Dataset.py:34-49``).
    level : ndarray of int64, shape (N, M)
        Cell level seen from atom i (Eq. 4.13), as :attr:`Partition.cell_levels`.
    """

    present: BoolArray
    center_of_mass: FloatArray
    counts: IntArray
    level: IntArray


# --------------------------------------------------------------------------- functions


def cutoff_radius(grid_size: float, max_local_level: int) -> float:
    """Cutoff radius r_c = (l_max + 1 + 1/√2)·Δg (Eq. 4.14, Tab. 5.1).

    Parameters
    ----------
    grid_size : float
        Cell edge Δg in Å.
    max_local_level : int
        Highest local cell level l_max.

    Returns
    -------
    float
        r_c in Å; the same expression as :attr:`lrmlp.config.GridConfig.cutoff`
        and the legacy ``Constants.py:34``.
    """
    return (max_local_level + 1 + 0.5**0.5) * grid_size


def _as_positions(positions: npt.ArrayLike) -> FloatArray:
    pos = np.asarray(positions, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3 or pos.shape[0] == 0:
        raise ValueError(f"positions must have shape (N, 3) with N >= 1, got {pos.shape}")
    if not np.all(np.isfinite(pos)):
        raise ValueError("positions must be finite")
    return pos


def _as_numbers(numbers: npt.ArrayLike, n_atoms: int) -> IntArray:
    z = np.asarray(numbers)
    if z.shape != (n_atoms,):
        raise ValueError(f"numbers must have shape ({n_atoms},), got {z.shape}")
    if not np.issubdtype(z.dtype, np.integer):
        raise ValueError(f"numbers must be integers, got dtype {z.dtype}")
    if np.any(z < 1):
        raise ValueError("atomic numbers must be >= 1")
    return z.astype(np.int64)


def assign_cells(positions: npt.ArrayLike, grid_size: float) -> CellAssignment:
    """Assign every atom to its grid cell (Eq. 4.11).

    Parameters
    ----------
    positions : array_like, shape (N, 3)
        Atom positions in Å.
    grid_size : float
        Cell edge Δg in Å.

    Returns
    -------
    CellAssignment
        Origin g_000 = component-wise minimum of the positions, cell index
        ``floor((r - g_000) / Δg)`` per atom (int32, as ``Dataset.py:200``) and
        the occupied cells in lexicographic order.
    """
    pos = _as_positions(positions)
    if not grid_size > 0:
        raise ValueError(f"grid_size must be positive, got {grid_size!r}")
    origin = pos.min(axis=0)
    cell_index = np.floor((pos - origin) / grid_size).astype(np.int32)
    cells, inverse = np.unique(cell_index, axis=0, return_inverse=True)
    return CellAssignment(
        origin=origin,
        cell_index=cell_index,
        cells=cells.astype(np.int32),
        cell_of_atom=np.asarray(inverse, dtype=np.int64).reshape(-1),
    )


def partition(positions: npt.ArrayLike, numbers: npt.ArrayLike, grid: GridConfig) -> Partition:
    """Partition all ordered atom pairs into local, long-range and dropped.

    Parameters
    ----------
    positions : array_like, shape (N, 3)
        Atom positions in Å.
    numbers : array_like of int, shape (N,)
        Atomic numbers (only validated here; the partition is purely geometric).
    grid : GridConfig
        Δg, l_max and the ``assignment`` mode; r_c is ``grid.cutoff`` (Eq. 4.14).

    Returns
    -------
    Partition
        Distances, cells, cell levels (Eq. 4.13) and the pair masks, see the
        module docstring for the two assignment modes.

    Raises
    ------
    ValueError
        If the shapes are wrong, an atomic number is < 1 or two distinct atoms
        coincide (distance 0, B12).
    """
    pos = _as_positions(positions)
    n = pos.shape[0]
    _as_numbers(numbers, n)
    cells = assign_cells(pos, grid.grid_size)

    # Same expression as the legacy norm of ``position - neighbour_pos``.
    distances = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    off_diagonal = ~np.eye(n, dtype=bool)
    if np.any(off_diagonal & (distances == 0.0)):
        i, j = np.argwhere(off_diagonal & (distances == 0.0))[0]
        raise ValueError(f"atoms {i} and {j} coincide (distance 0)")

    cell_levels = np.abs(cells.cell_index[:, None, :] - cells.cells[None, :, :]).max(axis=-1)
    cell_levels = cell_levels.astype(np.int64)
    pair_levels = cell_levels[:, cells.cell_of_atom]

    cutoff = grid.cutoff
    inside = distances <= cutoff
    l_max = grid.max_local_level
    if grid.assignment == "thesis":
        local = off_diagonal & inside
        long_range = off_diagonal & ~inside
        dropped = np.zeros((n, n), dtype=bool)
    elif grid.assignment == "legacy":
        near = pair_levels <= l_max + 1
        local = off_diagonal & near & inside
        long_range = off_diagonal & (
            (pair_levels >= l_max + 2) | ((pair_levels == l_max + 1) & ~inside)
        )
        dropped = off_diagonal & (pair_levels <= l_max) & ~inside
    else:  # pragma: no cover - GridConfig validates the mode
        raise ValueError(f"unknown assignment {grid.assignment!r}")

    return Partition(
        distances=distances,
        cell_index=cells.cell_index,
        cell_of_atom=cells.cell_of_atom,
        cells=cells.cells,
        cell_levels=cell_levels,
        local_mask=local,
        long_range_mask=long_range,
        dropped_mask=dropped,
        cutoff=cutoff,
        assignment=grid.assignment,
    )


def cell_aggregates(
    positions: npt.ArrayLike,
    numbers: npt.ArrayLike,
    partition: Partition,
    max_atomic_number: int,
    species: npt.ArrayLike | None = None,
) -> CellAggregates:
    """Centre of mass and element counts of the long-range cells of every atom.

    Only the atoms of ``partition.long_range_mask`` enter (reduced grid, p. 43,
    Fig. 4.1). In the ``"legacy"`` mode cells on level ≥ l_max + 2 keep all of
    their atoms and cells on level l_max + 1 keep the atoms beyond r_c, exactly
    as ``Dataset.py:310-356``.

    Parameters
    ----------
    positions : array_like, shape (N, 3)
        Atom positions in Å, the same as for :func:`partition`.
    numbers : array_like of int, shape (N,)
        Atomic numbers Z.
    partition : Partition
        Result of :func:`partition` for these positions.
    max_atomic_number : int
        Length m_max of the count vector (p. 42).
    species : array_like of int, optional
        Count only these distinct elements (1 ≤ Z ≤ ``max_atomic_number``):
        ``counts[..., s]`` then holds element ``species[s]`` and has
        ``len(species)`` columns instead of m_max. This is a compact form of the
        same counts (``counts_full[..., species[s] - 1]``); the default is the
        full m_max layout.

    Returns
    -------
    CellAggregates
        ``present``, the Z-weighted centre of mass (Eq. 4.12), the element counts
        (index Z − 1, Z > ``max_atomic_number`` ignored, or the ``species``
        columns) and the cell level.
    """
    pos = _as_positions(positions)
    n = pos.shape[0]
    z = _as_numbers(numbers, n)
    if partition.n_atoms != n:
        raise ValueError(f"partition has {partition.n_atoms} atoms, positions have {n}")
    if max_atomic_number < 1:
        raise ValueError(f"max_atomic_number must be positive, got {max_atomic_number}")
    m = partition.n_cells

    long_range = partition.long_range_mask.astype(np.float64)  # (N, N)
    cell_one_hot = np.zeros((n, m), dtype=np.float64)  # (N, M)
    cell_one_hot[np.arange(n), partition.cell_of_atom] = 1.0

    weights = long_range * z[None, :].astype(np.float64)  # z_j of the long-range atoms
    weight_sum = weights @ cell_one_hot  # (N, M), Σ z_j
    present = (long_range @ cell_one_hot) > 0.0

    weighted_positions = (cell_one_hot[:, :, None] * pos[:, None, :]).reshape(n, 3 * m)
    numerator = (weights @ weighted_positions).reshape(n, m, 3)  # Σ z_j r_j
    denominator = np.where(present, weight_sum, 1.0)[:, :, None]
    center_of_mass = np.where(present[:, :, None], numerator / denominator, 0.0)

    # Counts per (cell, element) group, scattered into the dense (M, n_columns) layout.
    if species is None:
        n_columns = max_atomic_number
        column = z - 1
        counted = z <= max_atomic_number
    else:
        sp = np.asarray(species).astype(np.int64).reshape(-1)
        if sp.size and (sp.min() < 1 or sp.max() > max_atomic_number):
            raise ValueError(f"species must lie in 1 … {max_atomic_number}, got {sp.tolist()}")
        if np.unique(sp).size != sp.size:
            raise ValueError(f"species must be distinct, got {sp.tolist()}")
        n_columns = sp.size
        lookup = np.full(max(max_atomic_number, int(z.max(initial=0))) + 1, -1, dtype=np.int64)
        lookup[sp] = np.arange(sp.size)
        column = lookup[z]
        counted = column >= 0
    counts = np.zeros((n, m * n_columns), dtype=np.int64)
    if np.any(counted):
        keys = partition.cell_of_atom[counted] * n_columns + column[counted]
        groups, group_of_atom = np.unique(keys, return_inverse=True)
        group_one_hot = np.zeros((int(counted.sum()), groups.size), dtype=np.float64)
        group_one_hot[np.arange(group_one_hot.shape[0]), group_of_atom.reshape(-1)] = 1.0
        group_counts = long_range[:, counted] @ group_one_hot  # (N, G), exact integers
        counts[:, groups] = np.rint(group_counts).astype(np.int64)

    return CellAggregates(
        present=present,
        center_of_mass=center_of_mass,
        counts=counts.reshape(n, m, n_columns),
        level=partition.cell_levels.copy(),
    )
