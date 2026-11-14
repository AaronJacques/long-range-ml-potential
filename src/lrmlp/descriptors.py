"""Local and long-range descriptor matrices of every atom (thesis Sec. 4.1-4.2).

This module replaces ``get_s``, ``get_expanded_distance_vector[_local]``,
``process_neighbour``, ``sort_matrix``, the row building of ``create_matrices``
and ``pad_df_entry`` of the legacy ``Neural-Network/Dataset.py`` (commit
``9413bfd``). The neighbour partition and the long-range cell aggregates come
from :mod:`lrmlp.grid`; everything here is vectorised NumPy.

For every atom i four matrices are built (equation numbers refer to the thesis):

R̃_local (Eq. 4.4-4.6, ``Dataset.py:258-259``)
    One row ``[s, s·x/r, s·y/r, s·z/r]`` per local neighbour j with
    ``(x, y, z) = r_i − r_j``, r = ‖r_i − r_j‖ and the switching function s(r)
    of Eq. 4.5. Rows with r = r_c (s = 0) are kept.
Z_i (Eq. 4.15)
    One row ``[z_i, z_j]`` per row of R̃_local.
R̃_long (Eq. 4.16-4.18, ``Dataset.py:238-240``)
    One row ``[1/r, x/r, y/r, z/r]`` per long-range cell k with
    ``(x, y, z) = r_i − r_m^k`` and the Z-weighted centre of mass r_m^k
    (Eq. 4.12). The direction is a unit vector, there is **no** s factor. For
    r = 0 the row is ``[0, 0, 0, 0]`` (legacy behaviour).
F_i (Eq. 4.19-4.20, ``Dataset.py:274``)
    One row ``[z_i, f_1, …, f_mmax]`` per long-range cell with the element
    counts f_Z of the cell. :class:`MoleculeDescriptors` stores only the counts
    (``long_range_counts``); the full row is built by :func:`pad_descriptors`
    and :meth:`AtomDescriptors.long_range_features`.

Row order per atom (both assignment modes)
    Local rows by (level of the cell of j, lexicographic cell rank, input index
    j); long-range rows by (level, cell rank). This is the traversal order of
    the legacy ``create_matrices`` (``Dataset.py:299-356``). With
    ``GridConfig.sort_neighbours`` the legacy expression
    ``np.flip(m[:, 0].argsort())`` (``Dataset.py:362-376``) is applied on top,
    to R̃_local with Z_i and to R̃_long with F_i.

Atom order
    Atoms are **always** in input order. The legacy code emitted them in cell
    traversal order while the force labels stayed in input order (B01).

The long-range differences r_i − r_m^k are computed relative to the grid origin
g_000 (Eq. 4.11). This is mathematically identical, deviates from the legacy
values by rounding only (≪ 1e-12) and makes the result bit-identical under
translations of positions on a dyadic lattice.

All arithmetic is float64; :func:`pad_descriptors` converts to the float32 /
uint8 model inputs. The module depends on NumPy only.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from lrmlp import grid as _grid
from lrmlp.config import GridConfig

__all__ = [
    "AtomDescriptors",
    "MoleculeDescriptors",
    "compute_descriptors",
    "max_neighbour_counts",
    "pad_descriptors",
    "switching_function",
]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

#: Largest atomic number that fits into the uint8 ``local_numbers`` model input.
_MAX_UINT8 = int(np.iinfo(np.uint8).max)


# --------------------------------------------------------------------------- data classes


@dataclass(frozen=True, slots=True, eq=False)
class AtomDescriptors:
    """Descriptor matrices of a single atom i (views into :class:`MoleculeDescriptors`).

    Attributes
    ----------
    center_number : int
        Atomic number z_i.
    local_matrix : ndarray of float64, shape (k, 4)
        R̃_local rows ``[s, s·x/r, s·y/r, s·z/r]`` (Eq. 4.4-4.6).
    local_numbers : ndarray of int64, shape (k, 2)
        Z_i rows ``[z_i, z_j]`` (Eq. 4.15).
    long_range_matrix : ndarray of float64, shape (m, 4)
        R̃_long rows ``[1/r, x/r, y/r, z/r]`` (Eq. 4.16-4.18).
    long_range_counts : ndarray of int64, shape (m, max_atomic_number)
        Element counts f_Z of the long-range cells, column Z − 1 holds element Z.
    """

    center_number: int
    local_matrix: FloatArray
    local_numbers: IntArray
    long_range_matrix: FloatArray
    long_range_counts: IntArray

    @property
    def long_range_features(self) -> IntArray:
        """F_i rows ``[z_i, f_1, …, f_mmax]`` (Eq. 4.19-4.20), shape (m, max_atomic_number + 1)."""
        m = self.long_range_counts.shape[0]
        center = np.full((m, 1), self.center_number, dtype=np.int64)
        return np.concatenate([center, self.long_range_counts], axis=1)


@dataclass(frozen=True, slots=True, eq=False)
class MoleculeDescriptors:
    """Ragged descriptor matrices of all atoms of one molecule.

    The rows of all atoms are concatenated in input atom order: atom i owns
    ``n_local[i]`` consecutive local rows and ``n_long[i]`` consecutive
    long-range rows. Use :meth:`atom` for the matrices of one atom.

    Attributes
    ----------
    local_matrix : ndarray of float64, shape (Σk, 4)
        R̃_local rows of all atoms (Eq. 4.4-4.6).
    local_numbers : ndarray of int64, shape (Σk, 2)
        Z_i rows ``[z_i, z_j]`` of all atoms (Eq. 4.15).
    n_local : ndarray of int64, shape (N,)
        Number k of local rows per atom.
    long_range_matrix : ndarray of float64, shape (Σm, 4)
        R̃_long rows of all atoms (Eq. 4.16-4.18).
    long_range_counts : ndarray of int64, shape (Σm, max_atomic_number)
        Element counts of the long-range cells (F_i without the z_i column,
        Eq. 4.19-4.20).
    n_long : ndarray of int64, shape (N,)
        Number m of long-range rows per atom.
    center_numbers : ndarray of int64, shape (N,)
        Atomic numbers z_i in input order.
    """

    local_matrix: FloatArray
    local_numbers: IntArray
    n_local: IntArray
    long_range_matrix: FloatArray
    long_range_counts: IntArray
    n_long: IntArray
    center_numbers: IntArray

    def __post_init__(self) -> None:
        """Check that the ragged arrays are consistent."""
        n = self.center_numbers.shape[0]
        if self.n_local.shape != (n,) or self.n_long.shape != (n,):
            raise ValueError("n_local and n_long must have one entry per atom")
        if np.any(self.n_local < 0) or np.any(self.n_long < 0):
            raise ValueError("row counts must be non-negative")
        k, m = int(self.n_local.sum()), int(self.n_long.sum())
        if self.local_matrix.shape != (k, 4) or self.local_numbers.shape != (k, 2):
            raise ValueError(f"expected {k} local rows of width 4 / 2")
        if self.long_range_matrix.shape != (m, 4):
            raise ValueError(f"expected {m} long-range rows of width 4")
        if self.long_range_counts.ndim != 2 or self.long_range_counts.shape[0] != m:
            raise ValueError(f"expected {m} long-range count rows")

    @classmethod
    def empty(cls, max_atomic_number: int = 100) -> MoleculeDescriptors:
        """Descriptors of a molecule without atoms (N = 0)."""
        return cls(
            local_matrix=np.zeros((0, 4), dtype=np.float64),
            local_numbers=np.zeros((0, 2), dtype=np.int64),
            n_local=np.zeros(0, dtype=np.int64),
            long_range_matrix=np.zeros((0, 4), dtype=np.float64),
            long_range_counts=np.zeros((0, max_atomic_number), dtype=np.int64),
            n_long=np.zeros(0, dtype=np.int64),
            center_numbers=np.zeros(0, dtype=np.int64),
        )

    @property
    def n_atoms(self) -> int:
        """Number N of atoms."""
        return int(self.center_numbers.shape[0])

    @property
    def max_atomic_number(self) -> int:
        """Length m_max of the element count vector (p. 42)."""
        return int(self.long_range_counts.shape[1])

    @property
    def local_offsets(self) -> IntArray:
        """Start of the local rows of every atom, shape (N + 1,)."""
        return np.concatenate([[0], np.cumsum(self.n_local)]).astype(np.int64)

    @property
    def long_range_offsets(self) -> IntArray:
        """Start of the long-range rows of every atom, shape (N + 1,)."""
        return np.concatenate([[0], np.cumsum(self.n_long)]).astype(np.int64)

    def atom(self, i: int) -> AtomDescriptors:
        """Matrices of atom ``i`` (input order; negative indices count from the end).

        Raises
        ------
        IndexError
            If ``i`` is out of range.
        """
        n = self.n_atoms
        i = operator.index(i)
        if not -n <= i < n:
            raise IndexError(f"atom index {i} out of range for {n} atoms")
        i %= n
        lo, hi = self.local_offsets[i : i + 2]
        lr_lo, lr_hi = self.long_range_offsets[i : i + 2]
        return AtomDescriptors(
            center_number=int(self.center_numbers[i]),
            local_matrix=self.local_matrix[lo:hi],
            local_numbers=self.local_numbers[lo:hi],
            long_range_matrix=self.long_range_matrix[lr_lo:lr_hi],
            long_range_counts=self.long_range_counts[lr_lo:lr_hi],
        )


# --------------------------------------------------------------------------- functions


def switching_function(
    r: npt.ArrayLike, cutoff: float, inner_cutoff: float
) -> npt.NDArray[np.float64]:
    """Smooth 1/r switching function s(r) (Eq. 4.5, legacy ``get_s``).

    * s(r) = 1/r for r < r_cs,
    * s(r) = 1/r · (0.5·cos(π (r − r_cs) / (r_c − r_cs)) + 0.5) for r_cs ≤ r < r_c,
    * s(r) = 0 for r ≥ r_c.

    s is continuous at r_cs and r_c.

    Parameters
    ----------
    r : array_like of float
        Distances in Å, r > 0 (r = 0 gives ``inf``).
    cutoff : float
        Cutoff radius r_c (Eq. 4.14).
    inner_cutoff : float
        Inner cutoff radius r_cs < r_c.

    Returns
    -------
    ndarray of float64
        s(r) with the shape of ``r``; the arithmetic is the one of ``get_s``
        (``Dataset.py:243-255``).
    """
    r = np.asarray(r, dtype=np.float64)
    s = np.zeros_like(r)
    with np.errstate(divide="ignore"):
        inner = r < inner_cutoff
        s[inner] = 1 / r[inner]
        switch = (r >= inner_cutoff) & (r < cutoff)
        rs = r[switch]
        s[switch] = (
            1 / rs * (0.5 * np.cos(np.pi * (rs - inner_cutoff) / (cutoff - inner_cutoff)) + 0.5)
        )
    return s


def _segment_order(atom: IntArray, *keys: npt.NDArray[np.integer]) -> IntArray:
    """Stable order by ``atom`` first, then by ``keys`` (most significant first)."""
    return np.lexsort((*reversed(keys), atom)).astype(np.int64)


def _legacy_sort(values: FloatArray, counts: IntArray) -> IntArray:
    """Permutation applying ``np.flip(m[:, 0].argsort())`` to every atom segment.

    ``values`` is the first column of the concatenated rows in legacy traversal
    order, ``counts`` the rows per atom. Segments without ties in ``values``
    have a unique descending order and are sorted in one vectorised
    ``lexsort``. Segments with ties are re-sorted with the legacy expression
    itself, because the tie order of the default (unstable) ``argsort`` is an
    implementation detail of NumPy.
    """
    n_rows = values.shape[0]
    atom = np.repeat(np.arange(counts.shape[0]), counts)
    perm = np.lexsort((-values, atom)).astype(np.int64)
    sorted_values = values[perm]
    tie = (sorted_values[1:] == sorted_values[:-1]) & (atom[1:] == atom[:-1])
    if n_rows > 1 and np.any(tie):
        offsets = np.concatenate([[0], np.cumsum(counts)])
        for a in np.unique(atom[1:][tie]):
            lo, hi = int(offsets[a]), int(offsets[a + 1])
            perm[lo:hi] = lo + np.flip(values[lo:hi].argsort())
    return perm


def compute_descriptors(
    positions: npt.ArrayLike, numbers: npt.ArrayLike, grid: GridConfig
) -> MoleculeDescriptors:
    """Build R̃_local, Z_i, R̃_long and the counts of F_i for every atom.

    Parameters
    ----------
    positions : array_like, shape (N, 3)
        Atom positions in Å.
    numbers : array_like of int, shape (N,)
        Atomic numbers Z ≥ 1.
    grid : GridConfig
        Δg, l_max, r_c / r_cs, m_max, the assignment mode and
        ``sort_neighbours`` (see :mod:`lrmlp.grid`).

    Returns
    -------
    MoleculeDescriptors
        Ragged rows of all atoms in input order (B01); the row order per atom is
        described in the module docstring. Values are float64.

    Raises
    ------
    ValueError
        If the input is malformed or two distinct atoms coincide (B12, raised by
        :func:`lrmlp.grid.partition`).

    Notes
    -----
    Local rows follow Eq. 4.4-4.6 and 4.15, long-range rows Eq. 4.16-4.20; the
    centre of mass is Eq. 4.12 and the cell levels Eq. 4.13.
    """
    part = _grid.partition(positions, numbers, grid)
    pos = np.asarray(positions, dtype=np.float64)
    z = np.asarray(numbers).astype(np.int64)
    n = pos.shape[0]

    # ---- local rows R̃_local and Z_i (Eq. 4.4-4.6, 4.15)
    ii, jj = np.nonzero(part.local_mask)
    pair_levels = part.cell_levels[ii, part.cell_of_atom[jj]]
    order = _segment_order(ii, pair_levels, part.cell_of_atom[jj], jj)
    ii, jj = ii[order], jj[order]
    diff = pos[ii] - pos[jj]  # legacy sign: position - neighbour_pos (A08)
    r = part.distances[ii, jj]
    s = switching_function(r, grid.cutoff, grid.inner_cutoff)
    local_matrix = np.empty((ii.shape[0], 4), dtype=np.float64)
    local_matrix[:, 0] = s
    local_matrix[:, 1:] = diff * s[:, None] / r[:, None]
    local_numbers = np.stack([z[ii], z[jj]], axis=1)
    n_local = np.bincount(ii, minlength=n).astype(np.int64)

    # ---- long-range rows R̃_long and F_i (Eq. 4.12, 4.16-4.20)
    rel = pos - pos.min(axis=0)  # relative to g_000 (Eq. 4.11), see module docstring
    agg = _grid.cell_aggregates(rel, z, part, grid.max_atomic_number)
    li, lc = np.nonzero(agg.present)
    order = _segment_order(li, agg.level[li, lc], lc)
    li, lc = li[order], lc[order]
    lr_diff = rel[li] - agg.center_of_mass[li, lc]
    lr_norm = np.linalg.norm(lr_diff, axis=-1)
    nonzero = lr_norm > 0
    inverse = np.zeros_like(lr_norm)
    inverse[nonzero] = 1 / lr_norm[nonzero]
    long_range_matrix = np.empty((li.shape[0], 4), dtype=np.float64)
    long_range_matrix[:, 0] = inverse
    long_range_matrix[:, 1:] = lr_diff * inverse[:, None]
    long_range_counts = agg.counts[li, lc]
    n_long = np.bincount(li, minlength=n).astype(np.int64)

    if grid.sort_neighbours:
        perm = _legacy_sort(local_matrix[:, 0], n_local)
        local_matrix, local_numbers = local_matrix[perm], local_numbers[perm]
        perm = _legacy_sort(long_range_matrix[:, 0], n_long)
        long_range_matrix, long_range_counts = long_range_matrix[perm], long_range_counts[perm]

    return MoleculeDescriptors(
        local_matrix=local_matrix,
        local_numbers=local_numbers.astype(np.int64),
        n_local=n_local,
        long_range_matrix=long_range_matrix,
        long_range_counts=long_range_counts.astype(np.int64),
        n_long=n_long,
        center_numbers=z,
    )


def max_neighbour_counts(desc: MoleculeDescriptors) -> tuple[int, int]:
    """Largest number of local and long-range rows over all atoms.

    Returns
    -------
    tuple of int
        ``(N_max^local, N_max^long)`` of the molecule (p. 44); ``(0, 0)`` for a
        molecule without atoms.
    """
    if desc.n_atoms == 0:
        return 0, 0
    return int(desc.n_local.max()), int(desc.n_long.max())


def _scatter_index(counts: IntArray) -> tuple[IntArray, IntArray]:
    """(atom, slot) of every concatenated row for the given rows per atom."""
    atom = np.repeat(np.arange(counts.shape[0]), counts)
    starts = (np.cumsum(counts) - counts).astype(np.int64)
    slot = np.arange(atom.shape[0]) - np.repeat(starts, counts)
    return atom, slot


def pad_descriptors(
    desc: MoleculeDescriptors, n_max_local: int, n_max_long_range: int
) -> dict[str, np.ndarray]:
    """Zero-pad the ragged rows of every atom to N_max (p. 44, legacy ``pad_df_entry``).

    Parameters
    ----------
    desc : MoleculeDescriptors
        Descriptors of one molecule.
    n_max_local : int
        Rows N_max^local of the padded R̃_local and Z_i.
    n_max_long_range : int
        Rows N_max^long of the padded R̃_long and F_i.

    Returns
    -------
    dict of str to ndarray
        ``local_matrix`` (N, n_max_local, 4) float32,
        ``local_numbers`` (N, n_max_local, 2) uint8,
        ``long_range_matrix`` (N, n_max_long_range, 4) float32 and
        ``long_range_features`` (N, n_max_long_range, max_atomic_number + 1)
        float32 with column 0 = z_i and column Z = count of element Z
        (Eq. 4.19-4.20). Padding rows are all zero, including column 0. A
        molecule without atoms gives arrays with leading dimension 0 (the legacy
        code returned shape (1, n, w) float64 here, B13).

    Raises
    ------
    ValueError
        If an atom has more rows than ``n_max_local`` / ``n_max_long_range``, a
        size is negative or an atomic number does not fit into uint8.
    """
    n_max_local = operator.index(n_max_local)
    n_max_long_range = operator.index(n_max_long_range)
    if n_max_local < 0 or n_max_long_range < 0:
        raise ValueError("n_max_local and n_max_long_range must be non-negative")
    max_local, max_long = max_neighbour_counts(desc)
    if max_local > n_max_local:
        raise ValueError(f"an atom has {max_local} local rows > n_max_local = {n_max_local}")
    if max_long > n_max_long_range:
        raise ValueError(
            f"an atom has {max_long} long-range rows > n_max_long_range = {n_max_long_range}"
        )
    if desc.local_numbers.size and (
        desc.local_numbers.max() > _MAX_UINT8 or desc.local_numbers.min() < 0
    ):
        raise ValueError(f"atomic numbers must be in [0, {_MAX_UINT8}] for uint8 local_numbers")

    n = desc.n_atoms
    width = desc.max_atomic_number + 1
    local_matrix = np.zeros((n, n_max_local, 4), dtype=np.float32)
    local_numbers = np.zeros((n, n_max_local, 2), dtype=np.uint8)
    long_range_matrix = np.zeros((n, n_max_long_range, 4), dtype=np.float32)
    long_range_features = np.zeros((n, n_max_long_range, width), dtype=np.float32)

    atom, slot = _scatter_index(desc.n_local)
    local_matrix[atom, slot] = desc.local_matrix
    local_numbers[atom, slot] = desc.local_numbers
    atom, slot = _scatter_index(desc.n_long)
    long_range_matrix[atom, slot] = desc.long_range_matrix
    long_range_features[atom, slot, 0] = desc.center_numbers[atom]
    long_range_features[atom, slot, 1:] = desc.long_range_counts

    return {
        "local_matrix": local_matrix,
        "local_numbers": local_numbers,
        "long_range_matrix": long_range_matrix,
        "long_range_features": long_range_features,
    }
