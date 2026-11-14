"""Tests for ``lrmlp.descriptors``: row shapes, row order, invariances and padding."""

from __future__ import annotations

import dataclasses
import subprocess
import sys
from collections import defaultdict

import numpy as np
import pytest

from lrmlp import grid as grid_mod
from lrmlp.config import GridConfig
from lrmlp.descriptors import (
    MoleculeDescriptors,
    compute_descriptors,
    max_neighbour_counts,
    pad_descriptors,
    switching_function,
)

#: Golden grid configurations (tests/fixtures/README.md): (Δg, l_max).
GRIDS = {"G-A": (1.0, 6), "G-B": (1.0, 2), "G-C": (2.0, 4)}
BOX = {"G-A": 16.0, "G-B": 8.0, "G-C": 26.0}
MODES = ("thesis", "legacy")
SPECIES = np.array([1, 1, 1, 6, 6, 7, 8, 9, 16, 17, 35, 53])


def _grid(name: str, mode: str = "thesis", **kwargs: object) -> GridConfig:
    grid_size, l_max = GRIDS[name]
    return GridConfig(grid_size=grid_size, max_local_level=l_max, assignment=mode, **kwargs)  # type: ignore[arg-type]


def _random_molecule(
    rng: np.random.Generator, n: int, box: float, cutoff: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Positions on a 2^-10 Å lattice, pair distances >= 0.3 Å and not within 1e-6 of r_c."""
    while True:
        pos = np.round(rng.uniform(-0.5 * box, 0.5 * box, size=(n, 3)) * 1024.0) / 1024.0
        d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)[~np.eye(n, dtype=bool)]
        if d.min() < 0.3 or (cutoff is not None and np.abs(d - cutoff).min() < 1e-6):
            continue
        return pos, rng.choice(SPECIES, size=n)


def _rows_by_atom(desc: MoleculeDescriptors) -> list[dict[str, np.ndarray]]:
    out = []
    for i in range(desc.n_atoms):
        a = desc.atom(i)
        out.append(
            {
                "local": a.local_matrix,
                "local_numbers": a.local_numbers,
                "long": a.long_range_matrix,
                "features": a.long_range_features,
            }
        )
    return out


def _canonical(rows: np.ndarray, extra: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rows and companion rows sorted lexicographically by all columns."""
    full = np.concatenate([rows, extra.astype(np.float64)], axis=1)
    order = np.lexsort(full.T[::-1])
    return rows[order], extra[order]


# --------------------------------------------------------------------------- brute-force reference


def _brute_force_thesis(pos: np.ndarray, z: np.ndarray, grid: GridConfig) -> list[dict]:
    """Pair loop after thesis pp. 43-44 (Eq. 4.4-4.6, 4.12, 4.15-4.20).

    Rows ordered by (cell level, cell rank, j) resp. (cell level, cell rank).
    """
    n = len(pos)
    rc, rcs, m_max = grid.cutoff, grid.inner_cutoff, grid.max_atomic_number
    origin = pos.min(axis=0)
    cell = [tuple(int(v) for v in np.floor((p - origin) / grid.grid_size)) for p in pos]
    rank = {c: k for k, c in enumerate(sorted(set(cell)))}

    def level(a: tuple[int, ...], b: tuple[int, ...]) -> int:
        return max(abs(x - y) for x, y in zip(a, b, strict=True))

    def s_of(r: float) -> float:
        if r < rcs:
            return 1.0 / r
        if r < rc:
            return 1.0 / r * (0.5 * np.cos(np.pi * (r - rcs) / (rc - rcs)) + 0.5)
        return 0.0

    result = []
    for i in range(n):
        local, rest = [], defaultdict(list)
        for j in range(n):
            if j == i:
                continue
            d = pos[i] - pos[j]
            r = float(np.sqrt(d @ d))
            if r <= rc:
                key = (level(cell[i], cell[j]), rank[cell[j]], j)
                s = s_of(r)
                local.append((key, [s, *(d * s / r)], [z[i], z[j]]))
            else:
                rest[cell[j]].append(j)
        long_range = []
        for c, members in rest.items():
            w = np.array([z[j] for j in members], dtype=float)
            com = (w[:, None] * pos[members]).sum(axis=0) / w.sum()
            d = pos[i] - com
            r = float(np.sqrt(d @ d))
            counts = np.zeros(m_max, dtype=np.int64)
            for j in members:
                if z[j] <= m_max:
                    counts[z[j] - 1] += 1
            long_range.append(((level(cell[i], c), rank[c]), [1 / r, *(d / r)], [z[i], *counts]))
        local.sort(key=lambda t: t[0])
        long_range.sort(key=lambda t: t[0])
        result.append(
            {
                "local": np.array([t[1] for t in local]).reshape(-1, 4),
                "local_numbers": np.array([t[2] for t in local], dtype=np.int64).reshape(-1, 2),
                "long": np.array([t[1] for t in long_range]).reshape(-1, 4),
                "features": np.array([t[2] for t in long_range], dtype=np.int64).reshape(
                    -1, m_max + 1
                ),
            }
        )
    return result


# --------------------------------------------------------------------------- 1. switching function


def test_switching_function_regions() -> None:
    rc, rcs = 7.0, 6.8
    r = np.array([1.0, 2.5, rcs, 6.9, rc, 7.5])
    s = switching_function(r, rc, rcs)
    np.testing.assert_allclose(s[:2], 1.0 / r[:2], rtol=0, atol=0)
    np.testing.assert_allclose(s[2], 1.0 / rcs, rtol=1e-15)  # cos(0) = 1
    np.testing.assert_allclose(s[3], 1.0 / 6.9 * 0.5, rtol=1e-12)  # midpoint: cos(π/2) = 0
    assert s[4] == 0.0 and s[5] == 0.0
    assert s.dtype == np.float64 and s.shape == r.shape


def test_switching_function_continuous_and_monotone() -> None:
    rc, rcs = 7.0, 6.8
    eps = 1e-9
    below, at = switching_function([rcs - eps, rcs], rc, rcs)
    assert abs(below - at) < 1e-8
    near, end = switching_function([rc - eps, rc], rc, rcs)
    assert abs(near - end) < 1e-8 and end == 0.0
    r = np.linspace(0.5, 8.0, 2001)
    assert np.all(np.diff(switching_function(r, rc, rcs)) <= 0)


def test_switching_function_scalar_and_shape() -> None:
    assert switching_function(2.0, 7.0, 6.8).shape == ()
    assert switching_function(np.ones((2, 3)), 7.0, 6.8).shape == (2, 3)


# --------------------------------------------------------------------------- 2. hand examples


def test_local_row_hand_example() -> None:
    # Two atoms 3 Å apart along x, well inside r_cs: s = 1/3, direction ±x.
    pos = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    desc = compute_descriptors(pos, [6, 1], GridConfig())
    a0, a1 = desc.atom(0), desc.atom(1)
    np.testing.assert_allclose(a0.local_matrix, [[1 / 3, -1 / 3, 0.0, 0.0]], rtol=1e-15)
    np.testing.assert_allclose(a1.local_matrix, [[1 / 3, 1 / 3, 0.0, 0.0]], rtol=1e-15)
    np.testing.assert_array_equal(a0.local_numbers, [[6, 1]])
    np.testing.assert_array_equal(a1.local_numbers, [[1, 6]])
    assert desc.long_range_matrix.shape == (0, 4)


def test_local_row_in_switching_region() -> None:
    grid = GridConfig()  # r_c ≈ 7.707, r_cs ≈ 7.507
    r = 0.5 * (grid.cutoff + grid.inner_cutoff)
    pos = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, r]])
    row = compute_descriptors(pos, [1, 1], grid).atom(0).local_matrix[0]
    s = 0.5 / r
    np.testing.assert_allclose(row, [s, 0.0, 0.0, -s], rtol=1e-12, atol=1e-15)


def test_row_at_cutoff_is_kept_with_zero_s() -> None:
    grid = GridConfig(grid_size=1.0, max_local_level=1)  # r_c = 2 + 1/√2
    pos = np.array([[0.0, 0.0, 0.0], [grid.cutoff, 0.0, 0.0]])
    desc = compute_descriptors(pos, [1, 8], grid)
    np.testing.assert_array_equal(desc.n_local, [1, 1])
    np.testing.assert_array_equal(desc.atom(0).local_matrix, [[0.0, -0.0, 0.0, 0.0]])


def test_long_range_row_hand_example() -> None:
    # Atom 0 alone; atoms 1 (H) and 2 (O) share a far cell. Thesis mode.
    grid = GridConfig(grid_size=1.0, max_local_level=1)  # r_c ≈ 2.71
    pos = np.array([[0.0, 0.0, 0.0], [6.25, 0.0, 0.0], [6.5, 0.5, 0.0]])
    z = np.array([6, 1, 8])
    desc = compute_descriptors(pos, z, grid)
    a0 = desc.atom(0)
    assert a0.local_matrix.shape == (0, 4)
    com = (1 * pos[1] + 8 * pos[2]) / 9
    d = pos[0] - com
    r = np.linalg.norm(d)
    np.testing.assert_allclose(a0.long_range_matrix, [[1 / r, *(d / r)]], rtol=1e-12)
    assert np.isclose(np.linalg.norm(a0.long_range_matrix[0, 1:]), 1.0)  # unit direction
    expected = np.zeros(101, dtype=np.int64)
    expected[[0, 1, 8]] = [6, 1, 1]
    np.testing.assert_array_equal(a0.long_range_features, [expected])
    np.testing.assert_array_equal(a0.long_range_counts, [expected[1:]])
    np.testing.assert_array_equal(desc.center_numbers, z)


def test_long_range_row_zero_distance_is_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """r = 0 gives the legacy row [0, 0, 0, 0] (``get_expanded_distance_vector``).

    Geometrically unreachable (a centre of mass lies inside its own cell, the atom
    outside), so the centre of mass is forced onto the atom.
    """
    import lrmlp.descriptors as mod

    original = mod._grid.cell_aggregates

    def patched(rel: np.ndarray, *args: object, **kwargs: object) -> grid_mod.CellAggregates:
        agg = original(rel, *args, **kwargs)  # type: ignore[arg-type]
        com = agg.center_of_mass.copy()
        com[0][agg.present[0]] = rel[0]
        return dataclasses.replace(agg, center_of_mass=com)

    monkeypatch.setattr(mod._grid, "cell_aggregates", patched)
    grid = GridConfig(grid_size=1.0, max_local_level=1)
    pos = np.array([[0.0, 0.0, 0.0], [6.25, 0.0, 0.0], [6.5, 0.5, 0.0]])
    desc = compute_descriptors(pos, [6, 1, 8], grid)
    np.testing.assert_array_equal(desc.atom(0).long_range_matrix, [[0.0, 0.0, 0.0, 0.0]])
    assert np.all(desc.atom(1).long_range_matrix[:, 0] > 0)


# --------------------------------------------------------------------------- 3. brute force


@pytest.mark.parametrize("config", sorted(GRIDS))
def test_thesis_matches_brute_force(config: str) -> None:
    rng = np.random.default_rng(2024)
    grid = _grid(config)
    for _ in range(8):  # 3 configurations × 8 = 24 random molecules
        n = int(rng.integers(2, 26))
        pos, z = _random_molecule(rng, n, BOX[config], grid.cutoff)
        desc = compute_descriptors(pos, z, grid)
        ref = _brute_force_thesis(pos, z, grid)
        for i, (got, want) in enumerate(zip(_rows_by_atom(desc), ref, strict=True)):
            for key in ("local", "long"):
                np.testing.assert_allclose(
                    got[key], want[key], rtol=1e-12, atol=1e-12, err_msg=f"{key} atom {i}"
                )
            for key in ("local_numbers", "features"):
                np.testing.assert_array_equal(got[key], want[key], err_msg=f"{key} atom {i}")


def test_thesis_rows_cover_all_atoms() -> None:
    rng = np.random.default_rng(5)
    grid = _grid("G-B")
    pos, z = _random_molecule(rng, 30, BOX["G-B"])
    desc = compute_descriptors(pos, z, grid)
    # Every other atom is either a local row or counted in exactly one long-range cell.
    lr_atoms = np.add.reduceat(desc.long_range_counts.sum(axis=1), desc.long_range_offsets[:-1])
    lr_atoms = np.where(desc.n_long > 0, lr_atoms, 0)
    np.testing.assert_array_equal(desc.n_local + lr_atoms, len(z) - 1)


# --------------------------------------------------------------------------- 4. input order (B01)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("sort", [False, True])
def test_rows_follow_input_order(mode: str, sort: bool) -> None:
    rng = np.random.default_rng(17)
    grid = _grid("G-B", mode, sort_neighbours=sort)
    pos, z = _random_molecule(rng, 25, BOX["G-B"])
    perm = rng.permutation(len(z))
    ref = compute_descriptors(pos, z, grid)
    moved = compute_descriptors(pos[perm], z[perm], grid)
    np.testing.assert_array_equal(moved.center_numbers, z[perm])
    np.testing.assert_array_equal(moved.n_local, ref.n_local[perm])
    np.testing.assert_array_equal(moved.n_long, ref.n_long[perm])
    for new, old in enumerate(perm):
        a, b = moved.atom(new), ref.atom(int(old))
        got, got_z = _canonical(a.local_matrix, a.local_numbers)
        want, want_z = _canonical(b.local_matrix, b.local_numbers)
        np.testing.assert_array_equal(got, want)
        np.testing.assert_array_equal(got_z, want_z)
        # Cell order is unchanged by an atom permutation; only the centre-of-mass sums
        # are reordered, hence the tolerance.
        if not sort:
            np.testing.assert_allclose(a.long_range_matrix, b.long_range_matrix, rtol=1e-13)
            np.testing.assert_array_equal(a.long_range_features, b.long_range_features)


def test_legacy_atom_order_is_not_used() -> None:
    """Atoms of one cell traversed late in legacy order still come at their input index."""
    grid = _grid("G-B", "legacy")
    pos = np.array([[5.5, 0.5, 0.5], [0.5, 0.5, 0.5], [1.5, 0.5, 0.5]])
    desc = compute_descriptors(pos, [8, 6, 1], grid)
    np.testing.assert_array_equal(desc.center_numbers, [8, 6, 1])
    np.testing.assert_array_equal(desc.atom(0).local_numbers[:, 0], 8)
    np.testing.assert_array_equal(desc.atom(1).local_numbers, [[6, 1]])


# --------------------------------------------------------------------------- 5. translation


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("k", [-1000, -37, 1, 5, 4096])
def test_translation_is_bit_exact(mode: str, k: int) -> None:
    rng = np.random.default_rng(3)
    grid = _grid("G-A", mode)
    pos, z = _random_molecule(rng, 30, BOX["G-A"])
    shift = k * 2.0**-4 * np.array([1.0, -3.0, 7.0])
    ref = compute_descriptors(pos, z, grid)
    moved = compute_descriptors(pos + shift, z, grid)
    for field in (
        "local_matrix",
        "local_numbers",
        "n_local",
        "long_range_matrix",
        "long_range_counts",
        "n_long",
        "center_numbers",
    ):
        np.testing.assert_array_equal(getattr(moved, field), getattr(ref, field), err_msg=field)


# --------------------------------------------------------------------------- 6. rotation (thesis)


def _rotation(rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


@pytest.mark.parametrize("config", sorted(GRIDS))
def test_local_rows_rotation_invariant(config: str) -> None:
    rng = np.random.default_rng(99)
    grid = _grid(config)
    pos, z = _random_molecule(rng, 24, BOX[config], grid.cutoff)
    rot = _rotation(rng)
    ref = compute_descriptors(pos, z, grid)
    moved = compute_descriptors(pos @ rot.T, z, grid)
    np.testing.assert_array_equal(moved.n_local, ref.n_local)
    for i in range(len(z)):
        a, b = moved.atom(i), ref.atom(i)
        oa = np.lexsort((a.local_numbers[:, 1], a.local_matrix[:, 0]))
        ob = np.lexsort((b.local_numbers[:, 1], b.local_matrix[:, 0]))
        np.testing.assert_allclose(a.local_matrix[oa, 0], b.local_matrix[ob, 0], rtol=1e-10)
        np.testing.assert_array_equal(a.local_numbers[oa], b.local_numbers[ob])
        # Canonically sorted Gram matrix R̃ R̃ᵀ is invariant (the directions rotate).
        ga = a.local_matrix[oa] @ a.local_matrix[oa].T
        gb = b.local_matrix[ob] @ b.local_matrix[ob].T
        np.testing.assert_allclose(ga, gb, rtol=1e-10, atol=1e-14)
        # The rows themselves rotate: R̃[:, 1:] → R̃[:, 1:] Rᵀ.
        np.testing.assert_allclose(
            a.local_matrix[oa, 1:], b.local_matrix[ob, 1:] @ rot.T, rtol=1e-10, atol=1e-12
        )


# --------------------------------------------------------------------------- 7. padding


def _small_desc() -> MoleculeDescriptors:
    rng = np.random.default_rng(8)
    pos, z = _random_molecule(rng, 12, BOX["G-B"])
    return compute_descriptors(pos, z, _grid("G-B", "legacy"))


def test_pad_shapes_dtypes_and_values() -> None:
    desc = _small_desc()
    n_loc, n_lr = max_neighbour_counts(desc)
    padded = pad_descriptors(desc, n_loc + 2, n_lr + 3)
    n = desc.n_atoms
    assert padded["local_matrix"].shape == (n, n_loc + 2, 4)
    assert padded["local_numbers"].shape == (n, n_loc + 2, 2)
    assert padded["long_range_matrix"].shape == (n, n_lr + 3, 4)
    assert padded["long_range_features"].shape == (n, n_lr + 3, 101)
    assert padded["local_matrix"].dtype == np.float32
    assert padded["local_numbers"].dtype == np.uint8
    assert padded["long_range_matrix"].dtype == np.float32
    assert padded["long_range_features"].dtype == np.float32
    for i in range(n):
        a, k, m = desc.atom(i), int(desc.n_local[i]), int(desc.n_long[i])
        np.testing.assert_array_equal(
            padded["local_matrix"][i, :k], a.local_matrix.astype(np.float32)
        )
        np.testing.assert_array_equal(padded["local_numbers"][i, :k], a.local_numbers)
        np.testing.assert_array_equal(
            padded["long_range_matrix"][i, :m], a.long_range_matrix.astype(np.float32)
        )
        np.testing.assert_array_equal(padded["long_range_features"][i, :m], a.long_range_features)
        np.testing.assert_array_equal(
            padded["long_range_features"][i, :m, 0], desc.center_numbers[i]
        )
        # Padding rows are all zero, including the z_i column.
        for key, rows in (
            ("local_matrix", k),
            ("local_numbers", k),
            ("long_range_matrix", m),
            ("long_range_features", m),
        ):
            assert not padded[key][i, rows:].any(), key


def test_pad_atom_without_rows() -> None:
    """An atom with neither local nor long-range rows (as in GB_edge/gb_assignment)."""
    grid = GridConfig(grid_size=1.0, max_local_level=2, assignment="legacy")
    # Atoms 0 and 1 are 2.6 Å apart (local). Atom 2 is beyond r_c ≈ 3.71 Å from both,
    # but all cells are on level 2 ≤ l_max, so its pairs are dropped (A02): no rows.
    pos = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.6], [2.9, 2.9, 0.1]])
    desc = compute_descriptors(pos, [6, 1, 8], grid)
    empty = np.flatnonzero((desc.n_local == 0) & (desc.n_long == 0))
    assert empty.size == 1
    padded = pad_descriptors(desc, *max_neighbour_counts(desc))
    for value in padded.values():
        assert not value[empty[0]].any()
    assert desc.atom(int(empty[0])).long_range_features.shape == (0, 101)


def test_pad_empty_molecule() -> None:
    """B13: no atoms → leading dimension 0 and float32 (legacy: (1, n, w) float64)."""
    desc = MoleculeDescriptors.empty(100)
    assert max_neighbour_counts(desc) == (0, 0)
    padded = pad_descriptors(desc, 5, 7)
    assert padded["local_matrix"].shape == (0, 5, 4)
    assert padded["local_numbers"].shape == (0, 5, 2)
    assert padded["long_range_matrix"].shape == (0, 7, 4)
    assert padded["long_range_features"].shape == (0, 7, 101)
    assert padded["local_matrix"].dtype == np.float32
    assert padded["long_range_features"].dtype == np.float32
    assert padded["local_numbers"].dtype == np.uint8


def test_pad_overflow_raises() -> None:
    desc = _small_desc()
    n_loc, n_lr = max_neighbour_counts(desc)
    assert n_loc > 0 and n_lr > 0
    with pytest.raises(ValueError, match="local"):
        pad_descriptors(desc, n_loc - 1, n_lr)
    with pytest.raises(ValueError, match="long-range"):
        pad_descriptors(desc, n_loc, n_lr - 1)
    with pytest.raises(ValueError):
        pad_descriptors(desc, -1, n_lr)


def test_pad_rejects_numbers_beyond_uint8() -> None:
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    desc = compute_descriptors(pos, [1, 300], GridConfig())
    with pytest.raises(ValueError, match="uint8"):
        pad_descriptors(desc, 1, 0)


# --------------------------------------------------------------------------- 8. validation


@pytest.mark.parametrize("mode", MODES)
def test_coincident_atoms_raise(mode: str) -> None:
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    with pytest.raises(ValueError, match="coincide"):
        compute_descriptors(pos, [6, 1, 1], _grid("G-A", mode))


def test_atom_accessor_bounds() -> None:
    desc = _small_desc()
    last = desc.atom(-1)
    assert last.center_number == desc.center_numbers[-1]
    np.testing.assert_array_equal(last.local_matrix, desc.atom(desc.n_atoms - 1).local_matrix)
    with pytest.raises(IndexError):
        desc.atom(desc.n_atoms)


def test_inconsistent_descriptors_raise() -> None:
    desc = MoleculeDescriptors.empty()
    with pytest.raises(ValueError):
        dataclasses.replace(desc, n_local=np.array([1]))
    with pytest.raises(AttributeError):
        desc.n_local = np.zeros(0)  # type: ignore[misc]


# --------------------------------------------------------------------------- 9. legacy sort


def _apply_legacy_sort(desc: MoleculeDescriptors) -> list[tuple[np.ndarray, ...]]:
    """``Dataset.sort_matrix`` (``np.flip(m[:, 0].argsort())``) applied per atom."""
    out = []
    for i in range(desc.n_atoms):
        a = desc.atom(i)
        rows = []
        for m, f in ((a.local_matrix, a.local_numbers), (a.long_range_matrix, a.long_range_counts)):
            idx = np.flip(m[:, 0].argsort())
            rows += [m[idx], f[idx]]
        out.append(tuple(rows))
    return out


@pytest.mark.parametrize("mode", MODES)
def test_sort_neighbours_reproduces_legacy_expression(mode: str) -> None:
    rng = np.random.default_rng(41)
    for _ in range(5):
        pos, z = _random_molecule(rng, 20, BOX["G-B"])
        plain = compute_descriptors(pos, z, _grid("G-B", mode))
        sorted_ = compute_descriptors(pos, z, _grid("G-B", mode, sort_neighbours=True))
        for i, want in enumerate(_apply_legacy_sort(plain)):
            a = sorted_.atom(i)
            got = (a.local_matrix, a.local_numbers, a.long_range_matrix, a.long_range_counts)
            for g, w in zip(got, want, strict=True):
                np.testing.assert_array_equal(g, w)


def test_sort_neighbours_with_ties() -> None:
    """Equal s values (octahedron) take the tie order of the legacy expression."""
    unit = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
    pos = np.concatenate([[[0.0, 0.0, 0.0]], 1.5 * unit, [[9.0, 9.0, 9.0], [-9.0, -9.0, -9.0]]])
    z = np.array([6, 1, 8, 1, 7, 1, 9, 6, 6])
    for mode in MODES:
        plain = compute_descriptors(pos, z, _grid("G-B", mode))
        assert len(set(plain.atom(0).local_matrix[:, 0].tolist())) == 1  # all tied
        sorted_ = compute_descriptors(pos, z, _grid("G-B", mode, sort_neighbours=True))
        for i, want in enumerate(_apply_legacy_sort(plain)):
            a = sorted_.atom(i)
            got = (a.local_matrix, a.local_numbers, a.long_range_matrix, a.long_range_counts)
            for g, w in zip(got, want, strict=True):
                np.testing.assert_array_equal(g, w)


# --------------------------------------------------------------------------- misc


def test_no_tensorflow_import() -> None:
    code = "import sys, lrmlp.descriptors; assert 'tensorflow' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
