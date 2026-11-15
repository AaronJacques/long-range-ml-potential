"""Tests for ``lrmlp.grid``: cells, levels and the thesis/legacy neighbour partition."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from lrmlp.config import GridConfig
from lrmlp.grid import (
    CellAggregates,
    Partition,
    assign_cells,
    cell_aggregates,
    cutoff_radius,
    partition,
)

try:
    from golden import fixtures as fx
except ImportError:  # pragma: no cover - depends on pytest import mode
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from golden import fixtures as fx

#: Golden grid configurations (tests/fixtures/README.md): (Δg, l_max).
GOLDEN_GRIDS = {"G-A": (1.0, 6), "G-B": (1.0, 2), "G-C": (2.0, 4)}
#: Box edge of the random molecules per configuration, a bit larger than 2·r_c so
#: that every pair class (local, A01, A02, reduced cell, far cell) occurs.
BOX = {"G-A": 16.0, "G-B": 8.0, "G-C": 26.0}
MODES = ("thesis", "legacy")
DESCRIPTOR_CASES = ("GA_aspirin", "GB_aspirin", "GA_edge", "GB_edge", "GC_stachyose")
SPECIES = np.array([1, 1, 1, 6, 6, 7, 8, 9, 16, 17, 35, 53, 101, 118])


def _grid(name: str, mode: str, **kwargs: object) -> GridConfig:
    grid_size, l_max = GOLDEN_GRIDS[name]
    return GridConfig(grid_size=grid_size, max_local_level=l_max, assignment=mode, **kwargs)  # type: ignore[arg-type]


def _random_molecule(rng: np.random.Generator, n: int, box: float) -> tuple[np.ndarray, np.ndarray]:
    """Positions on a 2^-10 Å lattice with pairwise distances >= 0.3 Å."""
    while True:
        pos = np.round(rng.uniform(-0.5 * box, 0.5 * box, size=(n, 3)) * 1024.0) / 1024.0
        d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
        if n == 1 or d[~np.eye(n, dtype=bool)].min() >= 0.3:
            return pos, rng.choice(SPECIES, size=n)


# --------------------------------------------------------------------------- brute-force references


def _cell_of(position: np.ndarray, origin: np.ndarray, grid_size: float) -> tuple[int, int, int]:
    idx = np.floor((position - origin) / grid_size).astype(np.int32)
    return int(idx[0]), int(idx[1]), int(idx[2])


def _chebyshev(a: tuple[int, int, int], b: tuple[int, int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]), abs(a[2] - b[2]))


def _reference(
    pos: np.ndarray, z: np.ndarray, grid: GridConfig
) -> tuple[dict[str, np.ndarray], list[dict[tuple[int, int, int], list[int]]]]:
    """Double Python loop over (i, j).

    ``"thesis"`` follows the wording of p. 43: local = all j != i with r_ij <= r_c,
    independent of the grid; every other atom stays in its cell of the reduced grid.
    ``"legacy"`` follows ``Dataset.py:310-356`` (loop over levels and cells of the
    neighbour tree, the self atom skipped).

    Returns the (N, N) masks and per atom ``{cell: [long-range atom indices]}``.
    """
    n = len(pos)
    origin = pos.min(axis=0)
    cell = [_cell_of(pos[i], origin, grid.grid_size) for i in range(n)]
    rc = grid.cutoff
    l_max = grid.max_local_level
    local = np.zeros((n, n), dtype=bool)
    long_range = np.zeros((n, n), dtype=bool)
    dropped = np.zeros((n, n), dtype=bool)
    lr_cells: list[dict[tuple[int, int, int], list[int]]] = []
    for i in range(n):
        reduced: dict[tuple[int, int, int], list[int]] = {}
        for j in range(n):
            if j == i:
                continue
            r = np.linalg.norm(pos[i] - pos[j], ord=2)
            level = _chebyshev(cell[i], cell[j])
            if grid.assignment == "thesis":
                if r <= rc:
                    local[i, j] = True
                else:
                    long_range[i, j] = True
                    reduced.setdefault(cell[j], []).append(j)
            elif level > l_max + 1:  # whole cell via its centroid (A01)
                long_range[i, j] = True
                reduced.setdefault(cell[j], []).append(j)
            elif r <= rc:
                local[i, j] = True
            elif level > l_max:  # reduced cell on level l_max + 1
                long_range[i, j] = True
                reduced.setdefault(cell[j], []).append(j)
            else:  # beyond r_c on level <= l_max: lost (A02)
                dropped[i, j] = True
        lr_cells.append(reduced)
    levels = np.array([[_chebyshev(cell[i], cell[j]) for j in range(n)] for i in range(n)])
    return {
        "local": local,
        "long_range": long_range,
        "dropped": dropped,
        "pair_levels": levels,
    }, lr_cells


def _check_against_reference(
    pos: np.ndarray, z: np.ndarray, grid: GridConfig
) -> tuple[Partition, CellAggregates]:
    part = partition(pos, z, grid)
    agg = cell_aggregates(pos, z, part, grid.max_atomic_number)
    ref, lr_cells = _reference(pos, z, grid)

    np.testing.assert_array_equal(part.local_mask, ref["local"])
    np.testing.assert_array_equal(part.long_range_mask, ref["long_range"])
    np.testing.assert_array_equal(part.dropped_mask, ref["dropped"])
    np.testing.assert_array_equal(part.pair_levels, ref["pair_levels"])
    np.testing.assert_array_equal(agg.level, part.cell_levels)

    rank = {tuple(int(v) for v in c): k for k, c in enumerate(part.cells)}
    m_max = grid.max_atomic_number
    for i, cells in enumerate(lr_cells):
        expected_present = np.zeros(part.n_cells, dtype=bool)
        for c, members in cells.items():
            k = rank[c]
            expected_present[k] = True
            zs = z[members].astype(np.float64)
            com = (pos[members] * zs[:, None]).sum(axis=0) / zs.sum()
            np.testing.assert_allclose(agg.center_of_mass[i, k], com, rtol=1e-13, atol=1e-12)
            counts = np.zeros(m_max, dtype=np.int64)
            for j in members:
                if z[j] <= m_max:
                    counts[z[j] - 1] += 1
            np.testing.assert_array_equal(agg.counts[i, k], counts)
        np.testing.assert_array_equal(agg.present[i], expected_present)
        assert not agg.counts[i, ~expected_present].any()
        assert not agg.center_of_mass[i, ~expected_present].any()
    return part, agg


# --------------------------------------------------------------------------- 1. cutoff


@pytest.mark.parametrize(
    ("grid_size", "l_max", "expected"),
    [(1.0, 6, 7.707), (2.0, 2, 7.414), (2.0, 4, 11.414), (1.0, 2, 3.707)],
)
def test_cutoff_radius_table_5_1(grid_size: float, l_max: int, expected: float) -> None:
    rc = cutoff_radius(grid_size, l_max)
    assert rc == pytest.approx(expected, abs=5e-4)
    assert rc == GridConfig(grid_size=grid_size, max_local_level=l_max).cutoff


# --------------------------------------------------------------------------- 2. cells and levels


def test_assign_cells_hand_example() -> None:
    pos = np.array(
        [
            [-1.0, 0.0, 2.0],  # origin component x
            [0.0, -0.5, 2.5],  # exactly on the x boundary -> upper cell
            [1.4, 1.6, 0.5],  # origin component z
            [-0.6, 0.4, 2.1],
        ]
    )
    cells = assign_cells(pos, grid_size=1.0)
    np.testing.assert_array_equal(cells.origin, [-1.0, -0.5, 0.5])
    assert cells.cell_index.dtype == np.int32
    np.testing.assert_array_equal(cells.cell_index, [[0, 0, 1], [1, 0, 2], [2, 2, 0], [0, 0, 1]])
    np.testing.assert_array_equal(cells.cells, [[0, 0, 1], [1, 0, 2], [2, 2, 0]])
    np.testing.assert_array_equal(cells.cell_of_atom, [0, 1, 2, 0])
    np.testing.assert_array_equal(cells.cells[cells.cell_of_atom], cells.cell_index)
    assert cells.n_cells == 3


def test_cells_are_lexicographic() -> None:
    pos, _ = _random_molecule(np.random.default_rng(7), 40, 12.0)
    cells = assign_cells(pos, 1.0).cells
    keys = [tuple(int(v) for v in c) for c in cells]
    assert keys == sorted(keys)
    assert len(set(keys)) == len(keys)


def test_levels_hand_example() -> None:
    pos = np.array([[0.2, 0.2, 0.2], [0.7, 0.9, 0.1], [2.5, 0.1, 0.3], [1.1, 3.9, 5.0]])
    part = partition(pos, [6, 1, 8, 7], GridConfig(grid_size=1.0, max_local_level=1))
    # cells: (0,0,0) [atoms 0, 1], (0,3,4) [atom 3], (2,0,0) [atom 2]
    np.testing.assert_array_equal(part.cells, [[0, 0, 0], [0, 3, 4], [2, 0, 0]])
    np.testing.assert_array_equal(part.cell_of_atom, [0, 0, 2, 1])
    np.testing.assert_array_equal(part.cell_levels, [[0, 4, 2], [0, 4, 2], [2, 4, 0], [4, 0, 4]])
    # level 0 = own cell
    np.testing.assert_array_equal(part.cell_levels[np.arange(4), part.cell_of_atom], 0)
    np.testing.assert_array_equal(np.diag(part.pair_levels), 0)
    assert part.pair_levels[0, 1] == 0


def test_single_atom() -> None:
    part = partition([[1.0, 2.0, 3.0]], [6], GridConfig())
    agg = cell_aggregates([[1.0, 2.0, 3.0]], [6], part, 100)
    assert part.n_atoms == 1 and part.n_cells == 1
    assert not (part.local_mask | part.long_range_mask | part.dropped_mask).any()
    assert agg.present.shape == (1, 1) and not agg.present.any()
    assert agg.counts.shape == (1, 1, 100)


# --------------------------------------------------------------------------- 3. brute force


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("config", sorted(GOLDEN_GRIDS))
def test_partition_matches_brute_force(config: str, mode: str) -> None:
    rng = np.random.default_rng([sorted(GOLDEN_GRIDS).index(config), MODES.index(mode)])
    grid = _grid(config, mode)
    seen = {"local": 0, "long_range": 0, "dropped": 0, "a01": 0}
    for k in range(24):
        n = int(rng.integers(2, 31)) if k else 1
        pos, z = _random_molecule(rng, n, BOX[config])
        part, _ = _check_against_reference(pos, z, grid)
        seen["local"] += int(part.local_mask.sum())
        seen["long_range"] += int(part.long_range_mask.sum())
        seen["dropped"] += int(part.dropped_mask.sum())
        a01 = part.long_range_mask & (part.distances <= grid.cutoff)
        seen["a01"] += int(a01.sum())
    assert seen["local"] > 0 and seen["long_range"] > 0
    if mode == "legacy":
        assert seen["dropped"] > 0 and seen["a01"] > 0, seen
    else:
        assert seen["dropped"] == 0 and seen["a01"] == 0, seen


@pytest.mark.parametrize("mode", MODES)
def test_brute_force_small_max_atomic_number(mode: str) -> None:
    rng = np.random.default_rng(99)
    grid = _grid("G-B", mode, max_atomic_number=8)
    for _ in range(5):
        pos, z = _random_molecule(rng, 20, BOX["G-B"])
        _, agg = _check_against_reference(pos, z, grid)
        assert agg.counts.shape[-1] == 8


def test_legacy_matches_independent_pair_classes() -> None:
    rng = np.random.default_rng(5)
    for config, (grid_size, l_max) in GOLDEN_GRIDS.items():
        grid = _grid(config, "legacy")
        for _ in range(5):
            pos, z = _random_molecule(rng, 25, BOX[config])
            part = partition(pos, z, grid)
            cls = fx.legacy_pair_classes(pos, grid_size, l_max, grid.cutoff)
            np.testing.assert_array_equal(part.local_mask, cls["local"])
            np.testing.assert_array_equal(part.dropped_mask, cls["dropped"])
            np.testing.assert_array_equal(part.long_range_mask, cls["reduced"] | cls["centroid"])


# --------------------------------------------------------------------------- 4. golden


def _golden_frames(case: str) -> list[tuple[GridConfig, dict[str, np.ndarray]]]:
    golden = fx.load_descriptor_golden(case)
    assert not bool(golden["sort_matrix"])
    grid = GridConfig(
        grid_size=float(golden["grid_size"]),
        max_local_level=int(golden["max_local_level"]),
        assignment="legacy",
    )
    assert grid.cutoff == float(golden["cut_off"])
    return [(grid, fx.frame(golden, f)) for f in range(int(golden["n_frames"]))]


@pytest.mark.golden
@pytest.mark.parametrize("case", DESCRIPTOR_CASES)
def test_golden_legacy_counts(case: str) -> None:
    for grid, fr in _golden_frames(case):
        pos, z, order = fr["positions"], fr["numbers"], fr["order"]
        part = partition(pos, z, grid)
        agg = cell_aggregates(pos, z, part, grid.max_atomic_number)
        local = np.asarray(fx.to_input_order(list(fr["local_counts"]), order))
        long_range = np.asarray(fx.to_input_order(list(fr["lr_counts"]), order))
        np.testing.assert_array_equal(part.local_mask.sum(axis=1), local)
        np.testing.assert_array_equal(agg.present.sum(axis=1), long_range)


@pytest.mark.golden
@pytest.mark.parametrize("case", DESCRIPTOR_CASES)
def test_golden_legacy_long_range_rows(case: str) -> None:
    """Aggregates in legacy row order (level, then cell rank) reproduce the LR rows."""
    for grid, fr in _golden_frames(case):
        pos, z, order = fr["positions"], fr["numbers"], fr["order"]
        part = partition(pos, z, grid)
        agg = cell_aggregates(pos, z, part, grid.max_atomic_number)
        rows = fx.split_ragged(fr["lr_rows"], fr["lr_counts"])
        features = fx.split_ragged(fr["lr_features"], fr["lr_counts"])
        for k, i in enumerate(order):
            cells = np.flatnonzero(agg.present[i])
            cells = cells[np.lexsort((cells, agg.level[i, cells]))]
            np.testing.assert_array_equal(features[k][:, 0], z[i])
            np.testing.assert_array_equal(features[k][:, 1:], agg.counts[i, cells])
            diff = pos[i] - agg.center_of_mass[i, cells]
            norm = np.linalg.norm(diff, axis=-1)
            np.testing.assert_allclose(rows[k][:, 0], 1.0 / norm, rtol=1e-12)
            np.testing.assert_allclose(rows[k][:, 1:], diff / norm[:, None], atol=1e-12)


# --------------------------------------------------------------------------- 5. modes


@pytest.mark.parametrize("config", sorted(GOLDEN_GRIDS))
def test_thesis_loses_no_atom(config: str) -> None:
    rng = np.random.default_rng(11)
    cases = list(fx.load_edge_cases().values())
    cases += [
        dict(zip(("R", "z"), _random_molecule(rng, 30, BOX[config]), strict=True)) for _ in range(5)
    ]
    for case in cases:
        part = partition(case["R"], case["z"], _grid(config, "thesis"))
        n = part.n_atoms
        covered = part.local_mask | part.long_range_mask | np.eye(n, dtype=bool)
        assert covered.all()
        assert not (part.local_mask & part.long_range_mask).any()
        assert not part.dropped_mask.any()


@pytest.mark.parametrize("config", sorted(GOLDEN_GRIDS))
def test_legacy_masks_are_disjoint_and_complete(config: str) -> None:
    for case in fx.load_edge_cases().values():
        part = partition(case["R"], case["z"], _grid(config, "legacy"))
        stack = np.stack([part.local_mask, part.long_range_mask, part.dropped_mask])
        np.testing.assert_array_equal(stack.sum(axis=0), ~np.eye(part.n_atoms, dtype=bool))


def test_edge_level_lmax2_within_rc() -> None:
    case = fx.load_edge_cases()["level_lmax2_within_rc"]
    thesis = partition(case["R"], case["z"], _grid("G-A", "thesis"))
    legacy = partition(case["R"], case["z"], _grid("G-A", "legacy"))
    # atom 1 (x = 0.999, cell 0) and atom 2 (x = 8.0, cell 8 = level l_max + 2)
    assert thesis.pair_levels[1, 2] == 8
    assert thesis.distances[1, 2] <= thesis.cutoff
    assert thesis.local_mask[1, 2] and thesis.local_mask[2, 1]
    assert not legacy.local_mask[1, 2] and not legacy.local_mask[2, 1]
    assert legacy.long_range_mask[1, 2] and legacy.long_range_mask[2, 1]
    assert legacy.local_mask[2].sum() == 0  # A01: no local row for the atom at x = 8.0


def test_edge_dropped_pairs() -> None:
    case = fx.load_edge_cases()["dropped_pairs"]
    thesis = partition(case["R"], case["z"], _grid("G-A", "thesis"))
    legacy = partition(case["R"], case["z"], _grid("G-A", "legacy"))
    cross = np.zeros((4, 4), dtype=bool)
    cross[:2, 2:] = cross[2:, :2] = True
    np.testing.assert_array_equal(legacy.dropped_mask, cross)
    assert not (legacy.long_range_mask & cross).any()
    assert not thesis.dropped_mask.any()
    assert thesis.long_range_mask[cross].all()
    agg = cell_aggregates(case["R"], case["z"], thesis, 100)
    # each atom sees the other cell as one long-range row with both of its atoms
    np.testing.assert_array_equal(agg.present.sum(axis=1), [1, 1, 1, 1])
    np.testing.assert_array_equal(agg.counts[0, 1, [0, 5]], [1, 1])  # H and C in cell (6,6,6)


def test_edge_case_mode_differences() -> None:
    """Ordered pairs whose class differs between the modes: only the A01/A02 cases."""
    expected = {
        "G-A": {"level_lmax2_within_rc": 2, "dropped_pairs": 8},
        "G-B": {"gb_assignment": 8},
    }
    for config, cases in expected.items():
        differing = {}
        for name, case in fx.load_edge_cases().items():
            thesis = partition(case["R"], case["z"], _grid(config, "thesis"))
            legacy = partition(case["R"], case["z"], _grid(config, "legacy"))
            n_diff = int(
                (
                    (thesis.local_mask != legacy.local_mask)
                    | (thesis.long_range_mask != legacy.long_range_mask)
                    | (thesis.dropped_mask != legacy.dropped_mask)
                ).sum()
            )
            if n_diff:
                differing[name] = n_diff
        assert differing == cases, config


# --------------------------------------------------------------------------- 6. translation


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("k", [-1000, -37, 1, 5, 4096])
def test_translation_is_bit_exact(mode: str, k: int) -> None:
    rng = np.random.default_rng(3)
    grid = _grid("G-A", mode)
    pos, z = _random_molecule(rng, 30, BOX["G-A"])
    shift = k * 2.0**-4 * np.array([1.0, -3.0, 7.0])
    ref = partition(pos, z, grid)
    moved = partition(pos + shift, z, grid)
    for field in ("distances", "cell_index", "cell_of_atom", "cells", "cell_levels"):
        np.testing.assert_array_equal(getattr(moved, field), getattr(ref, field), err_msg=field)
    for field in ("local_mask", "long_range_mask", "dropped_mask"):
        np.testing.assert_array_equal(getattr(moved, field), getattr(ref, field), err_msg=field)


# --------------------------------------------------------------------------- 7. validation


@pytest.mark.parametrize("mode", MODES)
def test_coincident_atoms_raise(mode: str) -> None:
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    with pytest.raises(ValueError, match="coincide"):
        partition(pos, [6, 1, 1], _grid("G-A", mode))


@pytest.mark.parametrize(
    ("positions", "numbers"),
    [
        (np.zeros((3, 2)), [1, 1, 1]),
        (np.zeros((0, 3)), []),
        ([[0.0, 0.0, 0.0], [1.0, 0.0, np.nan]], [1, 1]),
        ([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [1]),
        ([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [1, 0]),
        ([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [1.0, 6.0]),
    ],
)
def test_invalid_input_raises(positions: object, numbers: object) -> None:
    with pytest.raises(ValueError):
        partition(positions, numbers, GridConfig())  # type: ignore[arg-type]


def test_assign_cells_rejects_bad_grid_size() -> None:
    with pytest.raises(ValueError):
        assign_cells([[0.0, 0.0, 0.0]], 0.0)


def test_cell_aggregates_rejects_mismatch() -> None:
    pos = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    part = partition(pos, [1, 6], GridConfig(grid_size=1.0, max_local_level=1))
    with pytest.raises(ValueError):
        cell_aggregates(pos[:1], [1], part, 100)
    with pytest.raises(ValueError):
        cell_aggregates(pos, [1, 6], part, 0)
    with pytest.raises(ValueError, match="species must lie"):
        cell_aggregates(pos, [1, 6], part, 100, species=[1, 101])
    with pytest.raises(ValueError, match="distinct"):
        cell_aggregates(pos, [1, 6], part, 100, species=[6, 6])


@pytest.mark.parametrize("mode", ["thesis", "legacy"])
def test_cell_aggregates_species_columns(mode: str) -> None:
    """The compact counts equal the selected columns of the full m_max layout."""
    rng = np.random.default_rng(7)
    pos = np.round(rng.uniform(-8.0, 8.0, size=(40, 3)) * 64.0) / 64.0
    z = rng.choice([1, 6, 7, 8, 101], size=40)
    grid = GridConfig(grid_size=1.0, max_local_level=2, assignment=mode)  # type: ignore[arg-type]
    part = partition(pos, z, grid)
    full = cell_aggregates(pos, z, part, 100)
    for species in ([1, 6, 7, 8], [8, 1], [6], []):
        compact = cell_aggregates(pos, z, part, 100, species=species)
        assert compact.counts.shape == (*full.counts.shape[:2], len(species))
        np.testing.assert_array_equal(
            compact.counts, full.counts[:, :, np.asarray(species, dtype=np.int64) - 1]
        )
        np.testing.assert_array_equal(compact.present, full.present)
        np.testing.assert_array_equal(compact.center_of_mass, full.center_of_mass)


def test_dataclasses_are_frozen() -> None:
    part = partition([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [1, 1], GridConfig())
    with pytest.raises(AttributeError):
        part.cutoff = 1.0  # type: ignore[misc]


def test_no_tensorflow_import() -> None:
    code = "import sys, lrmlp.grid; assert 'tensorflow' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


# --------------------------------------------------------------------------- 8. runtime


@pytest.mark.parametrize("mode", MODES)
def test_partition_runtime_87_atoms(mode: str) -> None:
    data = fx.load_synthetic("stachyose_like")
    pos, z = data["R"][0], data["z"]
    assert pos.shape == (87, 3)
    grid = _grid("G-C", mode)
    partition(pos, z, grid)  # warm-up
    start = time.perf_counter()
    for _ in range(20):
        partition(pos, z, grid)
    mean = (time.perf_counter() - start) / 20
    assert mean < 0.020, f"partition took {mean * 1e3:.2f} ms"
