"""Physical invariances of the descriptors and the atomic network.

The tests are deliberately honest about what the method guarantees:

* **Must hold**: translation (descriptors bit-identical, energy equal), the local
  neighbourhood under rotation in the ``"thesis"`` mode (neighbour set, s values
  and the canonically sorted Gram matrix R̃ R̃ᵀ), permutation of the input atoms
  (outputs permute along, row multisets per atom are unchanged), the energy under
  permutations that keep the order of the atoms inside every grid cell, the
  smoothness of s(r) and of the local rows at r_cs and r_c (Eq. 4.5), and the
  masking of zero padding rows in D = Gᵀ R̃ R̃ᵀ G2 (Eq. 4.8).
* **No guarantee** (deviation A10, a property of the method, not of the code): the
  single embedding network of p. 45 / Fig. 4.3 sees the *flattened, ordered*
  neighbour list, and order and cells depend on the axis-aligned grid of
  Eq. 4.11. The energy therefore changes in general under rotations and under
  permutations that reorder the atoms inside a cell. These tests are
  ``xfail(strict=False, raises=AssertionError)``: any other exception is a failure.
* **Not applicable** (deviation A09): forces are a direct network output, not
  −∂E/∂x, so a finite-difference gradient check does not apply. The alignment of
  force labels and atoms (B01) is tested in ``test_dataset.py`` and
  ``test_training.py``.

The model has fixed random weights (seeded) and the architecture of
``tests/fixtures/configs/smoke.toml``. With N = 21 atoms an atom has at most 20
local rows and at most 20 long-range cells, so N_max = 20 fits every transformed
geometry of the test molecule.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import keras
import numpy as np
import pytest

from lrmlp.config import ExperimentConfig, GridConfig, load_config
from lrmlp.descriptors import MoleculeDescriptors, compute_descriptors, switching_function
from lrmlp.grid import partition
from lrmlp.model import FeatureMatrix, build_model
from lrmlp.potential import Potential

A10_REASON = (
    "A10: axis-aligned grid and ordered flattened embedding; see docs/methodology.md#invariances"
)
A09_REASON = "A09: forces are a direct network output, not -dE/dx; see docs/methodology.md#forces"

MODES = ("thesis", "legacy")
MODEL_SEED = 1234
#: Lattice of the test positions (2^-10 Å) and of the translations (2^-4 Å).
POSITION_LATTICE = 2.0**-10
SHIFT_LATTICE = 2.0**-4
SHIFTS = (
    np.array([3.0, -5.0, 7.0]),
    np.array([16.0, 0.0, -1.0]),
    np.array([-40.0, 12.0, 3.0]),
)
ROTATION_SEEDS = (11, 22, 33)


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def smoke_config(fixtures_dir: Path) -> ExperimentConfig:
    """The smoke configuration (thesis mode, Δg = 1 Å, l_max = 2)."""
    return load_config(fixtures_dir / "configs" / "smoke.toml")


@pytest.fixture(scope="module")
def molecule(fixtures_dir: Path, smoke_config: ExperimentConfig) -> tuple[np.ndarray, np.ndarray]:
    """Frame 0 of the synthetic fixture on a 2^-10 Å lattice (21 atoms, C9H8O4).

    The lattice makes translations by multiples of 2^-4 Å exact in float64. The
    molecule has no pair distance within 1e-6 Å of r_c or r_cs, so the local
    neighbourhood is unambiguous under rotation.
    """
    with np.load(fixtures_dir / "synthetic_small.npz") as raw:
        pos = np.round(raw["R"][0] / POSITION_LATTICE) * POSITION_LATTICE
        z = raw["z"].astype(np.int64)
    d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)[~np.eye(len(z), dtype=bool)]
    grid = smoke_config.grid
    assert np.abs(d - grid.cutoff).min() > 1e-6
    assert np.abs(d - grid.inner_cutoff).min() > 1e-6
    return pos, z


def _config(base: ExperimentConfig, mode: str) -> ExperimentConfig:
    return dataclasses.replace(base, grid=dataclasses.replace(base.grid, assignment=mode))


@pytest.fixture(scope="module")
def potentials(smoke_config: ExperimentConfig, molecule: tuple[np.ndarray, np.ndarray]) -> dict:
    """One seeded smoke-architecture potential per assignment mode (N_max = N − 1)."""
    n_max = len(molecule[1]) - 1
    out = {}
    for mode in MODES:
        cfg = _config(smoke_config, mode)
        keras.utils.set_random_seed(MODEL_SEED)
        model = build_model(cfg.model, n_max, n_max, cfg.grid.max_atomic_number)
        out[mode] = Potential(model, cfg, n_max, n_max)
    return out


# --------------------------------------------------------------------------- helpers


def _energy(potential: Potential, pos: np.ndarray, z: np.ndarray) -> float:
    return float(potential.predict(pos, z).energies[0])


def _atomic(potential: Potential, pos: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Atomic energies E_i (output column 0) in input atom order."""
    x = {k: v.astype(np.float32) for k, v in potential.descriptors(pos, z).items()}
    return np.asarray(potential.model(x, training=False))[:, 0]


def _rotation(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def _descriptor_arrays(desc: MoleculeDescriptors) -> dict[str, np.ndarray]:
    return {f.name: getattr(desc, f.name) for f in dataclasses.fields(desc) if f.name != "species"}


def _sorted_rows(matrix: np.ndarray, numbers: np.ndarray) -> np.ndarray:
    """Rows ``[matrix | numbers]`` sorted lexicographically (canonical multiset form)."""
    full = np.concatenate([matrix, numbers.astype(np.float64)], axis=1)
    return full[np.lexsort(full.T[::-1])]


def _relative_change(a: float, b: float) -> float:
    return abs(a - b) / abs(b)


# --------------------------------------------------------------------------- 1. translation


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("shift", SHIFTS, ids=["s1", "s2", "s3"])
def test_translation_descriptors_bit_identical_and_energy_equal(
    mode: str,
    shift: np.ndarray,
    potentials: dict[str, Potential],
    molecule: tuple[np.ndarray, np.ndarray],
) -> None:
    pos, z = molecule
    potential = potentials[mode]
    moved = pos + shift * SHIFT_LATTICE
    ref = _descriptor_arrays(compute_descriptors(pos, z, potential.config.grid))
    new = _descriptor_arrays(compute_descriptors(moved, z, potential.config.grid))
    for key, value in ref.items():
        np.testing.assert_array_equal(new[key], value, err_msg=key)
    np.testing.assert_allclose(_energy(potential, moved, z), _energy(potential, pos, z), rtol=1e-6)


# --------------------------------------------------------------------------- 2./3. rotation


@pytest.mark.parametrize("seed", ROTATION_SEEDS)
def test_rotation_local_neighbourhood_invariant(
    seed: int, potentials: dict[str, Potential], molecule: tuple[np.ndarray, np.ndarray]
) -> None:
    """Thesis mode: the local environment N_rc(i) does not depend on the grid (p. 43)."""
    pos, z = molecule
    grid = potentials["thesis"].config.grid
    rotated = pos @ _rotation(seed).T
    np.testing.assert_array_equal(
        partition(rotated, z, grid).local_mask, partition(pos, z, grid).local_mask
    )
    ref, rot = compute_descriptors(pos, z, grid), compute_descriptors(rotated, z, grid)
    np.testing.assert_array_equal(rot.n_local, ref.n_local)
    for i in range(len(z)):
        a, b = rot.atom(i), ref.atom(i)
        np.testing.assert_allclose(
            np.sort(a.local_matrix[:, 0]), np.sort(b.local_matrix[:, 0]), rtol=1e-9
        )
        # Canonical order: descending s (distinct, so the order is well defined).
        ob = np.argsort(-b.local_matrix[:, 0])
        oa = np.argsort(-a.local_matrix[:, 0])
        assert b.local_matrix.shape[0] < 2 or np.diff(b.local_matrix[ob, 0]).max() < -1e-9
        np.testing.assert_array_equal(a.local_numbers[oa], b.local_numbers[ob])
        gram_a = a.local_matrix[oa] @ a.local_matrix[oa].T
        gram_b = b.local_matrix[ob] @ b.local_matrix[ob].T
        np.testing.assert_allclose(gram_a, gram_b, rtol=1e-9, atol=1e-12)


@pytest.mark.xfail(strict=False, raises=AssertionError, reason=A10_REASON)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("seed", ROTATION_SEEDS)
def test_rotation_energy(
    mode: str,
    seed: int,
    potentials: dict[str, Potential],
    molecule: tuple[np.ndarray, np.ndarray],
) -> None:
    pos, z = molecule
    potential = potentials[mode]
    rotated = pos @ _rotation(seed).T
    e_ref, e_rot = _energy(potential, pos, z), _energy(potential, rotated, z)
    assert _relative_change(e_rot, e_ref) <= 1e-6, (
        f"relative energy change under rotation: {_relative_change(e_rot, e_ref):.3e}"
    )


# --------------------------------------------------------------------------- 4./5. permutation


@pytest.fixture(scope="module")
def dense_molecule(molecule: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """The composition of ``molecule`` packed into a 3 Å box, several atoms per cell.

    With Δg = 1 Å the C9H8O4 frame has one atom per occupied cell, so the order
    inside a cell never matters there. Here many cells hold several atoms, which
    makes the permutation tests discriminating. Positions lie on the 2^-10 Å
    lattice, pair distances are ≥ 0.4 Å.
    """
    rng = np.random.default_rng(2024)
    z = molecule[1]
    while True:
        pos = np.round(rng.uniform(0.0, 3.0, size=(len(z), 3)) / POSITION_LATTICE)
        pos *= POSITION_LATTICE
        d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)[~np.eye(len(z), dtype=bool)]
        if d.min() >= 0.4:
            return pos, z


def _multi_atom_cells(pos: np.ndarray, z: np.ndarray, grid: GridConfig) -> int:
    return int((np.bincount(partition(pos, z, grid).cell_of_atom) > 1).sum())


def _same_element_permutation(z: np.ndarray, seed: int) -> np.ndarray:
    """Random permutation that only exchanges atoms of the same element."""
    rng = np.random.default_rng(seed)
    perm = np.arange(len(z))
    for element in np.unique(z):
        idx = np.flatnonzero(z == element)
        perm[idx] = rng.permutation(idx)
    assert not np.array_equal(perm, np.arange(len(z)))
    return perm


def _cell_order_preserving_permutation(
    pos: np.ndarray, z: np.ndarray, grid: GridConfig
) -> np.ndarray:
    """Reverse the order of the cells, keep the order of the atoms inside each cell."""
    cell = partition(pos, z, grid).cell_of_atom
    perm = np.lexsort((np.arange(len(z)), -cell))
    assert not np.array_equal(perm, np.arange(len(z)))
    return perm


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("kind", ["same_element", "arbitrary"])
def test_permutation_descriptors_permute_along(
    mode: str,
    kind: str,
    potentials: dict[str, Potential],
    dense_molecule: tuple[np.ndarray, np.ndarray],
) -> None:
    """Atom k of the permuted input carries the rows of atom perm[k] (as a multiset)."""
    pos, z = dense_molecule
    grid = potentials[mode].config.grid
    assert _multi_atom_cells(pos, z, grid) >= 3
    if kind == "same_element":
        perm = _same_element_permutation(z, seed=5)
    else:
        perm = np.random.default_rng(6).permutation(len(z))
    ref = compute_descriptors(pos, z, grid)
    new = compute_descriptors(pos[perm], z[perm], grid)
    np.testing.assert_array_equal(new.center_numbers, ref.center_numbers[perm])
    np.testing.assert_array_equal(new.n_local, ref.n_local[perm])
    np.testing.assert_array_equal(new.n_long, ref.n_long[perm])
    for k, i in enumerate(perm):
        a, b = new.atom(k), ref.atom(int(i))
        # Local rows: identical arithmetic, only the order inside a cell may change.
        np.testing.assert_array_equal(
            _sorted_rows(a.local_matrix, a.local_numbers),
            _sorted_rows(b.local_matrix, b.local_numbers),
        )
        # Long-range rows are ordered by (level, cell), which does not depend on the
        # atom order; the centre of mass is a sum and may differ by rounding.
        np.testing.assert_array_equal(a.long_range_counts, b.long_range_counts)
        np.testing.assert_allclose(a.long_range_matrix, b.long_range_matrix, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("which", ["dense", "c9h8o4"])
def test_permutation_energy_cell_order_preserving(
    mode: str,
    which: str,
    potentials: dict[str, Potential],
    molecule: tuple[np.ndarray, np.ndarray],
    dense_molecule: tuple[np.ndarray, np.ndarray],
) -> None:
    """Model inputs permute bit-exactly, so atomic and total energies are invariant.

    The model output itself depends on the row position in the batch by float32
    rounding (≈ 1e-6 relative); this is the only tolerance.
    """
    pos, z = dense_molecule if which == "dense" else molecule
    potential = potentials[mode]
    perm = _cell_order_preserving_permutation(pos, z, potential.config.grid)
    ref, new = potential.descriptors(pos, z), potential.descriptors(pos[perm], z[perm])
    for key, value in ref.items():
        np.testing.assert_array_equal(new[key], value[perm], err_msg=key)
    np.testing.assert_allclose(
        _atomic(potential, pos[perm], z[perm]), _atomic(potential, pos, z)[perm], atol=1e-5
    )
    np.testing.assert_allclose(
        _energy(potential, pos[perm], z[perm]), _energy(potential, pos, z), rtol=1e-6
    )


@pytest.mark.xfail(strict=False, raises=AssertionError, reason=A10_REASON)
@pytest.mark.parametrize("mode", MODES)
def test_permutation_energy_arbitrary(
    mode: str, potentials: dict[str, Potential], dense_molecule: tuple[np.ndarray, np.ndarray]
) -> None:
    """Reversing the input reverses the atom order inside every multi-atom cell."""
    pos, z = dense_molecule
    potential = potentials[mode]
    assert _multi_atom_cells(pos, z, potential.config.grid) >= 3
    perm = np.arange(len(z))[::-1]
    e_ref, e_perm = _energy(potential, pos, z), _energy(potential, pos[perm], z[perm])
    assert _relative_change(e_perm, e_ref) <= 1e-6, (
        f"relative energy change under permutation: {_relative_change(e_perm, e_ref):.3e}"
    )


# --------------------------------------------------------------------------- 6. smoothness


STEPS = (1e-3, 1e-5, 1e-7)


@pytest.mark.parametrize("radius", ["inner_cutoff", "cutoff"])
def test_switching_function_continuous_at_radii(
    radius: str, smoke_config: ExperimentConfig
) -> None:
    grid = smoke_config.grid
    r0 = getattr(grid, radius)
    jumps = []
    for h in STEPS:
        below, above = switching_function([r0 - h, r0 + h], grid.cutoff, grid.inner_cutoff)
        jumps.append(abs(below - above))
        assert jumps[-1] <= 2 * h  # |s'| < 1/r_cs² < 1 on both sides
    assert jumps == sorted(jumps, reverse=True)
    assert jumps[-1] < 1e-7


def _crossing_molecule(r: float) -> tuple[np.ndarray, np.ndarray]:
    """Atom 2 at distance ``r`` from atom 0 along (1, 1, 1); the rest is far inside r_c."""
    u = np.ones(3) / np.sqrt(3.0)
    pos = np.array([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], r * u])
    return pos, np.array([6, 1, 8])


@pytest.mark.parametrize("radius", ["inner_cutoff", "cutoff"])
def test_local_rows_continuous_when_crossing_radius(
    radius: str, smoke_config: ExperimentConfig
) -> None:
    """Thesis mode: the local rows of every atom change continuously across r_cs and r_c.

    Beyond r_c the row of the leaving neighbour is gone; it is compared as a zero
    row (its padding value), i.e. the multiset of rows padded to a fixed length is
    continuous. The long-range part is not smooth (thesis Sec. 7.1) and not tested.
    """
    grid = smoke_config.grid
    assert grid.assignment == "thesis"
    r0 = getattr(grid, radius)
    jumps = []
    for h in STEPS:
        inside = compute_descriptors(*_crossing_molecule(r0 - h), grid)
        outside = compute_descriptors(*_crossing_molecule(r0 + h), grid)
        jump = 0.0
        for i in range(3):
            a, b = inside.atom(i).local_matrix, outside.atom(i).local_matrix
            assert a.shape[0] - b.shape[0] == (1 if radius == "cutoff" and i != 1 else 0)
            b = np.concatenate([b, np.zeros((a.shape[0] - b.shape[0], 4))])
            a, b = a[np.argsort(-a[:, 0])], b[np.argsort(-b[:, 0])]
            jump = max(jump, float(np.abs(a - b).max()))
        jumps.append(jump)
        assert jump <= 2 * h
    assert jumps == sorted(jumps, reverse=True)
    assert jumps[-1] < 1e-7


# --------------------------------------------------------------------------- 7. masking


def test_padding_rows_do_not_change_descriptor() -> None:
    """D = Gᵀ R̃ R̃ᵀ G2 (Eq. 4.8) ignores zero rows of R̃, whatever G holds there."""
    rng = np.random.default_rng(0)
    batch, n_real, n_pad, k1, k2 = 5, 7, 6, 9, 4
    r = rng.normal(size=(batch, n_real, 4)).astype(np.float32)
    g = rng.normal(size=(batch, n_real, k1)).astype(np.float32)
    r_pad = np.concatenate([r, np.zeros((batch, n_pad, 4), np.float32)], axis=1)
    g_pad = np.concatenate([g, 1e3 * rng.normal(size=(batch, n_pad, k1)).astype(np.float32)], 1)
    layer = FeatureMatrix(k2)
    d = np.asarray(layer(r, g))
    d_pad = np.asarray(layer(r_pad, g_pad))
    assert d.shape == (batch, k1, k2)
    np.testing.assert_allclose(d_pad, d, rtol=1e-5, atol=1e-5)
    expected = np.einsum("bnk,bnc,bmc,bml->bkl", g, r, r, g[..., :k2])
    np.testing.assert_allclose(d, expected, rtol=1e-4, atol=1e-4)


# --------------------------------------------------------------------------- 8. forces


def test_force_gradient_check_not_applicable() -> None:
    """F = −∇E by finite differences does not apply to the force mode (A09)."""
    pytest.skip(A09_REASON)
