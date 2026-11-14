"""Deterministic synthetic test data (NumPy only, Python >= 3.10).

All generators are pure functions of their arguments: the same ``(n_frames, seed)``
always yields bit-identical arrays. Energies and forces come from a Lennard-Jones
pair potential with analytic gradients, so they are physically consistent
(``F = -dE/dR``) without needing any quantum-chemistry reference.

Units follow MD17: positions in Angstrom, energies in kcal/mol, forces in
kcal/(mol*Angstrom).

Public API
----------
``aspirin_like(n_frames, seed=0)``    21 atoms (C9H8O4), box ~ 9 Angstrom
``stachyose_like(n_frames, seed=1)``  87 atoms (C/H/O), box ~ 16 Angstrom
``lennard_jones(R)``                  energy (S,) and forces (S, N, 3)
``edge_cases()``                      dict of hand-built single molecules
``write_npz(path, E, F, R, z)``       MD17-style ``.npz`` (keys E, F, R, z)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

LJ_EPSILON = 0.1  # kcal/mol
LJ_SIGMA = 1.2  # Angstrom
POSITION_QUANTUM = 2.0**-10  # Angstrom; positions are rounded to this grid
MIN_DISTANCE = 1.0  # Angstrom; minimum pair distance in every frame
BASE_MIN_DISTANCE = 1.3  # Angstrom; minimum pair distance of the reference geometry
DISPLACEMENT_STD = 0.1  # Angstrom; per-frame Gaussian displacement

ASPIRIN_Z = np.array([6] * 9 + [1] * 8 + [8] * 4, dtype=np.int64)
# Stachyose is C24H42O21 (87 atoms), in fixed element order.
STACHYOSE_Z = np.array([6] * 24 + [1] * 42 + [8] * 21, dtype=np.int64)


def _quantize(x: np.ndarray) -> np.ndarray:
    return np.round(x / POSITION_QUANTUM) * POSITION_QUANTUM


def _min_pair_distance(R: np.ndarray) -> float:
    diff = R[:, None, :] - R[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    iu = np.triu_indices(len(R), k=1)
    return float(dist[iu].min()) if len(iu[0]) else np.inf


def _base_geometry(rng: np.random.Generator, n_atoms: int, box: float) -> np.ndarray:
    """Place atoms one by one uniformly in the box (rejection sampling)."""
    positions: list[np.ndarray] = []
    attempts = 0
    while len(positions) < n_atoms:
        attempts += 1
        if attempts > 1_000_000:
            raise RuntimeError("rejection sampling did not converge")
        candidate = _quantize(rng.uniform(0.0, box, size=3))
        if positions:
            d = np.sqrt(np.sum((np.asarray(positions) - candidate) ** 2, axis=1))
            if d.min() < BASE_MIN_DISTANCE:
                continue
        positions.append(candidate)
    return np.asarray(positions, dtype=np.float64)


def _frames(n_frames: int, seed: int, n_atoms: int, box: float) -> np.ndarray:
    """Reference geometry plus quantized Gaussian displacements per frame.

    Every frame is resampled until all pair distances are >= ``MIN_DISTANCE``.
    """
    rng = np.random.default_rng(seed)
    base = _base_geometry(rng, n_atoms, box)
    R = np.empty((n_frames, n_atoms, 3), dtype=np.float64)
    for f in range(n_frames):
        for _ in range(10_000):
            frame = _quantize(base + rng.normal(0.0, DISPLACEMENT_STD, size=base.shape))
            if _min_pair_distance(frame) >= MIN_DISTANCE:
                break
        else:  # pragma: no cover - would need an absurd displacement
            raise RuntimeError("could not sample a frame with the minimum distance")
        R[f] = frame
    return R


def lennard_jones(
    R: np.ndarray, epsilon: float = LJ_EPSILON, sigma: float = LJ_SIGMA
) -> tuple[np.ndarray, np.ndarray]:
    """Lennard-Jones energy and analytic forces.

    ``E = sum_{i<j} 4 eps [(sigma/r)^12 - (sigma/r)^6]`` and ``F = -grad E``.

    Args:
        R: positions, shape (S, N, 3) or (N, 3).

    Returns:
        ``(E, F)`` with shapes (S,) and (S, N, 3) (or () and (N, 3)).
    """
    R = np.asarray(R, dtype=np.float64)
    single = R.ndim == 2
    if single:
        R = R[None]
    n = R.shape[1]
    iu, ju = np.triu_indices(n, k=1)
    rij = R[:, iu, :] - R[:, ju, :]  # (S, P, 3), r_i - r_j
    r = np.sqrt(np.sum(rij * rij, axis=-1))  # (S, P)
    sr6 = (sigma / r) ** 6
    sr12 = sr6 * sr6
    E = np.sum(4.0 * epsilon * (sr12 - sr6), axis=1)
    # dE/dr = 4 eps (-12 sigma^12 / r^13 + 6 sigma^6 / r^7)
    dE_dr = 4.0 * epsilon * (-12.0 * sr12 + 6.0 * sr6) / r
    # F_i = -dE/dr * (r_i - r_j)/r ; F_j = -F_i (pair contribution)
    fij = -(dE_dr / r)[..., None] * rij
    F = np.zeros_like(R)
    for s in range(R.shape[0]):
        np.add.at(F[s], iu, fij[s])
        np.add.at(F[s], ju, -fij[s])
    if single:
        return E[0], F[0]
    return E, F


def _molecule(n_frames: int, seed: int, z: np.ndarray, box: float) -> dict[str, np.ndarray]:
    R = _frames(n_frames, seed, len(z), box)
    E, F = lennard_jones(R)
    return {"E": E, "F": F, "R": R, "z": z.copy()}


def aspirin_like(n_frames: int, seed: int = 0) -> dict[str, np.ndarray]:
    """Aspirin-sized molecule: z = C9H8O4 in fixed order, 21 atoms, box ~ 9 Angstrom.

    Returns:
        dict with ``E`` (S,), ``F`` (S, 21, 3), ``R`` (S, 21, 3), ``z`` (21,).
    """
    return _molecule(n_frames, seed, ASPIRIN_Z, box=9.0)


def stachyose_like(n_frames: int, seed: int = 1) -> dict[str, np.ndarray]:
    """Stachyose-sized molecule: C24H42O21, 87 atoms, box ~ 16 Angstrom.

    Returns:
        dict with ``E`` (S,), ``F`` (S, 87, 3), ``R`` (S, 87, 3), ``z`` (87,).
    """
    return _molecule(n_frames, seed, STACHYOSE_Z, box=16.0)


def _case(description: str, R: list[list[float]], z: list[int]) -> dict:
    return {
        "description": description,
        "R": np.asarray(R, dtype=np.float64),
        "z": np.asarray(z, dtype=np.int64),
    }


def edge_cases() -> dict[str, dict]:
    """Hand-built single molecules that pin down legacy assignment behaviour.

    Each value is ``{"description": str, "R": (N, 3) float64, "z": (N,) int64}``.
    Descriptions refer to G-A (grid 1.0, l_max 6, r_c = (7 + sqrt(0.5)) ~ 7.707);
    the same molecules are also evaluated with G-B. Grid indices are
    ``floor((R - R.min(axis=0)) / grid)``; all cases have their minimum at 0.
    """
    return {
        "level_lmax2_within_rc": _case(
            "A01: atoms at x = 0, 0.999, 8.0. The atom at x = 8.0 sits in a cell on "
            "level 8 = l_max + 2 seen from cell 0, but its distance to the atom at "
            "0.999 is 7.001 < r_c. Legacy code uses it only via the cell centroid, "
            "so that atom gets 0 local rows despite a neighbour inside r_c.",
            [[0.0, 0.0, 0.0], [0.999, 0.0, 0.0], [8.0, 0.0, 0.0]],
            [6, 1, 8],
        ),
        "dropped_pairs": _case(
            "A02: two atoms in cell (0,0,0) and two in cell (6,6,6) (level 6 = l_max). "
            "All four cross pairs are ~10.4 Angstrom > r_c and are dropped: neither "
            "local rows nor long-range rows.",
            [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [6.5, 6.5, 6.5], [6.9, 6.2, 6.6]],
            [6, 8, 6, 1],
        ),
        "partial_cell": _case(
            "Cell (7,0,0) is on level 7 = l_max + 1 from cell (0,0,0). From the atom at "
            "the origin, the atom at x = 7.1 is inside r_c (local row) and the atom at "
            "(7.9, 0.9, 0.9) is beyond r_c (reduced cell -> one long-range row).",
            [[0.0, 0.0, 0.0], [7.1, 0.0, 0.0], [7.9, 0.9, 0.9]],
            [6, 8, 1],
        ),
        "single_cell": _case(
            "Whole molecule in one cell: only local rows, no long-range rows.",
            [[0.1, 0.2, 0.3], [0.8, 0.4, 0.6], [0.5, 0.9, 0.1]],
            [6, 1, 1],
        ),
        "no_long_range": _case(
            "The atoms at x = 2.0 and x = 4.0 have no long-range rows: every other "
            "cell is on level <= l_max + 1 and every atom is inside r_c (the reduced "
            "cell on level 7 for x = 2.0 is empty). The atom at x = 0 sees the atom at "
            "x = 9.5 (level 9) as a long-range cell.",
            [[0.0, 0.0, 0.0], [2.0, 0.3, 0.1], [4.0, 0.1, 0.4], [9.5, 0.2, 0.2]],
            [6, 6, 8, 1],
        ),
        "cell_boundary": _case(
            "Atoms exactly on cell boundaries (x = 1.0, 7.0, 8.0): floor puts them into "
            "the upper cell. Seen from x = 0, x = 7.0 is level 7 at distance 7.0 "
            "(local) and x = 8.0 is level 8 (long range).",
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [7.0, 0.0, 0.0], [8.0, 0.0, 0.0]],
            [1, 6, 7, 8],
        ),
        "gb_assignment": _case(
            "A01 and A02 for G-B (grid 1.0, l_max 2, r_c ~ 3.707). x = 4.0 is on level "
            "4 = l_max + 2 from x = 0.999 at distance 3.001 < r_c (centroid only); the "
            "atom in cell (2,2,2) (level 2 = l_max) is beyond r_c from all others "
            "(dropped). Under G-A everything is local.",
            [[0.0, 0.0, 0.0], [0.999, 0.0, 0.0], [4.0, 0.0, 0.0], [2.9, 2.9, 2.9]],
            [6, 1, 8, 7],
        ),
        "multi_element": _case(
            "Nine elements (H, C, N, O, F, S, Cl, Br, I) spread over ~10 Angstrom, "
            "several sharing a cell, to pin Z-weighted centroids and the atomic "
            "feature vectors.",
            [
                [0.1, 0.1, 0.1],
                [0.6, 0.3, 0.2],
                [1.4, 0.2, 0.5],
                [3.3, 2.1, 0.4],
                [9.2, 0.4, 0.3],
                [9.6, 0.8, 0.7],
                [9.4, 9.3, 0.2],
                [5.2, 9.7, 9.1],
                [0.3, 9.5, 9.8],
            ],
            [1, 6, 7, 8, 9, 16, 17, 35, 53],
        ),
    }


def write_npz(path: str | Path, E: np.ndarray, F: np.ndarray, R: np.ndarray, z: np.ndarray) -> None:
    """Write an MD17-style dataset with keys ``E`` (S,), ``F``, ``R`` (S, N, 3), ``z`` (N,)."""
    from .fixtures import save_npz

    save_npz(
        path,
        {
            "E": np.asarray(E, dtype=np.float64),
            "F": np.asarray(F, dtype=np.float64),
            "R": np.asarray(R, dtype=np.float64),
            "z": np.asarray(z, dtype=np.int64),
        },
    )
