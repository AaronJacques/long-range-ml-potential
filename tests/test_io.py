"""Tests of ``lrmlp.io``: reading sGDML / MD17 trajectories."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lrmlp.io import REQUIRED_KEYS, Trajectory, load_trajectory

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _write(path: Path, **arrays: np.ndarray) -> Path:
    np.savez(path, **arrays)
    return path


def _arrays(s: int = 4, n: int = 3, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "E": rng.normal(size=s),
        "F": rng.normal(size=(s, n, 3)),
        "R": rng.normal(size=(s, n, 3)) * 3,
        "z": np.array([6, 1, 8][:n] + [1] * max(0, n - 3), dtype=np.int64),
    }


def test_shapes_and_dtypes_of_synthetic_fixture() -> None:
    traj = load_trajectory(FIXTURES / "synthetic_small.npz")
    assert isinstance(traj, Trajectory)
    assert traj.energies.shape == (60,)
    assert traj.forces.shape == traj.positions.shape == (60, 21, 3)
    assert traj.numbers.shape == (21,)
    assert traj.energies.dtype == traj.forces.dtype == traj.positions.dtype == np.float64
    assert traj.numbers.dtype == np.int64
    assert (traj.n_frames, traj.n_atoms, traj.n_available) == (60, 21, 60)
    with np.load(FIXTURES / "synthetic_small.npz") as raw:
        np.testing.assert_array_equal(traj.positions, raw["R"])
        np.testing.assert_array_equal(traj.forces, raw["F"])
        np.testing.assert_array_equal(traj.energies, raw["E"])
        np.testing.assert_array_equal(traj.numbers, raw["z"])


def test_energy_column_is_flattened(tmp_path: Path) -> None:
    a = _arrays()
    path = _write(tmp_path / "e2d.npz", **{**a, "E": a["E"][:, None]})
    traj = load_trajectory(path)
    assert traj.energies.shape == (4,)
    np.testing.assert_array_equal(traj.energies, a["E"])


@pytest.mark.parametrize("n_samples", [1, 3, 4, 10])
def test_n_samples_takes_leading_frames(tmp_path: Path, n_samples: int) -> None:
    a = _arrays(s=4)
    traj = load_trajectory(_write(tmp_path / "t.npz", **a), n_samples=n_samples)
    size = min(n_samples, 4)
    assert traj.n_frames == size
    assert traj.n_available == 4
    np.testing.assert_array_equal(traj.energies, a["E"][:size])
    np.testing.assert_array_equal(traj.positions, a["R"][:size])
    np.testing.assert_array_equal(traj.forces, a["F"][:size])
    assert traj.positions.flags.c_contiguous


def test_extra_keys_are_ignored_and_integer_like_float_z_accepted(tmp_path: Path) -> None:
    a = _arrays()
    path = _write(tmp_path / "x.npz", **{**a, "z": a["z"].astype(float)}, name=np.array("x"))
    traj = load_trajectory(path)
    np.testing.assert_array_equal(traj.numbers, a["z"])
    assert traj.numbers.dtype == np.int64


def test_missing_keys_are_listed(tmp_path: Path) -> None:
    a = _arrays()
    path = _write(tmp_path / "m.npz", E=a["E"], z=a["z"])
    with pytest.raises(ValueError, match=r"\['F', 'R'\]"):
        load_trajectory(path)
    assert REQUIRED_KEYS == ("E", "F", "R", "z")


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("E", np.zeros((4, 2)), "E must have shape"),
        ("F", np.zeros((4, 2, 3)), "F must have shape"),
        ("R", np.zeros((3, 3, 3)), "R must have shape"),
        ("z", np.zeros((3, 1), dtype=np.int64), "z must have shape"),
        ("z", np.array([6.5, 1.0, 8.0]), "integer"),
        ("z", np.array([6, 0, 8]), ">= 1"),
    ],
)
def test_malformed_arrays_raise(tmp_path: Path, key: str, value: np.ndarray, match: str) -> None:
    a = _arrays()
    a[key] = value
    with pytest.raises(ValueError, match=match):
        load_trajectory(_write(tmp_path / "bad.npz", **a))


@pytest.mark.parametrize("n_samples", [0, -1, True])
def test_invalid_n_samples_raise(tmp_path: Path, n_samples: int) -> None:
    path = _write(tmp_path / "t.npz", **_arrays())
    with pytest.raises(ValueError, match="n_samples"):
        load_trajectory(path, n_samples=n_samples)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_trajectory(tmp_path / "nope.npz")


def test_pickled_objects_are_refused(tmp_path: Path) -> None:
    a = _arrays()
    path = tmp_path / "obj.npz"
    np.savez(path, **{**a, "E": np.array(list(a["E"]), dtype=object)})
    with pytest.raises(ValueError):
        load_trajectory(path)
