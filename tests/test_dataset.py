"""Tests of ``lrmlp.dataset``: split, storage format, alignment and tf.data pipelines."""

from __future__ import annotations

import dataclasses
import json
import os
import stat
import subprocess
import sys
import textwrap
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from lrmlp.config import ExperimentConfig, load_config
from lrmlp.dataset import (
    ARRAY_DTYPES,
    FORMAT_VERSION,
    SPLITS,
    DescriptorDataset,
    DescriptorSplit,
    build_dataset,
    load_dataset,
    make_batched_tf_dataset,
    make_tf_dataset,
    split_indices,
)
from lrmlp.descriptors import compute_descriptors, pad_descriptors

try:
    from golden import fixtures as fx
except ImportError:  # pragma: no cover - depends on pytest import mode
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from golden import fixtures as fx

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures"
SMOKE = FIXTURES / "configs" / "smoke.toml"
SPLIT_SIZES = (1, 2, 7, 60, 101, 2000, 65000)
FRACTIONS = ((0.1, 0.1), (0.05, 0.15), (0.2, 0.0), (0.0, 0.3))


def _raw(name: str = "synthetic_small") -> dict[str, np.ndarray]:
    return fx.load_synthetic(name)


@pytest.fixture(scope="module")
def smoke_config() -> ExperimentConfig:
    return load_config(SMOKE)


@pytest.fixture(scope="module")
def smoke_dir(smoke_config: ExperimentConfig, tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_dataset(
        smoke_config, output_dir=tmp_path_factory.mktemp("ds") / "smoke", n_workers=1
    )


@pytest.fixture(scope="module")
def smoke(smoke_dir: Path) -> DescriptorDataset:
    return load_dataset(smoke_dir)


@pytest.fixture(scope="module")
def mixed(tmp_path_factory: pytest.TempPathFactory, smoke_config: ExperimentConfig) -> Path:
    """Two sources with different molecules: 10 aspirin-like frames + 5 (clipped) stachyose."""
    config = smoke_config.with_overrides(
        {
            "data.sources": [
                str(FIXTURES / "synthetic_small.npz"),
                str(FIXTURES / "stachyose_like.npz"),
            ],
            "data.n_samples_per_source": 10,
            "data.val_fraction": 0.2,
            "data.test_fraction": 0.2,
        }
    )
    return build_dataset(config, output_dir=tmp_path_factory.mktemp("mixed") / "m", n_workers=2)


# --------------------------------------------------------------------------- 1. split


@pytest.mark.parametrize("n", SPLIT_SIZES)
@pytest.mark.parametrize(("v", "t"), FRACTIONS)
def test_split_matches_pandas(n: int, v: float, t: float) -> None:
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"frame": np.arange(n)})
    train_df = df.sample(frac=1 - (v + t), random_state=42)
    rest = df.drop(train_df.index)
    val_df = rest.sample(frac=v / (v + t), random_state=42) if v + t > 0 else rest.iloc[:0]
    test_df = rest.drop(val_df.index)
    train, val, test = split_indices(n, v, t, 42)
    np.testing.assert_array_equal(train, train_df.index.to_numpy())
    np.testing.assert_array_equal(val, val_df.index.to_numpy())
    np.testing.assert_array_equal(test, test_df.index.to_numpy())
    assert train.dtype == val.dtype == test.dtype == np.int64


def test_split_matches_golden() -> None:
    for name, case in fx.load_split_golden().items():
        parts = split_indices(case["n"], case["val_split"], case["test_split"], 42)
        assert [len(p) for p in parts] == case["sizes"], name
        assert [fx.split_index_sha256(p) for p in parts] == case["sha256"], name
        for split, part in zip(SPLITS, parts, strict=True):
            if split in case:
                np.testing.assert_array_equal(part, case[split], err_msg=name)
    stachyose = fx.load_split_golden()["n27272_v10_t10"]
    assert stachyose["sizes"][0] == 21818  # pandas round(), not int() truncation


def test_split_properties() -> None:
    train, val, test = split_indices(1000, 0.1, 0.1, 7)
    assert len(np.intersect1d(train, val)) == len(np.intersect1d(train, test)) == 0
    np.testing.assert_array_equal(np.sort(np.concatenate([train, val, test])), np.arange(1000))
    assert np.all(np.diff(test) > 0)
    assert not np.all(np.diff(train) > 0)  # sampling order, not sorted
    other = split_indices(1000, 0.1, 0.1, 8)
    assert not np.array_equal(train, other[0])
    empty = split_indices(0, 0.1, 0.1, 42)
    assert all(p.shape == (0,) for p in empty)
    no_eval = split_indices(10, 0.0, 0.0, 42)
    assert (len(no_eval[0]), len(no_eval[1]), len(no_eval[2])) == (10, 0, 0)


@pytest.mark.parametrize(("n", "v", "t"), [(-1, 0.1, 0.1), (5, -0.1, 0.1), (5, 0.5, 0.5)])
def test_split_rejects_invalid_input(n: int, v: float, t: float) -> None:
    with pytest.raises(ValueError):
        split_indices(n, v, t, 42)


# --------------------------------------------------------------------------- 2. layout


def test_layout_and_metadata(smoke_dir: Path, smoke: DescriptorDataset) -> None:
    meta = json.loads((smoke_dir / "metadata.json").read_text())
    assert meta == smoke.metadata
    for key in (
        "format_version",
        "lrmlp_version",
        "created",
        "sources",
        "grid",
        "n_max_local",
        "n_max_long_range",
        "species",
        "splits",
        "split_seed",
        "units",
        "val_fraction",
        "test_fraction",
    ):
        assert key in meta, key
    assert meta["format_version"] == FORMAT_VERSION
    assert meta["species"] == [1, 6, 8]
    grid = meta["grid"]
    assert grid["assignment"] == "thesis"
    assert grid["cutoff"] == smoke.grid.cutoff == (2 + 1 + 0.5**0.5) * 1.0
    assert grid["inner_cutoff"] == smoke.grid.inner_cutoff
    (source,) = meta["sources"]
    assert source["sha256"] == fx.load_manifest()["files"]["synthetic_small.npz"]["sha256"]
    assert (source["n_frames_used"], source["frame_offset"], source["n_atoms"]) == (60, 0, 21)
    assert meta["units"]["energy"] == "kcal/mol"
    assert {s: meta["splits"][s]["n_molecules"] for s in SPLITS} == {
        "train": 48,
        "val": 6,
        "test": 6,
    }
    for name in SPLITS:
        split = smoke[name]
        assert sorted(p.name for p in (smoke_dir / name).iterdir()) == sorted(
            f"{k}.npy" for k in ARRAY_DTYPES
        )
        a, s = split.n_atoms, len(split)
        n_loc, n_lr = smoke.n_max_local, smoke.n_max_long_range
        expected = {
            "energies": (s,),
            "atom_offsets": (s + 1,),
            "sample_index": (s,),
            "atomic_numbers": (a,),
            "forces": (a, 3),
            "local_matrix": (a, n_loc, 4),
            "local_numbers": (a, n_loc, 2),
            "long_range_matrix": (a, n_lr, 4),
            "long_range_counts": (a, n_lr, 3),
        }
        for key, shape in expected.items():
            arr = split.arrays[key]
            assert arr.shape == shape, key
            assert arr.dtype == ARRAY_DTYPES[key], key
            if arr.size:
                assert isinstance(arr, np.memmap), key
        assert meta["splits"][name] == {"n_molecules": s, "n_atoms": a}
    train, val, test = split_indices(60, 0.1, 0.1, 42)
    for name, idx in zip(SPLITS, (train, val, test), strict=True):
        np.testing.assert_array_equal(smoke[name].sample_index, idx)
    assert not [p for p in smoke_dir.parent.iterdir() if p.name.startswith(".")]


def test_existing_target_raises_unless_overwrite(
    smoke_config: ExperimentConfig, tmp_path: Path
) -> None:
    target = build_dataset(smoke_config, output_dir=tmp_path / "ds", n_workers=1)
    created = load_dataset(target).metadata["created"]
    with pytest.raises(FileExistsError):
        build_dataset(smoke_config, output_dir=target, n_workers=1)
    assert load_dataset(target).metadata["created"] == created
    other = smoke_config.with_overrides({"data.split_seed": 1})
    build_dataset(other, output_dir=target, n_workers=1, overwrite=True)
    assert load_dataset(target).metadata["split_seed"] == 1
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "keep.txt").write_text("x")
    with pytest.raises(FileExistsError, match="not a dataset"):
        build_dataset(smoke_config, output_dir=foreign, n_workers=1, overwrite=True)
    assert (foreign / "keep.txt").is_file()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ds", "foreign"]


def test_default_output_dir_from_config(smoke_config: ExperimentConfig, tmp_path: Path) -> None:
    config = smoke_config.with_overrides({"data.output_dir": str(tmp_path / "x" / "y")})
    assert build_dataset(config, n_workers=1) == tmp_path / "x" / "y"


def test_load_rejects_unknown_format(smoke_dir: Path, tmp_path: Path) -> None:
    import shutil

    copy = tmp_path / "copy"
    shutil.copytree(smoke_dir, copy)
    meta = json.loads((copy / "metadata.json").read_text())
    (copy / "metadata.json").write_text(json.dumps({**meta, "format_version": 99}))
    with pytest.raises(ValueError, match="format_version"):
        load_dataset(copy)
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path)
    (copy / "metadata.json").write_text(json.dumps(meta))
    (copy / "val" / "forces.npy").unlink()
    with pytest.raises(ValueError, match="forces"):
        load_dataset(copy)


def test_multiple_sources_offsets_and_clipping(mixed: Path) -> None:
    ds = load_dataset(mixed)
    sources = ds.metadata["sources"]
    assert [(s["n_frames_used"], s["n_frames_available"]) for s in sources] == [(10, 60), (5, 5)]
    assert [s["frame_offset"] for s in sources] == [0, 10]
    assert ds.metadata["species"] == [1, 6, 8]
    small, stach = _raw("synthetic_small"), _raw("stachyose_like")
    all_index = np.sort(np.concatenate([ds[s].sample_index for s in SPLITS]))
    np.testing.assert_array_equal(all_index, np.arange(15))
    for name in SPLITS:
        split = ds[name]
        for k, frame in enumerate(split.sample_index):
            raw, f = (small, frame) if frame < 10 else (stach, frame - 10)
            mol = split.molecule(k)
            assert mol["forces"].shape == (len(raw["z"]), 3)
            np.testing.assert_array_equal(mol["atomic_numbers"], raw["z"])
            np.testing.assert_array_equal(mol["forces"], raw["F"][f].astype(np.float32))
            assert mol["energy"] == raw["E"][f]


# --------------------------------------------------------------------------- 3. workers


def test_result_independent_of_n_workers(
    smoke_config: ExperimentConfig, smoke_dir: Path, tmp_path: Path
) -> None:
    other = build_dataset(smoke_config, output_dir=tmp_path / "w2", n_workers=2)
    a, b = load_dataset(smoke_dir), load_dataset(other)
    for name in SPLITS:
        for key in ARRAY_DTYPES:
            x, y = a[name].arrays[key], b[name].arrays[key]
            assert x.dtype == y.dtype and x.shape == y.shape, key
            assert np.asarray(x).tobytes() == np.asarray(y).tobytes(), (name, key)
    meta_a = {k: v for k, v in a.metadata.items() if k != "created"}
    meta_b = {k: v for k, v in b.metadata.items() if k != "created"}
    assert meta_a == meta_b


# --------------------------------------------------------------------------- 4./5. alignment


@pytest.mark.parametrize("name", SPLITS)
def test_forces_aligned_with_rows(smoke: DescriptorDataset, name: str) -> None:
    """B01: row a of molecule k and forces[a] belong to input atom a of its frame."""
    raw = _raw()
    split = smoke[name]
    for k in range(len(split)):
        frame = int(split.sample_index[k])
        desc = compute_descriptors(raw["R"][frame], raw["z"], smoke.grid)
        mol = split.molecule(k)
        lo, hi = split.atom_offsets[k : k + 2]
        assert hi - lo == 21
        for a in (0, 7, 20):
            atom = desc.atom(a)
            np.testing.assert_array_equal(
                mol["local_matrix"][a, : len(atom.local_matrix)],
                atom.local_matrix.astype(np.float32),
            )
            np.testing.assert_array_equal(
                mol["long_range_matrix"][a, : len(atom.long_range_matrix)],
                atom.long_range_matrix.astype(np.float32),
            )
        np.testing.assert_array_equal(mol["forces"], raw["F"][frame].astype(np.float32))
        np.testing.assert_array_equal(mol["atomic_numbers"], raw["z"])
        assert mol["energy"] == raw["E"][frame]


@pytest.mark.parametrize("name", SPLITS)
def test_reconstruction_equals_pad_descriptors(smoke: DescriptorDataset, name: str) -> None:
    raw = _raw()
    split = smoke[name]
    for k in range(len(split)):
        frame = int(split.sample_index[k])
        desc = compute_descriptors(raw["R"][frame], raw["z"], smoke.grid)
        expected = pad_descriptors(desc, smoke.n_max_local, smoke.n_max_long_range)
        mol = split.molecule(k)
        for key, value in expected.items():
            assert mol[key].dtype == value.dtype, key
            np.testing.assert_array_equal(mol[key], value, err_msg=f"{name}[{k}] {key}")


def test_reconstruction_needs_counts_not_inverse_distance(tmp_path: Path) -> None:
    """A real long-range row with 1/r = 0 still carries z_i (validity from the counts)."""
    from lrmlp.dataset import ARRAY_DTYPES as dtypes

    d = tmp_path / "s"
    d.mkdir()
    arrays = {
        "energies": np.zeros(1),
        "atom_offsets": np.array([0, 1]),
        "sample_index": np.array([0]),
        "atomic_numbers": np.array([6]),
        "forces": np.zeros((1, 3)),
        "local_matrix": np.zeros((1, 1, 4)),
        "local_numbers": np.zeros((1, 1, 2)),
        "long_range_matrix": np.zeros((1, 2, 4)),
        "long_range_counts": np.array([[[2, 1], [0, 0]]]),
    }
    for key, value in arrays.items():
        np.save(d / f"{key}.npy", value.astype(dtypes[key]))
    split = DescriptorSplit(d, "x", [1, 8], 100)
    features = split.molecule(0)["long_range_features"]
    assert features.shape == (1, 2, 101)
    np.testing.assert_array_equal(features[0, 0, [0, 1, 8]], [6, 2, 1])
    assert features[0, 0].sum() == 9 and features[0, 1].sum() == 0
    with pytest.raises(IndexError):
        split.molecule(1)


def test_long_range_counts_size(smoke: DescriptorDataset) -> None:
    """Memory sanity: the compact F_i needs A·n_lr·n_species bytes."""
    n_species = len(smoke.metadata["species"])
    for name in SPLITS:
        split = smoke[name]
        counts = split.arrays["long_range_counts"]
        assert counts.nbytes == split.n_atoms * smoke.n_max_long_range * n_species
        path = split.path / "long_range_counts.npy"
        with path.open("rb") as fh:
            version = np.lib.format.read_magic(fh)
            if version == (1, 0):
                np.lib.format.read_array_header_1_0(fh)
            else:
                np.lib.format.read_array_header_2_0(fh)
            header = fh.tell()
        assert path.stat().st_size - header == counts.nbytes


# --------------------------------------------------------------------------- 6.-8. tf.data


def _energies(ds: Any) -> np.ndarray:
    return np.array([float(y["energy"][0]) for _, y in ds])


def test_tf_element_spec_and_values(smoke: DescriptorDataset) -> None:
    import tensorflow as tf

    split = smoke["val"]
    ds = make_tf_dataset(split, shuffle=False, shuffle_buffer=1000, seed=None)
    inputs, targets = ds.element_spec
    n_loc, n_lr = smoke.n_max_local, smoke.n_max_long_range
    assert inputs == {
        "local_matrix": tf.TensorSpec((None, n_loc, 4), tf.float32),
        "local_numbers": tf.TensorSpec((None, n_loc, 2), tf.float32),
        "long_range_matrix": tf.TensorSpec((None, n_lr, 4), tf.float32),
        "long_range_features": tf.TensorSpec((None, n_lr, 101), tf.float32),
    }
    assert targets == {
        "energy": tf.TensorSpec((1,), tf.float32),
        "forces": tf.TensorSpec((None, 3), tf.float32),
    }
    count = 0
    for k, (x, y) in enumerate(ds):
        mol = split.molecule(k)
        for key in x:
            np.testing.assert_array_equal(x[key].numpy(), mol[key].astype(np.float32), key)
        np.testing.assert_array_equal(y["forces"].numpy(), mol["forces"])
        assert y["energy"].numpy()[0] == np.float32(mol["energy"])
        count += 1
    assert count == len(split)

    local_only = make_tf_dataset(
        split, shuffle=False, shuffle_buffer=1, seed=None, include_long_range=False
    )
    assert set(local_only.element_spec[0]) == {"local_matrix", "local_numbers"}
    x, _ = next(iter(local_only))
    np.testing.assert_array_equal(x["local_matrix"].numpy(), split.molecule(0)["local_matrix"])


def test_shuffle_deterministic_with_seed(smoke: DescriptorDataset) -> None:
    """B11: seeded shuffling is reproducible and differs between epochs."""
    split = smoke["train"]
    stored = split.energies.astype(np.float32)
    ds = make_tf_dataset(split, shuffle=True, shuffle_buffer=1000, seed=3)
    epoch1, epoch2 = _energies(ds), _energies(ds)
    again = make_tf_dataset(split, shuffle=True, shuffle_buffer=1000, seed=3)
    np.testing.assert_array_equal(epoch1, _energies(again))
    assert not np.array_equal(epoch1, epoch2)
    assert not np.array_equal(epoch1, stored)
    np.testing.assert_array_equal(np.sort(epoch1), np.sort(stored))
    other = _energies(make_tf_dataset(split, shuffle=True, shuffle_buffer=1000, seed=4))
    assert not np.array_equal(epoch1, other)
    for buffer in (0, 1):
        unshuffled = make_tf_dataset(split, shuffle=True, shuffle_buffer=buffer, seed=3)
        np.testing.assert_array_equal(_energies(unshuffled), stored)


def test_eval_pipeline_not_shuffled(smoke: DescriptorDataset) -> None:
    """B19: validation / test pipelines keep the stored order in every epoch."""
    for name in ("val", "test"):
        split = smoke[name]
        ds = make_tf_dataset(split, shuffle=False, shuffle_buffer=1000, seed=1)
        for _ in range(2):
            np.testing.assert_array_equal(_energies(ds), split.energies.astype(np.float32))
        batched = make_batched_tf_dataset(split, 4)
        for _ in range(2):
            got = np.concatenate([y["energy"].numpy() for _, y in batched])
            np.testing.assert_array_equal(got, split.energies.astype(np.float32))


@pytest.mark.parametrize("per_batch", [1, 5, 48, 100])
def test_batched_pipeline_segments(smoke: DescriptorDataset, per_batch: int) -> None:
    import tensorflow as tf

    split = smoke["train"]
    ds = make_batched_tf_dataset(split, per_batch)
    inputs, targets = ds.element_spec
    assert inputs["segment_ids"] == tf.TensorSpec((None,), tf.int32)
    assert targets["energy"] == tf.TensorSpec((None,), tf.float32)
    k = 0
    n_batches = 0
    for x, y in ds:
        b = int(y["energy"].shape[0])
        assert b == min(per_batch, len(split) - k)
        seg = x["segment_ids"]
        lo, hi = int(split.atom_offsets[k]), int(split.atom_offsets[k + b])
        assert seg.shape[0] == hi - lo == x["local_matrix"].shape[0]
        np.testing.assert_array_equal(np.diff(seg.numpy()) >= 0, True)
        summed = tf.math.unsorted_segment_sum(y["forces"], seg, b).numpy()
        expected = np.add.reduceat(
            np.asarray(split.arrays["forces"][lo:hi]),
            np.asarray(split.atom_offsets[k : k + b]) - lo,
        )
        # float32 sums in another order (TF segment sum vs NumPy reduceat); the LJ
        # forces of a molecule sum to ~0, so atol binds: observed max |Δ| 6e-7 on
        # macOS arm64. A row in the wrong segment moves a sum by that atom's
        # force, >= 2.8e-4 per component here (median 1.4e-2).
        np.testing.assert_allclose(summed, expected, rtol=1e-5, atol=2e-5)
        features = tf.math.unsorted_segment_sum(
            tf.reduce_sum(x["long_range_features"], axis=[1, 2]), seg, b
        ).numpy()
        ref = [split.molecule(j)["long_range_features"].sum() for j in range(k, k + b)]
        np.testing.assert_allclose(features, ref, rtol=1e-6)
        np.testing.assert_array_equal(
            y["energy"].numpy(), split.energies[k : k + b].astype(np.float32)
        )
        k += b
        n_batches += 1
    assert k == len(split)
    assert n_batches == -(-len(split) // per_batch)
    with pytest.raises(ValueError):
        make_batched_tf_dataset(split, 0)
    local_only = make_batched_tf_dataset(split, per_batch, include_long_range=False)
    assert set(local_only.element_spec[0]) == {"local_matrix", "local_numbers", "segment_ids"}


def test_mixed_molecules_through_pipelines(mixed: Path) -> None:
    split = load_dataset(mixed)["train"]
    sizes = {int(n) for n in np.diff(split.atom_offsets)}
    assert sizes == {21, 87}
    for k, (x, _) in enumerate(make_tf_dataset(split, shuffle=False, shuffle_buffer=1, seed=0)):
        assert x["local_matrix"].shape[0] == split.atom_offsets[k + 1] - split.atom_offsets[k]
    ((x, y),) = make_batched_tf_dataset(split, len(split))
    assert x["segment_ids"].shape[0] == split.n_atoms


# --------------------------------------------------------------------------- 9. no pandas


def test_no_pandas_import(tmp_path: Path) -> None:
    code = textwrap.dedent(
        f"""
        import sys
        from lrmlp.config import load_config
        from lrmlp.dataset import build_dataset, load_dataset
        path = build_dataset(load_config({str(SMOKE)!r}), output_dir={str(tmp_path / "d")!r},
                             n_workers=2)
        ds = load_dataset(path)
        ds["train"].molecule(0)
        assert "pandas" not in sys.modules, "pandas imported"
        print("ok")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO, timeout=300
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


# ------------------------------------------------------------- permissions, errors, merging


@pytest.mark.parametrize("umask", [0o022, 0o027])
def test_dataset_directory_has_default_mode(
    smoke_config: ExperimentConfig, tmp_path: Path, umask: int
) -> None:
    """The dataset directory gets ``0o777 & ~umask`` like a plain mkdir, not 0700."""
    previous = os.umask(umask)
    try:
        target = build_dataset(smoke_config, output_dir=tmp_path / "ds", n_workers=1)
        plain = tmp_path / "plain"
        plain.mkdir()
        assert os.umask(umask) == umask
    finally:
        os.umask(previous)
    assert stat.S_IMODE(target.stat().st_mode) == 0o777 & ~umask
    assert stat.S_IMODE(target.stat().st_mode) == stat.S_IMODE(plain.stat().st_mode)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ds", "plain"]


@pytest.mark.parametrize("n_workers", [1, 2])
def test_frame_errors_name_source_and_frame(
    smoke_config: ExperimentConfig, tmp_path: Path, n_workers: int
) -> None:
    """A defective frame is reported with its source and frame index."""
    raw = dict(_raw())
    positions = raw["R"].copy()
    positions[37, 5] = positions[37, 4]  # two coincident atoms (B12)
    raw["R"] = positions
    bad = tmp_path / "bad.npz"
    np.savez(bad, **raw)
    config = smoke_config.with_overrides(
        {"data.sources": [str(FIXTURES / "synthetic_small.npz"), str(bad)]}
    )
    expected = rf"bad\.npz: frame 37 \(global frame {60 + 37}\): .*coincide"
    with pytest.raises(ValueError, match=expected):
        build_dataset(config, output_dir=tmp_path / "ds", n_workers=n_workers)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["bad.npz"]


def _random_ragged(rng: np.random.Generator, atoms: int, n_species: int) -> Any:
    from lrmlp.dataset import _Ragged

    n_local = rng.integers(0, 6, size=atoms)
    n_long = rng.integers(0, 4, size=atoms)
    k, m = int(n_local.sum()), int(n_long.sum())
    return _Ragged(
        local_matrix=rng.normal(size=(k, 4)).astype(np.float32),
        local_numbers=rng.integers(0, 255, size=(k, 2), dtype=np.uint8),
        n_local=n_local.astype(np.int64),
        long_range_matrix=rng.normal(size=(m, 4)).astype(np.float32),
        long_range_counts=rng.integers(0, 255, size=(m, n_species), dtype=np.uint8),
        n_long=n_long.astype(np.int64),
    )


def _field_arrays(ragged: Any) -> list[np.ndarray]:
    return [getattr(ragged, f.name) for f in dataclasses.fields(ragged)]


def test_ragged_concatenate_equals_numpy_concatenate() -> None:
    """The preallocated merge gives exactly ``np.concatenate`` of the parts."""
    from lrmlp.dataset import _Ragged

    rng = np.random.default_rng(3)
    parts = [_random_ragged(rng, atoms, 3) for atoms in (5, 0, 17, 1, 40)]
    expected = {
        f.name: np.concatenate([getattr(p, f.name) for p in parts])
        for f in dataclasses.fields(_Ragged)
    }
    merged = _Ragged.concatenate(parts, 3)
    assert parts == []  # the parts are consumed
    for name, value in expected.items():
        array = getattr(merged, name)
        assert array.dtype == value.dtype and array.shape == value.shape, name
        assert array.tobytes() == value.tobytes(), name
    empty = _Ragged.concatenate([], 3)
    assert empty.long_range_counts.shape == (0, 3) and empty.n_local.dtype == np.int64


def test_ragged_concatenate_does_not_double_memory() -> None:
    """Merging allocates at most the largest field on top of the parts."""
    from lrmlp.dataset import _Ragged

    rng = np.random.default_rng(5)
    tracemalloc.start()  # before the parts exist, so that releasing them is seen
    try:
        parts = [_random_ragged(rng, 2000, 3) for _ in range(20)]
        total = sum(sum(a.nbytes for a in _field_arrays(p)) for p in parts)
        largest = max(
            sum(getattr(p, name).nbytes for p in parts)
            for name in ("local_matrix", "local_numbers", "long_range_matrix", "long_range_counts")
        )
        tracemalloc.reset_peak()
        base, _ = tracemalloc.get_traced_memory()
        merged = _Ragged.concatenate(parts, 3)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert sum(a.nbytes for a in _field_arrays(merged)) == total
    # np.concatenate of all fields would need ``total`` extra bytes at the peak.
    assert peak - base < largest + 0.1 * total < total


def test_mixed_dataset_equals_direct_descriptors(mixed: Path) -> None:
    """Two sources, two workers; every stored molecule equals the direct result."""
    ds = load_dataset(mixed)
    raws = {0: _raw("synthetic_small"), 1: _raw("stachyose_like")}
    for name in SPLITS:
        split = ds[name]
        for k, frame in enumerate(split.sample_index):
            raw, f = (raws[0], frame) if frame < 10 else (raws[1], frame - 10)
            desc = compute_descriptors(raw["R"][f], raw["z"], ds.grid)
            expected = pad_descriptors(desc, ds.n_max_local, ds.n_max_long_range)
            mol = split.molecule(k)
            for key, value in expected.items():
                assert mol[key].dtype == value.dtype, key
                np.testing.assert_array_equal(mol[key], value, err_msg=f"{name}[{k}] {key}")


# --------------------------------------------------------------------------- BLAS threads


def test_workers_use_one_blas_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    """The spawned workers start with one BLAS/OpenMP thread each.

    The parent keeps its environment: a set variable gets its value back, an
    unset one stays unset.
    """
    from lrmlp.dataset import WORKER_THREAD_ENV, _worker_map

    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "7")
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    names = list(WORKER_THREAD_ENV) * 2
    assert _worker_map(os.getenv, names, 2) == ["1"] * len(names)
    assert os.environ["OPENBLAS_NUM_THREADS"] == "7"
    assert "OMP_NUM_THREADS" not in os.environ


def test_worker_environment_restored_after_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """The parent environment is restored when the workers fail."""
    from lrmlp.dataset import WORKER_THREAD_ENV, _worker_map

    for name in WORKER_THREAD_ENV:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError):
        _worker_map(int, ["1", "not a number"], 2)
    assert not any(name in os.environ for name in WORKER_THREAD_ENV)


def test_worker_environment_nested_blocks_restore_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Overlapping blocks (A enters, B enters, A exits, B exits) restore the parent values."""
    from lrmlp.dataset import WORKER_THREAD_ENV, _single_threaded_children

    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "4")
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    a, b = _single_threaded_children(), _single_threaded_children()
    a.__enter__()
    b.__enter__()
    a.__exit__(None, None, None)
    assert all(os.environ[name] == "1" for name in WORKER_THREAD_ENV)  # B still starting
    b.__exit__(None, None, None)
    assert os.environ["OPENBLAS_NUM_THREADS"] == "4"
    assert "OMP_NUM_THREADS" not in os.environ


def test_worker_environment_parallel_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Concurrent ``_worker_map`` calls from threads: workers see 1, the parent keeps its env."""
    import threading

    from lrmlp.dataset import WORKER_THREAD_ENV, _worker_map

    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "4")
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    names = list(WORKER_THREAD_ENV) * 2
    results: list[list[str | None]] = []
    errors: list[BaseException] = []
    barrier = threading.Barrier(3)

    def call() -> None:
        try:
            barrier.wait()
            results.append(_worker_map(os.getenv, names, 2))
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    for _ in range(2):
        threads = [threading.Thread(target=call) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=300)
        assert not errors, errors
        barrier.reset()
    assert len(results) == 6
    assert all(r == ["1"] * len(names) for r in results)
    assert os.environ["OPENBLAS_NUM_THREADS"] == "4"
    assert "OMP_NUM_THREADS" not in os.environ
