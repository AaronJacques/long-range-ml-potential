"""Golden regression of ``lrmlp.dataset.build_dataset`` against the legacy ``create_dataset``.

The fixtures (T02, ``data_dataset_<GA|GB>.npz``) hold the three legacy pickles:
split indices in stored order, energies, forces (input order) and the padded
arrays in legacy atom order; ``__order`` maps every legacy position to the
input atom index (B01).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from lrmlp.config import load_config
from lrmlp.dataset import SPLITS, DescriptorDataset, build_dataset, load_dataset

try:
    from golden import fixtures as fx
except ImportError:  # pragma: no cover - depends on pytest import mode
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from golden import fixtures as fx

pytestmark = pytest.mark.golden

CONFIG = Path(__file__).resolve().parent / "fixtures" / "configs" / "golden_legacy.toml"
#: golden name -> overrides of golden_legacy.toml (G-B)
CASES = {"GB": [], "GA": ["grid.max_local_level=6"]}


@pytest.fixture(scope="module", params=sorted(CASES))
def case(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> tuple[DescriptorDataset, dict[str, np.ndarray]]:
    name = request.param
    config = load_config(CONFIG, CASES[name])
    golden = fx.load_dataset_golden(name)
    assert config.grid.assignment == "legacy"
    assert config.grid.cutoff == float(golden["cut_off"])
    assert config.grid.inner_cutoff == float(golden["inner_cut_off"])
    path = build_dataset(config, output_dir=tmp_path_factory.mktemp(name) / name, n_workers=2)
    return load_dataset(path), golden


def test_n_max(case: tuple[DescriptorDataset, dict[str, np.ndarray]]) -> None:
    ds, golden = case
    assert ds.n_max_local == int(golden["n_max_local"])
    assert ds.n_max_long_range == int(golden["n_max_long_range"])


@pytest.mark.parametrize("split", SPLITS)
def test_split_energies_forces(
    case: tuple[DescriptorDataset, dict[str, np.ndarray]], split: str
) -> None:
    ds, golden = case
    s, p = ds[split], f"{split}__"
    np.testing.assert_array_equal(s.sample_index, golden[p + "index"])
    np.testing.assert_array_equal(s.energies, golden[p + "energy"])
    np.testing.assert_array_equal(s.atom_offsets, golden[p + "atom_offsets"])
    np.testing.assert_array_equal(s.arrays["forces"], golden[p + "forces"].astype(np.float32))


@pytest.mark.parametrize("split", SPLITS)
def test_padded_arrays(case: tuple[DescriptorDataset, dict[str, np.ndarray]], split: str) -> None:
    ds, golden = case
    s, p = ds[split], f"{split}__"
    offsets = s.atom_offsets
    for k in range(len(s)):
        lo, hi = int(offsets[k]), int(offsets[k + 1])
        order = golden[p + "order"][lo:hi]  # input index of every legacy position
        mol = s.molecule(k)
        msg = f"{split}[{k}]"
        for key in ("local_matrix", "long_range_matrix"):
            np.testing.assert_allclose(
                mol[key][order], golden[p + key][lo:hi], rtol=0, atol=1e-6, err_msg=msg
            )
        for key in ("local_numbers", "long_range_features"):
            np.testing.assert_array_equal(
                mol[key][order].astype(np.float32), golden[p + key][lo:hi], err_msg=msg
            )
        per_atom = golden[p + "n_max"][k]
        assert int((mol["local_matrix"][..., 0] != 0).sum(axis=1).max()) <= per_atom[0]
