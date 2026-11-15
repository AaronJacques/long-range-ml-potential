# Test fixtures

Deterministic synthetic inputs and golden fixtures frozen from the **unchanged
legacy code** (`Neural-Network/` at commit `9413bfd`). Later refactoring tasks
prove their equivalence against these files. Do not edit them by hand.

## Inputs (`tests/golden/synthetic.py`, NumPy only)

| File | Content |
|---|---|
| `synthetic_small.npz` | 60 frames, aspirin-like: z = C9H8O4 in fixed order `[6]*9 + [1]*8 + [8]*4` (21 atoms), box ~ 9 Å, `aspirin_like(60, seed=0)` |
| `stachyose_like.npz` | 5 frames, 87 atoms (C24H42O21), box ~ 16 Å, `stachyose_like(5, seed=1)` |
| `edge_cases.npz` | hand-built single molecules, keys `<case>__R` (N, 3) and `<case>__z` (N,) |

MD17 layout: `E` (S,) kcal/mol, `F` (S, N, 3) kcal/(mol·Å), `R` (S, N, 3) Å,
`z` (N,). Frames are a random reference geometry plus Gaussian displacements
(σ = 0.1 Å), positions rounded to 2⁻¹⁰ Å, all pair distances ≥ 1.0 Å, so all
positions are unique. Energy and forces are Lennard-Jones (ε = 0.1, σ = 1.2)
with analytic forces `F = -∇E`.

Edge cases (descriptions in `synthetic.edge_cases()`): `level_lmax2_within_rc`
(A01), `dropped_pairs` (A02), `partial_cell` (cell on level l_max+1 with atoms on
both sides of r_c), `single_cell`, `no_long_range`, `cell_boundary` (atoms exactly
on cell boundaries), `gb_assignment` (A01/A02 for G-B), `multi_element`.

## Golden configurations

Binding for all tasks. Always `MAX_ATOM_ELEMENTS = 100`, `use_charge_number = True`,
`CUT_OFF = (l_max + 1 + 0.5**0.5) * Δg`, `INNER_CUT_OFF = CUT_OFF - 0.2`.

| Name | Δg [Å] | l_max | r_c [Å] |
|---|---|---|---|
| G-A | 1.0 | 6 | 7.7071 |
| G-B | 1.0 | 2 | 3.7071 |
| G-C | 2.0 | 4 | 11.4142 |

## Golden files (`golden/`)

| File | Source |
|---|---|
| `data_descriptors_GA_aspirin.npz` | G-A, first 10 frames of `synthetic_small` |
| `data_descriptors_GB_aspirin.npz` | G-B, same 10 frames |
| `data_descriptors_GB_aspirin_sorted.npz` | G-B with `sort_matrix=True` |
| `data_descriptors_GA_edge.npz`, `data_descriptors_GB_edge.npz` | all edge cases, one frame each (`frame_names`) |
| `data_descriptors_GC_stachyose.npz` | G-C, first 3 frames of `stachyose_like` |
| `data_dataset_GA.npz`, `data_dataset_GB.npz` | `create_dataset([synthetic_small.npz], …)` → the three pickles (split 80/10/10, `random_state=42`) |
| `data_split.npz` | the legacy split expressions (`Dataset.py:475-477`, `df.sample(frac=…, random_state=42)`) on a `RangeIndex(n)` DataFrame for n ∈ {7, 13, 60, 61, 63, 65, 101, 27272, 65000} and (val, test) ∈ {(0.1, 0.1), (0.05, 0.15)}: sizes and sha256 per split, index arrays for n ≤ 1000 |
| `manifest_data.json` | legacy commit, versions, constants per config, per-case stats, sha256 of every file |

Descriptor goldens hold the raw output of `create_grid` + `create_matrices` per
frame; dataset goldens hold the padded float32 arrays, labels and split indices.
The key layout is documented in `tests/golden/fixtures.py`, which also provides
the loaders (`load_descriptor_golden`, `load_dataset_golden`, `split_ragged`,
`to_input_order`).

Legacy behaviour these files freeze (see also
[docs/methodology.md](../../docs/methodology.md#deviations-and-decisions)):

- Rows are in **cell traversal order** (cells lexicographic by grid index, atoms
  in input order within a cell), not input order. `order[k]` is the input index
  of legacy row `k`. Forces and positions stay in input order (bug B01).
- A02: atoms beyond r_c in cells on level ≤ l_max are dropped.
- A01: atoms inside r_c in cells on level ≥ l_max+2 enter only via the cell
  centroid.
- Split sizes follow pandas `round(frac * n)` (round half to even), not `int()`:
  for Stachyose (n = 27272) truncation would give 21817 instead of 21818 training
  frames; n = 63 and 65 are x.5 cases (`data_split.npz`).

### Model, training, schedule and evaluation goldens (`generate_model_golden.py`)

Model M-1 on dataset G-B (all 60 frames of `synthetic_small`, split 0.1/0.1/42,
`SHUFFLE_BUFFER_SIZE = 1`): embedding `[16, 32, 32]`, K1 = N_max = (10, 20),
K2 = (4, 1), ELU, weights from `clear_session()` + `set_random_seed(1234)`.

| File | Content |
|---|---|
| `model_lr_energy.npz`, `model_local_energy.npz`, `model_lr_forces.npz` | initial weights, the first 5 test molecules (padded inputs, `order`) and `model(x, training=False)` per molecule, `sum_energy`, `sum_all` (B05) |
| `model_lr_energy_k1.npz` | forward only: M-1 with explicit K1 = (6, 24), i.e. M1_local < n_max_local and M1_long > n_max_long_range; also holds `M1_local`, `M1_long` |
| `train_lr_energy.npz` | weights after one epoch of the unchanged `Training.start_training` (start = `model_lr_energy.npz`), epoch losses, `final_loss`, optimizer steps |
| `schedules.json` | `ExponentialDecay(staircase=True)` and `Training.get_weighting_from_step` at fixed steps |
| `eval_lr_energy.json` | test MAE of `evaluate.evaluate_model` on the `.h5` checkpoint of epoch 0, per-molecule absolute errors |
| `manifest_model.json` | legacy commit, versions, all parameters, layer tables (construction order, legacy names, weight shapes), training log, sha256 of every file |

Weights are stored as `{Class}__{k:03d}__{j}` (k = construction index within the
layer class = Keras 2 name suffix, j = index in `layer.get_weights()`). Loaders:
`load_model_golden`, `load_train_golden`, `load_schedules_golden`,
`load_eval_golden`, `load_model_manifest`, `weights_by_class` in
`tests/golden/fixtures.py`.

`.npz` files are written by `fixtures.save_npz` instead of `np.savez_compressed`.
The container is the same (zip of `.npy`, deflate, readable with `np.load`), but
`np.savez_compressed` stamps every zip entry with the current time, so identical
arrays would get new bytes and a new sha256 on every regeneration. `save_npz` uses
a fixed timestamp (1980-01-01), which keeps the bytes and the manifest sha256
reproducible.

## Regeneration

Needs a legacy env: Python 3.12 with TF 2.21, tf-keras and pandas
(`pip install -e ".[legacy]"`, e.g. `.venv`), run with `TF_USE_LEGACY_KERAS=1`. The legacy code is taken
from `git archive 9413bfd` into a temp dir and never imported from the working
tree.

```sh
G=.venv/bin/python   # any env with the extra "legacy"
TF_USE_LEGACY_KERAS=1 PYTHONDONTWRITEBYTECODE=1 $G tests/golden/generate_data_golden.py          # write
TF_USE_LEGACY_KERAS=1 PYTHONDONTWRITEBYTECODE=1 $G tests/golden/generate_data_golden.py --check  # OK: identical
TF_USE_LEGACY_KERAS=1 PYTHONDONTWRITEBYTECODE=1 $G tests/golden/generate_model_golden.py          # write
TF_USE_LEGACY_KERAS=1 PYTHONDONTWRITEBYTECODE=1 $G tests/golden/generate_model_golden.py --check  # OK: identical
```

`generate_model_golden.py` reads `synthetic_small.npz`, so run the data
generator first when regenerating both.

`--check` regenerates into a temp dir and compares all arrays bit for bit, plus
every manifest field except the environment `versions` and the file sha256
(differences in those are only reported as notes). The manifest has no
timestamp. `tests/test_golden_fixtures.py` and
`tests/test_golden_model_fixtures.py` check the committed files without the
legacy env.

**Platform.** Bit-identical regeneration (`--check`) is only guaranteed on the
platform that wrote the fixtures: macOS arm64 with the versions in the manifest
`versions`. On other platforms (e.g. Linux x86_64) floating-point results such as
the Lennard-Jones forces `F` can differ in the last bits (libm, summation order),
so `--check` may report differences there. The committed files are the
reference. `test_synthetic_inputs_reproducible_from_generator` therefore compares
the regenerated `E` and `F` with `rtol = atol = 1e-12` (shapes and dtypes exact),
while `R`, `z` and the edge cases are compared exactly; the sha256 checks of the
committed files are unaffected.
