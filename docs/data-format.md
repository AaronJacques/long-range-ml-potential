# Data formats

This page describes every file `lrmlp` reads or writes: the raw trajectories,
the preprocessed descriptor dataset, the run directory and the input and output
of `lrmlp predict`. The method behind the stored quantities is described in
[methodology.md](methodology.md). Thesis references ("p. n") use the printed page
numbers of the [thesis PDF](thesis/Masterarbeit__Machine_Learning_Potentials_with_Long_Range_Interaction.pdf)
(PDF page = printed page + 8).

Authoritative sources are the module docstrings of
[`lrmlp.io`](../src/lrmlp/io.py), [`lrmlp.dataset`](../src/lrmlp/dataset.py),
[`lrmlp.run`](../src/lrmlp/run.py), [`lrmlp.training`](../src/lrmlp/training.py),
[`lrmlp.evaluation`](../src/lrmlp/evaluation.py) and
[`lrmlp.potential`](../src/lrmlp/potential.py). If this page and the code
disagree, the code is right and this page is out of date.

Units are those of the thesis data (Sec. 4.3.1, p. 40): positions in Å, energies
in kcal/mol, forces in kcal/(mol·Å).

## Contents

- [Raw trajectories (`.npz`)](#raw-trajectories-npz)
- [Preprocessed dataset](#preprocessed-dataset)
  - [`metadata.json`](#metadatajson)
  - [Arrays per split](#arrays-per-split)
  - [Atom order](#atom-order)
  - [Rebuilding the 101 columns of F_i](#rebuilding-the-101-columns-of-f_i)
  - [Split semantics](#split-semantics)
- [Run directory](#run-directory)
  - [`training_log.csv`](#training_logcsv)
  - [`best.json` and checkpoints](#bestjson-and-checkpoints)
  - [Evaluation results](#evaluation-results)
- [Prediction input and output](#prediction-input-and-output)
- [Legacy `.pkl.gzip` datasets](#legacy-pklgzip-datasets)

## Raw trajectories (`.npz`)

Read by `lrmlp.io.load_trajectory`. The thesis uses the datasets of [16]
(sGDML, <http://www.sgdml.org>, Sec. 4.3.1, p. 40, Tab. 4.1). The loader expects
their `.npz` layout (the same keys as the MD17 files):

| Key | Shape | Unit | Meaning |
|---|---|---|---|
| `E` | (S,) or (S, 1) | kcal/mol | total energy of every frame; (S, 1) is flattened |
| `F` | (S, N, 3) | kcal/(mol·Å) | force on every atom (not used for training in the thesis, p. 40) |
| `R` | (S, N, 3) | Å | atom positions |
| `z` | (N,) | – | atomic numbers (integers ≥ 1), identical for every frame |

S is the number of frames, N the number of atoms. All four keys are required;
further keys (`name`, `theory`, `md5`, …) are ignored. Pickled objects are never
loaded (`allow_pickle=False`). Shapes are checked: `F` and `R` must be
(S, N, 3) with the S of `E` and the N of `z`.

Only the first `data.n_samples_per_source` frames of every source are read
(thesis: at most the first 65 000 frames, p. 40, Tab. 4.1). The value is an
integer upper bound or `"all"`; a source with fewer frames contributes all of
them.

## Preprocessed dataset

Written by `lrmlp prepare` (`lrmlp.dataset.build_dataset`), read by
`lrmlp.dataset.load_dataset`. Preprocessing runs once; the result is reused by
every training run (p. 40). Default location `data/processed/<name>/`
(`data.output_dir`):

```text
data/processed/<name>/
├── metadata.json
├── train/   *.npy
├── val/     *.npy
└── test/    *.npy
```

The dataset is built in a temporary sibling directory and renamed at the end,
so an interrupted build leaves no partial dataset. An existing target is only
replaced with `overwrite=True`. `format_version` is currently `1`.

Building steps (from the module docstring of `lrmlp.dataset`):

1. The first `n_samples_per_source` frames of every source are read and
   concatenated in source order.
2. The descriptors of every frame are computed
   (`lrmlp.descriptors.compute_descriptors`, see
   [methodology.md](methodology.md#descriptors)), in worker processes if
   `data.n_workers` ≠ 1. The result does not depend on the number of workers.
3. N_max^local and N_max^long are the maxima over **all** frames, before the
   split (p. 44).
4. The frames are split (see [Split semantics](#split-semantics)) and every
   split is written.

Every atomic number must be ≤ `grid.max_atomic_number` (and ≤ 255, the range of
the `uint8` arrays); otherwise `build_dataset` raises `ValueError`. For the same
reason a long-range cell can hold at most 255 atoms of one element in the stored
counts.

### `metadata.json`

| Field | Type | Meaning |
|---|---|---|
| `format_version` | int | layout version of this page (`1`) |
| `lrmlp_version` | str | package version that wrote the dataset |
| `created` | str | creation time, UTC, ISO 8601 (seconds) |
| `name` | str | experiment name of the config |
| `sources` | list of objects | one entry per source, in concatenation order (below) |
| `n_samples_per_source` | int or `"all"` | value of `data.n_samples_per_source` (an upper bound) |
| `grid` | object | complete grid configuration: `grid_size`, `max_local_level`, `inner_cutoff_offset`, `max_atomic_number`, `assignment`, `sort_neighbours`, plus the derived `cutoff` ($r_c$, Eq. 4.14) and `inner_cutoff` ($r_{cs}$) in Å |
| `max_atomic_number` | int | $m_{max}$; F_i has `max_atomic_number + 1` columns |
| `n_max_local` | int | $N_{max}^{local}$, padded local rows per atom (p. 44) |
| `n_max_long_range` | int | $N_{max}^{long}$, padded long-range rows per atom (p. 44) |
| `species` | list of int | sorted distinct atomic numbers of all sources; order of the columns of `long_range_counts` |
| `n_samples` | int | total number of frames over all sources |
| `splits` | object | per split (`train`, `val`, `test`): `{"n_molecules": …, "n_atoms": …}` |
| `val_fraction`, `test_fraction` | float | split fractions (default 0.1, 0.1) |
| `split_seed` | int | seed of the split (default 42) |
| `units` | object | `{"energy": "kcal/mol", "forces": "kcal/(mol*Angstrom)", "positions": "Angstrom"}` |
| `arrays` | object | dtype name of every array file (table below) |

Fields of every `sources` entry:

| Field | Type | Meaning |
|---|---|---|
| `path` | str | path of the `.npz` file as given in the config (resolved) |
| `sha256` | str | SHA-256 of the file |
| `n_frames_available` | int | frames in the file |
| `n_frames_used` | int | leading frames taken from it |
| `frame_offset` | int | global index of its first frame (frames taken from the sources before it) |
| `n_atoms` | int | atoms per frame |

### Arrays per split

Each split directory holds uncompressed `.npy` files that are memory-mapped on
loading. S = molecules in the split, A = atoms in the split (molecules
concatenated in split order), N_loc = `n_max_local`, N_lr = `n_max_long_range`,
n_sp = `len(species)`:

| File | Shape | Dtype | Meaning |
|---|---|---|---|
| `energies.npy` | (S,) | float64 | total energy $E_k$, kcal/mol |
| `atom_offsets.npy` | (S + 1,) | int64 | atoms of molecule k are rows `atom_offsets[k]:atom_offsets[k+1]` of the per-atom arrays |
| `sample_index.npy` | (S,) | int64 | global frame index: frame f of source s has index `frame_offset_s + f` |
| `atomic_numbers.npy` | (A,) | uint8 | $z_i$ |
| `forces.npy` | (A, 3) | float32 | force label of atom i, kcal/(mol·Å) |
| `local_matrix.npy` | (A, N_loc, 4) | float32 | $\tilde R_i^{local}$, rows $(s, s\,x/r, s\,y/r, s\,z/r)$ (Eq. 4.4-4.6) |
| `local_numbers.npy` | (A, N_loc, 2) | uint8 | $Z_i$, rows $(z_i, z_j)$ (Eq. 4.15) |
| `long_range_matrix.npy` | (A, N_lr, 4) | float32 | $\tilde R_i^{long}$, rows $(1/r, x/r, y/r, z/r)$ (Eq. 4.16-4.18) |
| `long_range_counts.npy` | (A, N_lr, n_sp) | uint8 | element counts $f$ of each long-range cell, column s = element `species[s]` |

Rows beyond the real neighbours or cells of an atom are zero (zero padding,
p. 44). The row order within an atom is described in
[methodology.md](methodology.md#descriptors). The sign convention of the
relative vectors ($r_i - r_j$ instead of $r_j - r_i$) is deviation
[A08](methodology.md#deviations-and-decisions).

### Atom order

Atoms are **always in input order**: per-atom row `atom_offsets[k] + a` belongs
to atom a of the source frame, and so does `forces[atom_offsets[k] + a]`. The
legacy code emitted the descriptor rows in grid-cell traversal order while the
force labels stayed in input order (bug B01); this is fixed.

### Rebuilding the 101 columns of F_i

The thesis feeds the network the feature matrix
$F_i \in \mathbb{N}^{N_{max}^{long} \times (m_{max}+1)}$ with rows
$(z_i, (f_j)_1, \dots, (f_j)_{m_{max}})$ (Eq. 4.19-4.20, p. 44). With
$m_{max} = 100$ that is 101 columns, almost all zero; for Stachyose the full
matrix would take about 26.8 GB. It is therefore **not stored**. The model input
`long_range_features` (shape (A, N_lr, max_atomic_number + 1), float32) is
rebuilt on the fly:

1. A row is *real* if its `long_range_counts` are not all zero. Every long-range
   cell holds at least one atom, so this is exact. (Testing
   `long_range_matrix[..., 0] > 0` would miss a cell whose centre of mass
   coincides with the atom: that row is encoded as `[0, 0, 0, 0]`, as in the
   legacy code.)
2. Column 0 of every real row is $z_i$ (`atomic_numbers` of the atom).
3. Column Z (1 ≤ Z ≤ `max_atomic_number`) holds the count of element Z, i.e.
   column `species[s]` takes `long_range_counts[..., s]`.
4. All other columns, and all columns of padding rows, are zero.

The result equals `long_range_features` of
`lrmlp.descriptors.pad_descriptors`. `compute_descriptors(..., species=...)`
returns the same compact counts (one column per species) that the dataset
stores; without `species` it returns full counts with column Z − 1 = element Z.

### Split semantics

`lrmlp.dataset.split_indices` reproduces, without pandas, exactly the split of
the legacy code (`Dataset.py:475-477` at commit `9413bfd`):

```python
train = df.sample(frac=1 - (v + t), random_state=seed)
val = df.drop(train.index).sample(frac=v / (v + t), random_state=seed)
test = df.drop(train.index).drop(val.index)
```

with v = `val_fraction`, t = `test_fraction`, seed = `split_seed` (default
0.1 / 0.1 / 42, the 80/10/10 random split of p. 40).
`df.sample(frac=f, random_state=seed)` on n rows takes `round(f * n)` rows
(Python `round`, half to even) at the positions
`numpy.random.RandomState(seed).permutation(n)[:size]`. The validation sample is
drawn from the remaining frames in ascending order with a fresh generator.
`train` and `val` keep the sampling order, `test` is ascending; the arrays of a
split are written in that order (`sample_index` records it). For v + t = 0,
validation and test are empty.

The split is over frames of a trajectory. Consecutive MD frames are correlated,
so test frames have close neighbours in time in the training split; the thesis
does the same (p. 40, deviation [A16](methodology.md#deviations-and-decisions)).

## Run directory

Created by `lrmlp train` (`lrmlp.training.train`) as
`<train.run_root>/<name>-YYYYmmdd-HHMMSS/`; if that exists, `-1`, `-2`, … is
appended. Existing run directories are never overwritten.

```text
runs/<name>-YYYYmmdd-HHMMSS/
├── config.json                      resolved configuration
├── dataset.json                     dataset metadata + "path"
├── training_log.csv                 one row per epoch
├── tb/                              TensorBoard event files
├── checkpoints/epoch_0001.weights.h5
├── best.json                        {"epoch": …, "val_energy_mse": …}
└── evaluation_<split>_<epoch>.json  written by lrmlp evaluate
```

All JSON files are UTF-8 with sorted keys and are written atomically
(temporary file + rename).

- `config.json`: `ExperimentConfig.to_dict()` of the resolved configuration:
  sections `grid`, `data`, `model`, `train` and `name`; K1 = `"auto"` is
  replaced by the integer N_max of the dataset; `derived` holds `cutoff` and
  `inner_cutoff`.
- `dataset.json`: a copy of the dataset's `metadata.json` plus `path` (absolute
  dataset directory). `lrmlp evaluate` and `lrmlp predict` take the padding
  N_max and the dataset location from it.
- `tb/`: TensorBoard scalars of every log column per epoch and
  `step/train_loss` every `train.log_every_steps` steps.

### `training_log.csv`

One row per epoch (epochs are counted from 1), written with `csv` and flushed
after every epoch. Values are Python `repr` of floats. There are no NaN columns:
a column exists only in the mode where it has a value.

| Column | Unit | Meaning |
|---|---|---|
| `epoch` | – | epoch number, from 1 |
| `train_loss` | (kcal/mol)² | mean optimised loss over the steps of the epoch (= energy MSE without forces) |
| `train_energy_mse` | (kcal/mol)² | mean over the steps of $(\sum_i E_i - E)^2$ (Eq. 4.21 with $\lvert B\rvert = 1$) |
| `val_loss` | (kcal/mol)² | validation loss (= `val_energy_mse` without forces) |
| `val_energy_mse` | (kcal/mol)² | mean squared molecule-energy error on the validation split; selects the best epoch (p. 52) |
| `learning_rate` | as `train.learning_rate` | Adam step size $\eta(t)$ (Eq. 4.22) at the **end** of the epoch, t = optimizer steps so far, i.e. the rate of the next step (with the stepwise decay it can differ from the rate used during the epoch) |
| `seconds` | s | wall time of the epoch including validation |

In the experimental force mode (`model.predict_forces = true`, see
[methodology.md](methodology.md#forces)) four columns follow:

| Column | Unit | Meaning |
|---|---|---|
| `train_force_mse` | (kcal/(mol·Å))² | mean squared force-component error, training |
| `val_force_mse` | (kcal/(mol·Å))² | the same on the validation split |
| `p_energy`, `p_force` | – | loss prefactors $p_e$, $p_f$ (Eq. 4.10) at the end of the epoch |

In the force mode `train_loss` and `val_loss` are
$p_e \cdot \text{energy MSE} + p_f \cdot \text{force MSE}$ and therefore mix the
two units. The validation metrics are still the energy MSE `val_energy_mse`,
which also selects the best epoch.

### `best.json` and checkpoints

- `checkpoints/epoch_{:04d}.weights.h5`: Keras weights of the atomic network
  after that epoch. With `train.checkpoints = "best_and_last"` (default) only
  the best and the latest epoch are kept; `"all"` keeps every epoch.
  Checkpoints are written atomically (temporary file
  `.<random>.tmp.epoch_….weights.h5` in the same directory, then renamed), so an
  interruption never leaves a partial file under the final name.
- `best.json`: `{"epoch": <int>, "val_energy_mse": <float>}`, the epoch with the
  lowest validation energy MSE so far (model selection, p. 52, deviation
  [A12](methodology.md#deviations-and-decisions)). It is rewritten after the
  checkpoint of that epoch exists, so it never points to a missing or partial
  checkpoint. `--epoch best` (default of `lrmlp evaluate` and `lrmlp predict`)
  reads it; `last` takes the newest checkpoint.

### Evaluation results

`lrmlp evaluate --run R [--split test] [--epoch best]` writes
`evaluation_<split>_<epoch>.json` into the run directory (`<epoch>` is the
resolved epoch number). With `--json OUT` the metrics, i.e. the same object
without `dataset` and `batch_size`, are also written to `OUT` (`-` prints them).

| Field | Type | Meaning |
|---|---|---|
| `energy_mae` | float | $\frac{1}{S}\sum_k \lvert \sum_i E_i^{(k)} - E^{(k)}\rvert$, kcal/mol, the metric of Tab. 5.3 (p. 53) |
| `energy_rmse` | float | root mean squared molecule-energy error, kcal/mol |
| `energy_mse` | float | mean squared molecule-energy error, (kcal/mol)² |
| `force_mae` | float or null | MAE over all force components, kcal/(mol·Å); `null` for energy-only models |
| `force_rmse` | float or null | RMSE over all force components; `null` for energy-only models |
| `n_molecules` | int | molecules evaluated |
| `epoch` | int | evaluated checkpoint |
| `split` | str | evaluated split |
| `dataset` | str | dataset directory that was evaluated (as resolved by `lrmlp evaluate`); only in the run file, not in `--json OUT` |
| `batch_size` | int | molecules per model call; only in the run file, not in `--json OUT` |
| `dataset_matches_run` | bool | only present (as `false`) if the dataset differs from the run's `dataset.json` |
| `dataset_mismatch` | list of str | only present with `dataset_matches_run`: the differing metadata fields |

Before evaluating, the dataset's `metadata.json` is compared with the run's
`dataset.json` (`grid` incl. `assignment` and `sort_neighbours`, `sources`
without their paths, `n_samples`, `split_seed`, `val_fraction`,
`test_fraction`, `splits`, `species`, `max_atomic_number`, `n_max_local`,
`n_max_long_range`). A different dataset, e.g. one with another split seed
whose test split may contain training frames of the run, is an error.
`allow_dataset_mismatch=True` (CLI `--allow-dataset-mismatch`) evaluates it
anyway, logs a warning and records the difference in the two fields above.

## Prediction input and output

`lrmlp predict --run R --input IN.npz --output OUT.npz [--epoch best]
[--batch-size 256] [--workers 1]`, or `lrmlp.potential.Potential` in Python.

Input `.npz`:

| Key | Shape | Unit | Meaning |
|---|---|---|---|
| `R` | (S, N, 3) | Å | positions |
| `z` | (N,) | – | atomic numbers |

Other keys are ignored with a warning, so a raw sGDML file can be passed
directly.

Output `.npz`:

| Key | Shape | Dtype | Unit | Meaning |
|---|---|---|---|---|
| `E` | (S,) | float64 | kcal/mol | $E = \sum_i E_i$ (Eq. 2.7, p. 21), computed in float32 |
| `F` | (S, N, 3) | float64 | kcal/(mol·Å) | only for force models: direct network output in input atom order, **not** $-\nabla E$ (see [methodology.md](methodology.md#forces)) |

The descriptors are computed with the grid configuration of the run and padded
to the run's N_max. A frame in which an atom has more local or long-range rows
than the model's N_max raises `ValueError` (the network has a fixed input size,
p. 44). Coinciding atoms also raise `ValueError`.

`Potential.predict(positions, numbers, batch_size=…, n_workers=…)` accepts
positions of shape (S, N, 3) or (N, 3) and returns a `Prediction` with
`energies` (S,) and `forces` (S, N, 3) or `None`.

## Legacy `.pkl.gzip` datasets

The legacy code (commit `9413bfd`) stored datasets as pickled pandas data frames
`train.pkl.gzip`, `val.pkl.gzip` and `test.pkl.gzip` (`Constants.py:27-29`).
This format is **not supported** anymore: reading it would require pickle and
pandas, and its rows are in grid order, not input order (B01). To convert, run
`lrmlp prepare` again on the original raw `.npz` files. With the profile
`configs/legacy-head.toml` (`grid.assignment = "legacy"`) the new dataset
holds the descriptors of the legacy code (up to floating-point rounding), in
input atom order.
