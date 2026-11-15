# Configuration reference

An experiment is described by one TOML file. The file holds a top-level `name`
and the four tables `[grid]`, `[data]`, `[model]` and `[train]`. `lrmlp.config`
loads it into frozen dataclasses (`GridConfig`, `DataConfig`, `ModelConfig`,
`TrainConfig`, bundled in `ExperimentConfig`). Unknown keys, wrong types and
invalid values are rejected with an error message that names the key. Every field
except `name` and `data.sources` has a default.

This page lists **every** field. `tests/test_docs.py` checks that the list is
complete and that the defaults shown here match the code. The columns are:

- **Type**: the TOML type.
- **Default**: the value used when the key is missing.
- **Thesis**: the symbol or place in the thesis (printed page numbers; see
  [methodology.md](methodology.md)).
- **Legacy**: the name of the constant in the original code (`Neural-Network/` at
  commit `9413bfd`, mostly `Constants.py`).

"Not specified in the thesis" means that the thesis does not give a value. The
deviations A01-A17, open questions F1-F7 and legacy defects B01-B19 are explained in
[methodology.md](methodology.md#deviations-and-decisions).

```toml
name = "aspirin"

[grid]
grid_size = 1.0
max_local_level = 6

[data]
sources = ["../data/raw/aspirin.npz"]

[model]
k2_long = 1

[train]
epochs = 200
learning_rate = 1e-4
```

## Top level

| Key | Type | Default | Meaning | Thesis | Legacy |
|---|---|---|---|---|---|
| `name` | string | required | Experiment name. Prefix of the run directories (`<name>-YYYYmmdd-HHMMSS`) and name of the default dataset directory. It must be usable as a directory name | – | dataset / checkpoint folder names |

## `[grid]`: grid and cutoff

Thesis Sec. 4.2 and 4.3.2 (p. 39-44), Tab. 5.1 (p. 52).

| Key | Type | Default | Meaning | Thesis | Legacy |
|---|---|---|---|---|---|
| `grid.grid_size` | float | `1.0` | Edge length of the grid cells in Å | $\Delta g$ (Eq. 4.11, Tab. 5.1) | `Dataset.GRID_SIZE` |
| `grid.max_local_level` | int | `6` | Highest cell level (Chebyshev distance of cells) of the local region $G_i$. Together with `grid_size` it fixes the cutoff | $l_{max}$ (Eq. 4.13, Tab. 5.1) | `Dataset.MAX_LOCAL_LEVEL` |
| `grid.inner_cutoff_offset` | float | `0.2` | Distance in Å between cutoff and inner cutoff of the switching function $s(r)$ | $r_c - r_{cs} = 0.2$ Å (Eq. 4.5, Tab. 5.1) | `Dataset.CUT_OFF - Dataset.INNER_CUT_OFF` |
| `grid.max_atomic_number` | int | `100` | Length of the element-count vector $f^k$ and largest accepted atomic number. `F_i` has `max_atomic_number + 1` columns | $m_{max}$ (p. 42; value not specified in the thesis) | `Dataset.MAX_ATOM_ELEMENTS` |
| `grid.assignment` | `"thesis"` or `"legacy"` | `"thesis"` | Rule that splits the other atoms into local and long range. `"thesis"`: every atom with $r_{ij} \le r_c$ is local, all others go to the reduced grid. `"legacy"`: the cell-based rule of the original code (A01, A02) | p. 43, Fig. 4.1 | `Dataset.py:310-356` (= `"legacy"`) |
| `grid.sort_neighbours` | bool | `false` | Sort the local rows and the long-range rows of every atom by descending first column ($s$ resp. $1/r$) | – (not in the thesis; A17) | `Dataset.sort_matrix` |

Derived values (not configurable, stored as `derived` in `config.json` and in the
dataset metadata):

- `grid.cutoff` $= r_c = (l_{max} + 1 + \sqrt{0.5})\,\Delta g$ (Eq. 4.14). Legacy:
  `Dataset.CUT_OFF`.
- `grid.inner_cutoff` $= r_{cs} = r_c - $ `inner_cutoff_offset`. Legacy:
  `Dataset.INNER_CUT_OFF`.

## `[data]`: sources, samples and split

Thesis Sec. 4.3.1 (p. 40), Tab. 4.1.

| Key | Type | Default | Meaning | Thesis | Legacy |
|---|---|---|---|---|---|
| `data.sources` | list of paths | required | Raw `.npz` trajectories with `E`, `F`, `R`, `z` ([data-format.md](data-format.md#raw-trajectories-npz)). Several sources are concatenated in the given order | datasets of [16] | `create_dataset(paths, ...)` |
| `data.n_samples_per_source` | int or `"all"` | `65000` | Upper bound on the number of leading frames taken from every source. A source with fewer frames contributes all of them; `"all"` takes every frame | first 65 000 frames (p. 40, Tab. 4.1) | `create_dataset(n_samples_per)` |
| `data.val_fraction` | float | `0.1` | Fraction of the samples in the validation split | 80/10/10 split (p. 40) | `create_dataset(val_split)` |
| `data.test_fraction` | float | `0.1` | Fraction of the samples in the test split | 80/10/10 split (p. 40) | `create_dataset(test_split)` |
| `data.split_seed` | int | `42` | Seed of the random split (reproduces the pandas split of the original code exactly) | – (not specified in the thesis) | seed in `Dataset.py:475-477` |
| `data.output_dir` | path | `data/processed/<name>` | Directory of the preprocessed dataset (input of `train`). The default is relative to the config file | – | `Dataset.FOLDER` / `save_folder` |
| `data.n_workers` | int | `0` | Worker processes of `lrmlp prepare`: `0` = all CPUs, `1` = no worker processes. `prepare --workers` overrides it | – | – |

## `[model]`: network

Thesis Sec. 4.3.3 (p. 45-50), Fig. 4.2-4.6, Tab. 5.2 (p. 52).

| Key | Type | Default | Meaning | Thesis | Legacy |
|---|---|---|---|---|---|
| `model.activation` | `"elu"` or `"relu"` | `"elu"` | Activation after every hidden Dense layer | ELU (Fig. 4.2, 4.4) | `Model.activation` |
| `model.embedding_dims` | list of int | `[64, 128, 128, 256]` | Embedding network: the first entry is the width of the first Dense projection, the others are the widths of the Dense-Res-Blocks. The final Dense layer to $N_{max} \cdot K_1$ is added automatically | Fig. 4.3 (A04) | `Hyperparameters.embedding_dims` (`[64, 128, 128, 128]` at `9413bfd`) |
| `model.fitting_width` | int | `128` | Width of the first Dense layer of the energy network | Fig. 4.4 | hard-coded 128 in `Model.py:54-59` |
| `model.fitting_res_blocks` | list of int | `[128, 128, 128]` | Widths of the Dense-Res-Blocks of the energy network (may be empty) | Fig. 4.4: 3 × 128 (A05) | `Model.py:54-59` (`[128, 128]`) |
| `model.dropout_rate` | float | `0.2` | Dropout rate in the Res-Blocks and after the first energy layer, active during training. `0.0` reproduces the original code, where dropout was never active (A03) | dropout 0.2 (Fig. 4.2, 4.4) | 0.2 declared, never active |
| `model.k1_local` | int or `"auto"` | `"auto"` | Number of columns of $G_i$ of the local branch. `"auto"` sets $K_1 = N_{max}^{local}$ of the dataset | $K_1^{local}$ ($M_1$; p. 38, 45, Tab. 5.2) | `Hyperparameters.M1_local` |
| `model.k1_long` | int or `"auto"` | `"auto"` | As `k1_local` for the long-range branch ($N_{max}^{long}$) | $K_1^{long}$ (Tab. 5.2) | `Hyperparameters.M1_long` |
| `model.k2_local` | int | `4` | Number of columns of $G_{i2}$ of the local branch; $D_i$ is $K_1 \times K_2$ | $K_2^{local}$ ($M_2$; Eq. 4.8, 4.9, Tab. 5.2) | `Hyperparameters.M2_local` |
| `model.k2_long` | int | `1` | As `k2_local` for the long-range branch | $K_2^{long}$ (Tab. 5.2) | `Hyperparameters.M2_long` |
| `model.use_long_range` | bool | `true` | Full network with the long-range branch (Fig. 4.5) or local-only network (Fig. 4.6) | Fig. 4.5 / 4.6, Tab. 5.3 | `Model.use_long_range` |
| `model.predict_forces` | bool | `false` | **Experimental** force mode: the energy network gets four outputs per atom ($E_i$ and a direct force). The forces are not $-\nabla E$ and not equivariant (A09, F4) | not part of the method (forces not trained, p. 40) | `not Model.predict_only_energy` |

$N_{max}^{local}$ and $N_{max}^{long}$ (padding, p. 44) are **not** configured. They
are read from the dataset (`metadata.json`). The original code hard-coded
`Model.n_max_local = 86` and `Model.n_max_long_range = 28` (bug B08).

## `[train]`: optimisation and bookkeeping

Thesis Sec. 4.3.4 (p. 51), Tab. 5.2 (p. 52).

| Key | Type | Default | Meaning | Thesis | Legacy |
|---|---|---|---|---|---|
| `train.epochs` | int | `200` | Number of epochs | epochs (Tab. 5.2) | `Hyperparameters.epochs` |
| `train.learning_rate` | float | `1e-4` | Initial learning rate of Adam | $\eta_{init}$ (Eq. 4.22, Tab. 5.2) | `Hyperparameters.initial_learning_rate` |
| `train.lr_decay_steps` | float | `2.08e5` | Optimizer steps per decay of the learning rate (4 × the training split, p. 52) | $s$ (Eq. 4.22, Tab. 5.2) | `Hyperparameters.lr_decay_steps` |
| `train.lr_decay_rate` | float | `0.97` | Base of the exponential learning-rate decay | 0.97 (Eq. 4.22) | `Hyperparameters.decay_rate` (shared with the loss weights) |
| `train.lr_staircase` | bool | `true` | Decay in discrete steps, $0.97^{\lfloor t/s \rfloor}$, as the original code (A07). `false` gives the continuous Eq. 4.22 | Eq. 4.22 (continuous) | `staircase=True` (`Training.py:178`) |
| `train.p_energy_start` | float | `1.0` | Start value of the energy loss weight; acts in the force mode only | $p_e$ (Eq. 4.10) | `Hyperparameters.p_energy_start` |
| `train.p_energy_limit` | float | `1.0` | Limit of the energy loss weight; force mode only | $p_e$ (Eq. 4.10) | `Hyperparameters.p_energy_limit` |
| `train.p_force_start` | float | `1.0` | Start value of the force loss weight; force mode only | $p_f$ (Eq. 4.10) | `Hyperparameters.p_force_start` |
| `train.p_force_limit` | float | `0.0` | Limit of the force loss weight; force mode only | $p_f$ (Eq. 4.10) | `Hyperparameters.p_force_limit` |
| `train.loss_weight_decay_steps` | float | `7.5e3` | Decay steps $\tau$ of the loss-weight schedule $p(t) = p_{limit}(1 - r^{t/\tau}) + p_{start}\,r^{t/\tau}$; force mode only | – (not specified in the thesis) | `Hyperparameters.decay_steps` |
| `train.loss_weight_decay_rate` | float | `0.97` | Decay rate $r$ of the loss-weight schedule; force mode only | – (not specified in the thesis) | `Hyperparameters.decay_rate` (shared with the learning rate) |
| `train.shuffle_buffer` | int | `1000` | Shuffle buffer of the training pipeline; `1` disables shuffling | – (not specified in the thesis) | `Dataset.SHUFFLE_BUFFER_SIZE` |
| `train.seed` | int | `42` | Global random seed (weights, dropout, shuffling) | – (not specified in the thesis) | – (no seeds, B11) |
| `train.deterministic` | bool | `false` | Request deterministic TensorFlow ops | – | – |
| `train.steps_per_execution` | int | `8` | Training steps per compiled call. Purely technical: the result is bit-identical to `1` (also with dropout and XLA) and faster ([benchmarks.md](benchmarks.md)) | – | – |
| `train.jit_compile` | bool | `false` | Compile the training step with XLA. Purely technical and about 2× faster. With dropout, XLA draws other dropout masks than a run without XLA: statistically equivalent, not bit-identical. If XLA is not available, the trainer logs a warning, trains without XLA and records `false` in `config.json`. The four molecule presets set `true` | – | – |
| `train.log_every_steps` | int | `1000` | Interval of the step-level progress log (written at the end of the first compiled call that reaches the next multiple) | – | print interval in `Training.py:238` |
| `train.checkpoints` | `"best_and_last"` or `"all"` | `"best_and_last"` | Keep only the best and the last epoch, or one checkpoint per epoch. The best epoch has the lowest validation energy MSE (`best.json`, A12) | model selection by validation MSE (p. 52) | every epoch saved as `.h5` |
| `train.run_root` | path | `"runs"` | Directory below which `lrmlp train` creates the run directories. The default is relative to the config file | – | `../Checkpoints` |

The batch size is fixed to one molecule per optimizer step, $\lvert B\rvert = 1$
(p. 51, A15), and is not configurable.

## Overrides with `--set`

`lrmlp prepare` and `lrmlp train` accept `--set KEY=VALUE` (repeatable). The value
replaces the one from the file before validation. The same works in Python with
`load_config(path, overrides)` or `ExperimentConfig.with_overrides`.

- `KEY` is `name` or `<section>.<field>` with one of the fields on this page, for
  example `train.epochs`. Unknown keys are rejected.
- `VALUE` is parsed as a TOML literal: `2`, `1e-3`, `true`, `[16, 32]`, `"auto"`.
  If it does not parse, it is taken as a plain string, so
  `--set grid.assignment=legacy` and `--set 'grid.assignment="legacy"'` are the
  same.
- Types are checked as for the file: `--set train.epochs=2.5` is an error.

```sh
lrmlp train --config configs/aspirin.toml --set train.epochs=10 --set train.jit_compile=false
lrmlp train --config configs/aspirin.toml --set model.use_long_range=false --set name=aspirin-local
lrmlp prepare --config configs/legacy-head.toml --set data.n_samples_per_source=1000
```

`lrmlp tune` uses the same dotted keys in its search space files
(`configs/hparam_search.energy.toml`, `configs/hparam_search.example.toml`).

## Path resolution

- Relative paths **inside** a config file (`data.sources`, `data.output_dir`,
  `train.run_root`) are resolved relative to the directory of that file, not to
  the current working directory.
- The same holds for paths given with `--set`: with `configs/aspirin.toml`,
  `--set train.run_root=runs` gives `configs/runs`. Pass an absolute path to use
  another location.
- Without `data.output_dir` the dataset goes to `data/processed/<name>` below the
  directory of the config file. Without `train.run_root` the runs go to `runs/`
  below the directory of the config file.
- Path **arguments** of the CLI (`--config`, `--dataset`, `--run`, `--run-dir`,
  `--output`, `--input`, ...) are relative to the current working directory.

The presets in `configs/` use `../data/raw/<molecule>.npz`,
`../data/processed/<molecule>` and `../runs`, that is the directories `data/` and
`runs/` at the repository root.

The run directory stores the resolved configuration as `config.json` (absolute
paths, derived cutoffs, the $K_1$ values used) and the dataset metadata with
$N_{max}$ as `dataset.json` (see [data-format.md](data-format.md#run-directory)).

## Presets and `legacy-head`

`configs/` contains one preset per molecule of the thesis (Tab. 5.1/5.2, p. 52)
and the profile `legacy-head`. That profile reproduces `Constants.py` at commit
`9413bfd`, the last state of the original code. The table shows every field that
differs between them or from the default. All other fields have their default
values (for example `train.lr_decay_rate = 0.97`, `train.lr_staircase = true`,
`model.use_long_range = true`, `model.predict_forces = false`). `tests/test_docs.py`
checks the table against the files.

| Key | Default | `paracetamol` | `aspirin` | `ac-ala3-nhme` | `stachyose` | `legacy-head` |
|---|---|---|---|---|---|---|
| `grid.grid_size` | `1.0` | `1.0` | `1.0` | `2.0` | `2.0` | `1.0` |
| `grid.max_local_level` | `6` | `6` | `6` | `2` | `4` | `6` |
| `grid.assignment` | `"thesis"` | `"thesis"` | `"thesis"` | `"thesis"` | `"thesis"` | `"legacy"` |
| `data.n_samples_per_source` | `65000` | `65000` | `65000` | `65000` | `"all"` | `65000` |
| `model.embedding_dims` | `[64, 128, 128, 256]` | `[64, 128, 128, 256]` | `[64, 128, 128, 256]` | `[64, 128, 128, 256]` | `[64, 128, 128, 256]` | `[64, 128, 128, 128]` |
| `model.fitting_res_blocks` | `[128, 128, 128]` | `[128, 128, 128]` | `[128, 128, 128]` | `[128, 128, 128]` | `[128, 128, 128]` | `[128, 128]` |
| `model.dropout_rate` | `0.2` | `0.2` | `0.2` | `0.2` | `0.2` | `0.0` |
| `model.k1_local` | `"auto"` | `"auto"` | `"auto"` | `"auto"` | `30` | `30` |
| `model.k1_long` | `"auto"` | `"auto"` | `"auto"` | `"auto"` | `6` | `6` |
| `model.k2_local` | `4` | `4` | `4` | `4` | `4` | `4` |
| `model.k2_long` | `1` | `1` | `1` | `2` | `2` | `2` |
| `train.epochs` | `200` | `200` | `200` | `150` | `130` | `200` |
| `train.learning_rate` | `1e-4` | `5e-5` | `1e-4` | `5e-5` | `5e-5` | `5e-5` |
| `train.lr_decay_steps` | `2.08e5` | `2.08e5` | `2.08e5` | `2.08e5` | `8.727e4` | `8.727e4` |
| `train.steps_per_execution` | `8` | `8` | `8` | `8` | `8` | `8` |
| `train.jit_compile` | `false` | `true` | `true` | `true` | `true` | `false` |

Notes:

- **$r_c$** follows from the grid: 7.71 Å (Paracetamol, Aspirin, `legacy-head`),
  7.41 Å (Ac-Ala3-NHMe), 11.41 Å (Stachyose).
- **$K_1$ = `"auto"`** takes $N_{max}$ of the dataset ("$N_{max} = K_1$ always
  chosen", p. 52). For Stachyose, Tab. 5.2 gives $K_1 = 30/6$ for
  $N_{max} = 86/28$, and the preset follows the table.
- **`legacy-head`** is a mixed state of the original code: the grid and data of
  Aspirin (Δg 1, $l_{max}$ 6, `data/raw/aspirin.npz`) with the $K$ values and decay
  steps of Stachyose. It also uses the original assignment rule (A01, A02), the
  original widths (A04, A05, F3) and inactive dropout (A03). It is the profile of
  the golden tests and of the benchmarks. The mapping of every legacy constant to
  a field is in the header of `configs/legacy-head.toml`.
- **Execution settings** (`steps_per_execution`, `jit_compile`) do not change the
  method. Use `--set train.jit_compile=false` to compare a preset run bit for bit
  with a non-XLA run.
- The test configurations in `tests/fixtures/configs/` (`smoke.toml`,
  `golden_legacy.toml`) are small models for the tests and the
  [quickstart](../README.md#quickstart). They are not thesis experiments.

Search spaces for `lrmlp tune`:

- `configs/hparam_search.energy.toml` (48 combinations of $\eta_{init}$, $s$ and
  the decay rate, energy only) is the recommended starting point.
- `configs/hparam_search.example.toml` reproduces the axes of the original
  `HyperparameterTester.py` (15 552 combinations). It trains the experimental
  force mode; use `--max-trials`.
