# Reproducing the thesis experiments

This guide runs the experiments of Ch. 5 of the thesis (Tab. 5.3, p. 53, and
Fig. 5.1-5.3) with `lrmlp`. It covers four molecules. Each one is trained with and
without the long-range part, and the best validation epoch is evaluated on the test
split.

> **Status.** The repository contains no real data, and the numbers of the thesis
> have **not** been reproduced with this code. All tests use synthetic molecules.
> The commands below are checked for syntax and valid configurations
> (`tests/test_docs.py`), but they were not run on the sGDML data.

Install the package first ([README](../README.md#installation)) and run all
commands from the repository root.

## 1. Get the data

The thesis uses the datasets of [16], *sGDML Symmetric Gradient Domain Machine
Learning*, cited in the thesis bibliography as <http://www.sgdml.org> (p. 65). They
were computed with DFT (FHI-aims) and give positions in Å and energies in kcal/mol
(Sec. 4.3.1, p. 40). Download the dataset of every molecule there. Save it under
the path that the preset expects:

| Molecule | Atoms | Frames available / used (Tab. 4.1, p. 40) | Preset | Raw file |
|---|---|---|---|---|
| Paracetamol | 20 | 106 490 / 65 000 | `configs/paracetamol.toml` | `data/raw/paracetamol.npz` |
| Aspirin | 21 | 211 762 / 65 000 | `configs/aspirin.toml` | `data/raw/aspirin.npz` |
| Ac-Ala3-NHMe | 42 | 85 109 / 65 000 | `configs/ac-ala3-nhme.toml` | `data/raw/ac-ala3-nhme.npz` |
| Stachyose | 87 | 27 272 / 27 272 | `configs/stachyose.toml` | `data/raw/stachyose.npz` |

The file must be an `.npz` archive with the arrays `E`, `F`, `R` and `z`
([data-format.md](data-format.md#raw-trajectories-npz)). Other keys are ignored.
`data/` is ignored by git. To use another location, change `data.sources` with
`--set` (relative to the config file) or pass an absolute path.

After `lrmlp prepare` (step 2), `lrmlp info --dataset data/processed/<molecule>`
shows the number of frames in the file and the number used. Compare them with
Tab. 4.1. Only the first 65 000 frames are used; Stachyose uses all frames
(`data.n_samples_per_source = "all"`).

## 2. Prepare, train, evaluate

For one molecule, with and without the long-range part:

```sh
lrmlp prepare --config configs/aspirin.toml
RUN="$(lrmlp train --config configs/aspirin.toml)"
lrmlp evaluate --run "$RUN" --epoch best --json results/aspirin-long-range.json
RUN_LOCAL="$(lrmlp train --config configs/aspirin.toml --set model.use_long_range=false --set name=aspirin-local)"
lrmlp evaluate --run "$RUN_LOCAL" --epoch best --json results/aspirin-local.json
```

- `prepare` computes the descriptors once and writes `data/processed/aspirin`
  (`data.output_dir` of the preset). Use `--workers N` to choose the number of
  worker processes (default: all CPUs).
- `train` prints the run directory on standard output (the log goes to standard
  error). The run directory is `runs/<name>-YYYYmmdd-HHMMSS`. Both models use the
  same dataset; the local-only model gets its own name, so its runs are easy to
  tell apart.
- `evaluate --epoch best` evaluates the epoch with the lowest validation energy
  MSE on the test split (p. 52). The MAE in kcal/mol is the metric of Tab. 5.3. It
  also writes `evaluation_test_<epoch>.json` into the run directory.

All four molecules in one go:

```sh
for m in paracetamol aspirin ac-ala3-nhme stachyose; do
  lrmlp prepare --config "configs/$m.toml"
  RUN="$(lrmlp train --config "configs/$m.toml")"
  lrmlp evaluate --run "$RUN" --epoch best --json "results/$m-long-range.json"
  RUN_LOCAL="$(lrmlp train --config "configs/$m.toml" --set model.use_long_range=false --set "name=$m-local")"
  lrmlp evaluate --run "$RUN_LOCAL" --epoch best --json "results/$m-local.json"
done
```

`lrmlp info --run "$RUN"` shows the configuration, the best epoch and the
evaluations of a run.

The presets take grid and training parameters from Tab. 5.1 and 5.2 (p. 52):

| Preset | $\Delta g$ (Å) | $l_{max}$ | $r_c$ (Å) | $K_1$ local / long | $K_2$ local / long | $\eta_{init}$ | $s$ | Epochs |
|---|---|---|---|---|---|---|---|---|
| `paracetamol` | 1 | 6 | 7.71 | auto / auto | 4 / 1 | 5e-5 | 2.08e5 | 200 |
| `aspirin` | 1 | 6 | 7.71 | auto / auto | 4 / 1 | 1e-4 | 2.08e5 | 200 |
| `ac-ala3-nhme` | 2 | 2 | 7.41 | auto / auto | 4 / 2 | 5e-5 | 2.08e5 | 150 |
| `stachyose` | 2 | 4 | 11.41 | 30 / 6 | 4 / 2 | 5e-5 | 8.727e4 | 130 |

"auto" sets $K_1 = N_{max}$ of the dataset ("$N_{max} = K_1$ always chosen",
p. 52). Every field is explained in [configuration.md](configuration.md).

## 3. Expected runtime

There are no measurements on the real datasets. The figures below are
extrapolations from [benchmarks.md](benchmarks.md). Those benchmarks used
synthetic aspirin-like (21 atoms) and stachyose-like (87 atoms) molecules, the
network of `legacy-head` and the `"legacy"` assignment, on an Apple M4 CPU
(10 cores). Treat them as orders of magnitude only.

| Step | Measured rate | Extrapolation |
|---|---|---|
| `prepare`, aspirin-like, 1 process | 3,900 mol/s | 65 000 frames ≈ 17 s |
| `prepare`, stachyose-like, 1 process / 10 processes | 292 / 431 mol/s | 27 272 frames ≈ 93 s / 63 s |
| Dataset on disk | 12.2 kB (aspirin-like) / 230.2 kB (stachyose-like) per molecule | ≈ 0.8 GB (Aspirin) / ≈ 6.3 GB (Stachyose) |
| `train`, aspirin-like, `steps_per_execution = 8`, `jit_compile = true` (presets) | 720 steps/s | 52 000 steps per epoch ≈ 72 s; 200 epochs ≈ 4 h |
| `train`, aspirin-like, `steps_per_execution = 8`, `jit_compile = false` | 378 steps/s | ≈ 138 s per epoch; 200 epochs ≈ 7.6 h |
| Validation (no gradients) | 5,196 mol/s | 6 500 molecules ≈ 1.3 s per epoch |

One training step processes one molecule ($\lvert B\rvert = 1$, p. 51), so the
training time grows with the number of training frames and with the molecule size.
The thesis network (embedding `[64, 128, 128, 256]`, three residual blocks) is
larger than the benchmarked `legacy-head` network. Ac-Ala3-NHMe (42 atoms) and
Stachyose (87 atoms) were not benchmarked in training and will need more time per
step. Plan with several hours to about a day per model on a CPU, and two models per
molecule.

## 4. Presets versus `legacy-head`: what is comparable with Tab. 5.3

The numbers of Tab. 5.3 come from the **original code**. The presets follow the
**thesis text** where the original code deviated from it
([methodology.md](methodology.md#deviations-and-decisions)):

| Aspect | Molecule presets | Original code / `legacy-head` | Deviation |
|---|---|---|---|
| Local vs. long-range assignment | `grid.assignment = "thesis"`: every atom within $r_c$ is local, all others in the reduced grid (p. 43) | `"legacy"`: cell-based. Some atoms within $r_c$ enter only via a centre of mass, some atoms beyond $r_c$ are dropped | A01, A02 |
| Dropout | `model.dropout_rate = 0.2`, active in training (Fig. 4.2, 4.4) | declared 0.2 but never active, equivalent to `0.0` | A03 |
| Embedding widths | `[64, 128, 128, 256]` (Fig. 4.3) | `[64, 128, 128, 128]` at `9413bfd` | A04, F3 |
| Residual blocks of the energy network | `[128, 128, 128]` (Fig. 4.4) | `[128, 128]` at `9413bfd` | A05, F3 |
| $N_{max}$ | from the dataset | Tab. 5.2 values, obtained with the legacy assignment | – |
| Execution | `steps_per_execution = 8`, `jit_compile = true` | one step per call, no XLA | technical only |

Consequences:

- With the presets, the inputs differ from the ones of the thesis runs. In
  `"thesis"` mode no atom is dropped and every atom within $r_c$ is local, so
  $N_{max}$, and with `"auto"` also $K_1$, can differ from Tab. 5.2. `lrmlp info
  --dataset ...` shows the $N_{max}$ of a prepared dataset. With dropout active,
  the models are also trained differently. The resulting MAE values are therefore
  **not directly comparable** with Tab. 5.3. They show how the method performs when
  it is implemented as the thesis describes it.
- It is not known which widths the code had at submission (F3). The original code
  changed them after the submission date, so even the legacy settings may not match
  the thesis runs exactly.
- `configs/legacy-head.toml` is the exact state of `Constants.py` at `9413bfd`, a
  mixed state: Aspirin grid and data with the $K$ values and decay steps of
  Stachyose. It is the reference of the golden tests and benchmarks, but it is
  **not** the configuration of any single row of Tab. 5.3.

To come as close as possible to the original code for one molecule, start from the
molecule preset and switch the deviations back to the legacy behaviour. The
assignment changes the descriptors, so this needs its own dataset directory, and
`prepare` and `train` need the same grid overrides (`train` refuses a dataset
built with another grid):

```sh
lrmlp prepare --config configs/aspirin.toml --set grid.assignment=legacy --set data.output_dir=../data/processed/aspirin-legacy
RUN="$(lrmlp train --config configs/aspirin.toml --set grid.assignment=legacy --set data.output_dir=../data/processed/aspirin-legacy --set model.dropout_rate=0.0 --set "model.embedding_dims=[64, 128, 128, 128]" --set "model.fitting_res_blocks=[128, 128]" --set name=aspirin-legacy)"
lrmlp evaluate --run "$RUN" --epoch best
```

`data.output_dir` is relative to the config file, so `../data/processed/...` is
the `data/` directory at the repository root. The original code also trained
without XLA. With dropout 0 the XLA and non-XLA runs agree to rounding
([benchmarks.md](benchmarks.md#equivalence)). Add
`--set train.jit_compile=false` anyway if you want to rule this out. The run is
still not bit-identical to an original run: Keras 3 and the original Keras 2 code
differ in rounding, and the original code used no seeds.

## 5. Learning curves and weight distributions

Fig. 5.1 and 5.2 (p. 54) show the training and validation energy MSE of
Paracetamol, without the first 19 epochs (p. 53). Fig. 5.3 (p. 55) compares the
weight distributions of the models with and without long range. Both need the
extra `plot` (`pip install -e ".[plot]"`):

```sh
lrmlp plot-log --run "$RUN" --output figures/paracetamol --skip 19
lrmlp plot-log --run "$RUN_LOCAL" --output figures/paracetamol --skip 19
lrmlp plot-weights --run-a "$RUN" --run-b "$RUN_LOCAL" --output figures/paracetamol-weights.pdf
```

Here `RUN` and `RUN_LOCAL` are the two Paracetamol runs of step 2. The notebook
[`notebooks/training_curves.ipynb`](../notebooks/training_curves.ipynb) draws the
figures for all molecules: enter the run directories in `RUN_DIRS` at the top and
run all cells. The plots in `docs/assets/legacy-runs/` are from early runs of the
original code and are **not** the figures of the thesis
([docs/assets/legacy-runs/README.md](assets/legacy-runs/README.md)).

## 6. Hyperparameter search (optional)

The values of Tab. 5.2 come from hyperparameter tuning, but the thesis states that
systematic tuning was not possible (p. 52-53, Sec. 7.5). `lrmlp tune` trains one
model per point of a grid:

```sh
lrmlp tune --config configs/aspirin.toml --space configs/hparam_search.energy.toml --dry-run
lrmlp tune --config configs/aspirin.toml --space configs/hparam_search.energy.toml --max-trials 10
```

`configs/hparam_search.energy.toml` (energy only, 48 combinations) is the
recommended start. `configs/hparam_search.example.toml` reproduces the search
space of the original `HyperparameterTester.py`, which trains the experimental
force mode (A09).
