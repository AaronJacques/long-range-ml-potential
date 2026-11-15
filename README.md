# long-range-ml-potential

Code of the master's thesis *Machine Learning Potentials with Long Range
Interaction* (Aaron Schulz, Technische Universität München, 2024): a neural network
potential that adds a grid-based long-range term to a local DeepPot-SE-style
descriptor. The Python package is `lrmlp` and provides the `lrmlp` command line tool.

## The idea

The energy of a molecule is the sum of atomic energies, $E = \sum_i E_i$
(Eq. 2.7). One network predicts every $E_i$ from two descriptions of the
surroundings of atom $i$ (Ch. 4 of the thesis, p. 36-51):

1. **Local part (Sec. 4.1).** Every atom $j$ within the cutoff radius $r_c$ enters
   through a row $\big(s(r_{ji}),\ s(r_{ji})\,\vec r_{ji}/r_{ji}\big)$ of
   $\tilde R_i^{local}$, with the smooth weight $s(r)$ (Eq. 4.5) and the pair of
   atomic numbers $(z_i, z_j)$ in $Z_i$. This follows DeepPot-SE [14].
2. **Long-range part (Sec. 4.2, 4.3.2).** A fixed axis-aligned grid with cell size
   $\Delta g$ is laid over the molecule. The atoms outside $r_c$ are not used one by one.
   For every remaining cell $k$ the model uses the centre of mass weighted by
   atomic number, $r_m^k$ (Eq. 4.12), and the element counts $f^k$. They form one
   row of $\tilde R_i^{long}$ and of $F_i$ (Eq. 4.16-4.20). The cutoff follows
   from the grid, $r_c = (l_{max} + 1 + 1/\sqrt2)\,\Delta g$ (Eq. 4.14).
3. **Descriptor and network (Sec. 4.3.3).** For both parts, an embedding network
   maps the first column of $\tilde R_i$ together with $Z_i$ (or $F_i$) to $G_i$.
   The descriptor is
   $D_i = G_{i1}^\top \tilde R_i \tilde R_i^\top G_{i2}$ (Eq. 4.8). An energy network
   maps the concatenated local and long-range descriptors to $E_i$.
4. **Training (Sec. 4.3.4).** Energies only, squared error of the total energy,
   one molecule per step (Eq. 4.21), Adam with exponential learning-rate decay
   (Eq. 4.22). The epoch with the best validation MSE is evaluated on the test
   split (p. 52).

```
                      atom i, cutoff r_c (from Δg and l_max, Eq. 4.14)
      ┌──────────────────────────────────┴──────────────────────────────────┐
  neighbours j with r_ij ≤ r_c                  remaining atoms in grid cells k
  R̃_i^local (s, s·x/r, s·y/r, s·z/r)            R̃_i^long (1/r, x/r, y/r, z/r) to r_m^k
  Z_i (z_i, z_j)                                F_i (z_i, element counts f^k)
      │ local embedding → G_i                       │ long-range embedding → G_i
      │ D_i^local = G_i1ᵀ R̃ R̃ᵀ G_i2                 │ D_i^long = G_i1ᵀ R̃ R̃ᵀ G_i2
      └────────────── flatten ⊕ flatten → energy network → E_i ─────────────┘
                                 E = Σ_i E_i
```

Fig. 4.1 (p. 41) of the thesis shows the grid, $r_c$ and the reduced grid of an
atom. Fig. 4.5 (p. 49) shows the full network, and Fig. 4.6 (p. 50) the local-only
network used for comparison. [docs/methodology.md](docs/methodology.md) has every
equation with page references. It also lists the deviations between the original
code and the thesis.

What the package does:

- `lrmlp prepare` reads trajectories (`.npz` with `E`, `F`, `R`, `z`) and computes
  the descriptors once. It stores them as memory-mapped `.npy` arrays with a split
  of 80/10/10 (optionally with several processes).
- `lrmlp train` trains the network. It writes a run directory with the resolved
  config, a training log, checkpoints and the best epoch.
- `lrmlp evaluate` reports the test MAE as in Tab. 5.3. It checks that the dataset
  is the one the run was trained on (`--allow-dataset-mismatch` to override).
- `lrmlp predict` predicts energies for new positions (`--workers` for parallel
  descriptors). `lrmlp info` shows dataset and run metadata.
- `lrmlp tune` runs hyperparameter grid searches. `lrmlp plot-log` and
  `lrmlp plot-weights` draw learning curves and weight distributions
  (Fig. 5.1-5.3).
- One TOML preset per molecule of the thesis (Tab. 5.1/5.2) and a profile
  `legacy-head` that reproduces the original code.

## Installation

Python 3.12 is required (TensorFlow 2.21, Keras 3, NumPy < 2). Only CPU runs are
tested.

```sh
git clone https://github.com/AaronJacques/long-range-ml-potential.git
cd long-range-ml-potential
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[plot]"
lrmlp --help
```

Optional extras (`pyproject.toml`): `plot` (matplotlib, for `plot-log` and
`plot-weights`), `tune` (TensorBoard HParams for `lrmlp tune`), `dev` (pytest, ruff,
mypy) and `legacy` (pandas, tf-keras: only for regenerating the golden fixtures and
for the legacy benchmarks). For development: `pip install -e ".[dev,plot,tune]"`.
`requirements-lock.txt` pins the versions of the tested environment.

TensorBoard is optional. `pip install -e ".[plot,tune]"` additionally enables
TensorBoard logging (scalars in the run's `tb/` directory) and the hyperparameter
search. Without it, `lrmlp train` writes the training log, configuration, best
epoch and checkpoints as usual and logs one INFO line that TensorBoard is not
installed.

## Quickstart

The quickstart uses the small synthetic dataset of the tests
(`tests/fixtures/synthetic_small.npz`: 60 frames of a 21-atom molecule with
Lennard-Jones energies, not real data) and a tiny model
(`tests/fixtures/configs/smoke.toml`). Run it from the repository root. All
outputs go to a new temporary directory, and the whole block takes about ten
seconds:

```sh
OUT="$(mktemp -d "${TMPDIR:-/tmp}/lrmlp-quickstart.XXXXXX")"
lrmlp prepare --config tests/fixtures/configs/smoke.toml --output "$OUT/dataset"
lrmlp train --config tests/fixtures/configs/smoke.toml --dataset "$OUT/dataset" --run-dir "$OUT/run" --set train.epochs=2
lrmlp evaluate --run "$OUT/run"
lrmlp predict --run "$OUT/run" --input tests/fixtures/synthetic_small.npz --output "$OUT/predictions.npz"
lrmlp info --run "$OUT/run"
lrmlp plot-log --run "$OUT/run" --output "$OUT/plots"
echo "outputs in $OUT"
```

`evaluate` prints the test MAE, RMSE and MSE of the best epoch. `predict` writes
`E` (kcal/mol, one value per frame) to `predictions.npz`. `predict` ignores the
`E` and `F` arrays of the input file and logs a warning about them. The run
directory layout is described in [docs/data-format.md](docs/data-format.md).

## Reproducing the thesis experiments

The thesis uses four molecules of the sGDML datasets [16] (cited in the thesis as
<http://www.sgdml.org>; Sec. 4.3.1, p. 40). The repository contains **no real
data**. Place each trajectory as `data/raw/<molecule>.npz` (`paracetamol`,
`aspirin`, `ac-ala3-nhme`, `stachyose`), then for example for Aspirin:

```sh
lrmlp prepare --config configs/aspirin.toml
RUN="$(lrmlp train --config configs/aspirin.toml)"
lrmlp evaluate --run "$RUN" --epoch best
```

For the local-only comparison model (Fig. 4.6), add
`--set model.use_long_range=false --set name=aspirin-local` to `train`. The presets
follow the thesis (Tab. 5.1/5.2, assignment and dropout as described on p. 43 and in
Fig. 4.2/4.4). `configs/legacy-head.toml` reproduces the behaviour of the original code. The
two differ, so numbers from the presets are not directly comparable with Tab. 5.3.
[docs/reproducing.md](docs/reproducing.md) has the full guide: data, all four
molecules, expected runtime, `legacy-head` versus presets, and plotting.

## Results reported in the thesis

These numbers are **reported in the thesis**. They were obtained with the original
code and were **not** reproduced in this repository.

Test MAE of the energy in kcal/mol, with and without the long-range part
(Tab. 5.3, p. 53):

| Molecule | With long range | Only local |
|---|---|---|
| Paracetamol | 2.465 | 3.024 |
| Aspirin | 1.670 | 1.657 |
| Ac-Ala3-NHMe | 4.039 | 4.087 |
| Stachyose | 5.841 | 5.861 |

Comparison with other methods on the same datasets [16], MAE in kcal/mol
(Tab. 6.1, p. 56):

| Molecule | This approach (with long range) | DeepPot-SE [14] | sGDML [17] |
|---|---|---|---|
| Paracetamol | 2.465 | – | 0.113 |
| Aspirin | 1.670 | 0.155 | 0.047 |
| Ac-Ala3-NHMe | 4.039 | – | 0.391 |
| Stachyose | 5.841 | – | 4.002 |

The thesis concludes that the long-range part clearly helps only for Paracetamol.
It finds the result very sensitive to $K_1$ and $K_2$, and it notes that no
systematic hyperparameter tuning was possible (p. 53, Sec. 7.5).

### Limits

- **Invariance.** The energy is translation invariant. It is **not** rotation
  invariant, and **not** invariant under an arbitrary permutation of identical
  atoms. The reason is that the grid is axis-aligned and the embedding sees an
  ordered, flattened list of neighbours. The unchanged original code gives the
  same measurements, so this is a property of the method (A10 in
  [docs/methodology.md](docs/methodology.md#invariances)).
- **Forces** (`model.predict_forces = true`) are experimental. They are a direct
  network output, not $-\nabla E$, and they are not rotation-equivariant (A09).
  The thesis trains on energies only.
- **No real data, no re-measured thesis numbers.** All tests use synthetic
  molecules. With the thesis presets, $N_{max}$ (and hence $K_1$ = `"auto"`) can
  differ from Tab. 5.2.
- **Speed.** [docs/benchmarks.md](docs/benchmarks.md) compares against the
  original code on the same runtime. Preprocessing is 16.83× faster for an
  aspirin-like molecule and 26.11× for a stachyose-like one (up to 38.50× with
  10 processes). Inference from positions is 15.61× faster. Training is 2.28×
  faster with the preset settings (`steps_per_execution = 8`,
  `jit_compile = true`) and 1.20× with the `TrainConfig` defaults
  (`steps_per_execution = 8`, XLA off). Six targets of the refactoring were
  missed; that page names them. Under XLA, dropout draws other random masks than
  without it, so preset runs are not bit-identical to non-XLA runs with the same
  seed. If XLA is not available, training falls back to non-XLA.

## Project layout

```
configs/            presets per molecule (Tab. 5.1/5.2), legacy-head profile, search spaces
src/lrmlp/          the package
  config.py           TOML configuration (dataclasses, --set overrides)
  io.py               raw .npz trajectories
  grid.py             grid, cell levels, local/long-range partition, cell aggregates
  descriptors.py      local and long-range descriptor matrices, padding
  dataset.py          preprocessing, stored dataset, split, tf.data pipelines
  model.py            Keras 3 network (embedding, D_i, energy network)
  schedules.py        learning-rate and loss-weight schedules
  training.py         trainer, training log, checkpoints, best epoch
  evaluation.py       test MAE / RMSE / MSE
  potential.py        inference API (Potential)
  tuning.py           hyperparameter grid search
  analysis/           learning curves and weight distributions
  run.py              run directory contract
  cli.py              command line interface (prepare, train, evaluate, ...)
tests/              unit, golden (against the original code), invariance and end-to-end tests
  fixtures/           synthetic data, smoke config, golden fixtures
benchmarks/         legacy vs. new benchmarks and equivalence checks
notebooks/          training_curves.ipynb (figures of chapter 5 from runs)
docs/               documentation, the thesis PDF, legacy plots
```

`data/` and `runs/` are created by the commands and are ignored by git. The original
code (`Neural-Network/`) was removed and is still available in the history at commit
`9413bfd` (`git archive 9413bfd Neural-Network`).

## Documentation

- [docs/index.md](docs/index.md): overview of all pages
- [docs/methodology.md](docs/methodology.md): the method with references to the
  thesis, deviations of the original code (A01-A17), open questions (F1-F5),
  invariances and limitations
- [docs/configuration.md](docs/configuration.md): every configuration field, presets,
  `--set` overrides
- [docs/reproducing.md](docs/reproducing.md): reproducing the thesis experiments
- [docs/architecture.md](docs/architecture.md): modules, data flow, public API,
  design decisions
- [docs/data-format.md](docs/data-format.md): raw data, dataset, run directory,
  prediction files
- [docs/benchmarks.md](docs/benchmarks.md): speed and equivalence against the
  original code
- [The thesis (PDF)](docs/thesis/Masterarbeit__Machine_Learning_Potentials_with_Long_Range_Interaction.pdf)

## Citation

If you use this code, please cite the thesis:

```bibtex
@mastersthesis{schulz2024longrange,
  author = {Schulz, Aaron},
  title  = {Machine Learning Potentials with Long Range Interaction},
  school = {Technische Universität München},
  type   = {Master's Thesis in Quantum Science \& Technology},
  year   = {2024},
  month  = oct,
  note   = {Physics Department. Supervisor: Christian Mendl.
            Code: https://github.com/AaronJacques/long-range-ml-potential}
}
```

[CITATION.cff](CITATION.cff) has the same information in machine-readable form
(GitHub shows it as "Cite this repository").

## License

License: not yet specified – all rights reserved by the author. The repository
contains no license file. Until the author chooses a license, you may not copy,
modify or redistribute the code or the thesis PDF without the author's permission.
