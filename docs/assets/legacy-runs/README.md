# Legacy training plots (`Images/` until T16)

These files were tracked in `Images/` and were moved here unchanged with `git mv`
(`git log --follow` shows their history). They are kept as a record of early
experiments. **They are not the figures of the thesis** and nothing in the repository
references them; the thesis figures (Fig. 5.1, 5.2, 5.3, A.1-A.6) are drawn from
trained runs with `lrmlp.analysis` and [`notebooks/training_curves.ipynb`](../../../notebooks/training_curves.ipynb).

## Origin

| Files | Added in | Date |
|---|---|---|
| `Model_Graph.jpg` | `5d668a3` | 2024-01-19 |
| `plots_Model-aspirin_grid_size_1.5_n_samples_40000-…-decay-steps13-34-47/` | `b4c6deb` "Plots" | 2024-03-18 |
| `plots_Model-aspirin_grid_size_1.5_n_samples_40000-…-decay-steps14-53-56/`, `…-decay-steps17-57-45/` | `67a50ec` "Plots" | 2024-03-19 |
| `plots_Model-aspirin_max_local_level_2_grid_size_1_n_samples_40000-…/` | `60c100e` "Plots" | 2024-03-19 |
| the 6 folders `plots_Model-aspirin_max_local_level_{4,6,8}_grid_size_1_n_samples_50000-…/` | `e573b85` | 2024-03-26 |

All plots come from Aspirin runs of the legacy code (`Neural-Network/`) in March 2024,
an earlier state than the one used for the thesis (commit `9413bfd`, the reference of
the golden fixtures and benchmarks). That state trained energies **and** forces with
the loss weights p_E, p_F of Eq. 4.10, on 40 000 or 50 000 sampled configurations,
with mini-batches; the thesis runs train the energy only (Eq. 4.21), one molecule per
step, with other learning rates and decay steps (Tab. 5.1/5.2, `configs/*.toml`).

`Model_Graph.jpg` is a TensorBoard screenshot of the graph of an even earlier model
version (January 2024); it does not clearly match `Neural-Network/Model.py@9413bfd`
(e.g. its `tf.expand_dims` is commented out there) and does not describe `lrmlp.model`.

## Folder names

The legacy `Training.py` named a checkpoint folder
`Model-{dataset folder}-DS-{initial learning rate}-lr-{decay steps}-decay-steps{HH-MM-SS}`,
each value **before** its label (B12), and the plots were saved as `plots_<folder>`.
For example

```
plots_Model-aspirin_max_local_level_6_grid_size_1_n_samples_50000-DS-0.00025-lr-12000.0-decay-steps13-45-22
```

| Part | Meaning |
|---|---|
| `aspirin_max_local_level_6_grid_size_1_n_samples_50000` | dataset folder: molecule, l_max = 6 (missing in the oldest folders), grid cell size 1 Å (1.5 Å in the oldest), 50 000 sampled configurations |
| `DS` | separator after the dataset name |
| `0.00025-lr` | initial learning rate η = 2.5·10⁻⁴ (5·10⁻⁴ in most folders) |
| `12000.0-decay-steps` | decay steps of the loss weights p_E, p_F (`Hyperparameters.decay_steps`), not of the learning rate |
| `13-45-22` | start time of the run (no date, B12) |

Each folder holds `energy_loss.png`, `force_loss.png`, `total_loss.png` (training and
validation losses over the epochs) and `weights.png` (the loss weights p_E, p_F over
the epochs).
