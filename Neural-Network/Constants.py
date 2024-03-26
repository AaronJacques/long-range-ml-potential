from dataclasses import dataclass
import os


# Hyperparameters
@dataclass(frozen=True)
class Keys:
    ATOMS_KEY = "ase_atoms"
    ENERGY_KEY = "energy"
    FORCE_KEY = "forces"
    GRID_KEY = "grid"
    LOCAL_DISTANCE_MATRIX_KEY = "local_distance_matrix"
    LOCAL_ATOMIC_NUMBERS_KEY = "local_atomic_numbers"
    LONG_RANGE_DISTANCE_MATRIX_KEY = "long_range_distance_matrix"
    LONG_RANGE_ATOMIC_FEATURES_KEY = "long_range_atomic_features"
    N_MAX_LOCAL_KEY = "n_max_local"
    N_MAX_LONG_RANGE_KEY = "n_max_long_range"


@dataclass(frozen=True)
class Dataset:
    FILENAME = "df_8molecules_grid_size_1.pkl.gzip"
    NAME = FILENAME.split("_")[0]
    PATH = os.path.join("..", "Datasets", FILENAME)
    FOLDER_NAME = "aspirin_max_local_level_6_grid_size_1_n_samples_50000"  # "df_8molecules_grid_size_1_n_samples_5000"
    FOLDER = os.path.join("..", "Datasets", FOLDER_NAME)  # "df_8molecules_grid_size_1_n_samples_5000"
    TRAIN_NAME = "train.pkl.gzip"
    VAL_NAME = "val.pkl.gzip"
    MAX_ATOM_ELEMENTS = 100
    GRID_SIZE = 1  # in Angstrom
    MAX_LOCAL_LEVEL = 6  # in units of GRID_SIZE
    # has to be <= GRID_SIZE * MAX_LOCAL_LEVEL
    CUT_OFF = GRID_SIZE * MAX_LOCAL_LEVEL  # in Angstrom
    # has to be smaller than CUT_OFF
    INNER_CUT_OFF = CUT_OFF - 0.2  # in Angstrom
    SHUFFLE_BUFFER_SIZE = 1000
    BATCH_SIZE = 64


@dataclass(frozen=True)
class Hyperparameters:
    initial_learning_rate = 2.5e-4  # paper: 5e-4
    # paper: 32e+5 (but uses 2e+7 training samples => updates 6 times per epoch)
    # currently best: 1e+3
    lr_decay_steps = 9e+4  # 18e+4  # TODO: Check if different decay steps for lr are better
    decay_steps = 12e+3
    decay_rate = 0.97  # paper: 0.96
    EPOCHS = 1000
    # paper: start with 0.02 and ends with 1.0
    # currently best: 0.1 and 1.0
    p_energy_start = 0.02
    p_energy_limit = 1.0
    # paper: start with 1000 and ends with 1.0
    # currently best: 100 and 1.0
    p_force_start = 1000
    p_force_limit = 1.0


@dataclass(frozen=True)
class Model:
    small_model = True
    activation = "relu"  # "tanh" or "relu" or "gelu"
    n_max_local = 20  # 8 molecules: 8; aspirin: 6 (grid size 1); aspirin: 11 (grid size 1.5); aspirin: 13 (grid size 1, max local level 2) aspirin: 20 (grid size 1, max local level 4) aspirin: 20 (grid size 1, max local level 6) aspirin: 20  (grid size 1, max local level 8)
    n_max_long_range = 6  # 8 molecules: 20; aspirin: 20 (grid size 1); aspirin: 18 (grid size 1.5);  aspirin: 19 (grid size 1, max local level 2) aspirin: 12 (grid size 1, max local level 4) aspirin: 6 (grid size 1, max local level 6) aspirin: 0  (grid size 1, max local level 8)
    input_shape_local_matrix = (n_max_local, 4)
    input_shape_atomic_numbers = (n_max_local, 2)
    input_shape_long_range_matrix = (n_max_long_range, 4)
    input_shape_long_range_atomic_features = (n_max_long_range, Dataset.MAX_ATOM_ELEMENTS + 1)
    M1 = 100  # paper uses 100
    M2 = 4  # paper uses 4
    embedding_dims = [50, 100]  # paper uses [50, 100]
    predict_only_energy = False


@dataclass(frozen=True)
class Logging:
    epoch_key = "Epoch"
    train_total_loss_key = "Train Total Loss"
    train_energy_loss_key = "Energy Loss"
    train_force_loss_key = "Force Loss"
    val_total_loss_key = "Val Total Loss"
    val_energy_loss_key = "Val Energy Loss"
    val_force_loss_key = "Val Force Loss"
    p_energy_key = "P Energy"
    p_force_key = "P Force"
