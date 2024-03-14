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
    FOLDER_NAME = "aspirin_grid_size_1_n_samples_40000"  # "df_8molecules_grid_size_1_n_samples_5000"
    FOLDER = os.path.join("..", "Datasets", FOLDER_NAME)  # "df_8molecules_grid_size_1_n_samples_5000"
    TRAIN_NAME = "train.pkl.gzip"
    VAL_NAME = "val.pkl.gzip"
    MAX_ATOM_ELEMENTS = 100
    SHUFFLE_BUFFER_SIZE = 1000
    BATCH_SIZE = 64


@dataclass(frozen=True)
class Hyperparameters:
    learning_rate = 5e-4  # paper: 5e-4
    decay_steps = 1e+3  # paper: 32e+5
    decay_rate = 0.97  # paper: 0.96
    EPOCHS = 1000
    # paper: start with 0.02 and ends with 1.0
    p_energy_start = 0.1
    p_energy_limit = 1.0
    # paper: start with 1000 and ends with 1.0
    p_force_start = 500
    p_force_limit = 1.0


@dataclass(frozen=True)
class Model:
    small_model = False
    n_max_local = 6  # 8 molecules: 8; aspirin: 6
    n_max_long_range = 20  # 8 molecules: 20; aspirin: 20
    input_shape_local_matrix = (n_max_local, 4)
    input_shape_atomic_numbers = (n_max_local, 2)
    input_shape_long_range_matrix = (n_max_long_range, 4)
    input_shape_long_range_atomic_features = (n_max_long_range, Dataset.MAX_ATOM_ELEMENTS + 1)
    M1 = 100  # paper uses 100
    M2 = 4  # paper uses 4
    embedding_dims = [50, 100]  # paper uses [50, 100]
