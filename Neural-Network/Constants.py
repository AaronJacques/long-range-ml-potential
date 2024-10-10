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
    FOLDER_NAME = "stachyose_max_local_level_4_grid_size_2"
    FOLDER = os.path.join("..", "Datasets", FOLDER_NAME)
    TRAIN_NAME = "train.pkl.gzip"
    VAL_NAME = "val.pkl.gzip"
    TEST_NAME = "test.pkl.gzip"
    MAX_ATOM_ELEMENTS = 100
    GRID_SIZE = 1 # in Angstrom
    MAX_LOCAL_LEVEL = 6  # in units of GRID_SIZE
    # CUT_OFF is calculated such that all atoms in grid cells with level <= MAX_LOCAL_LEVEL are included
    CUT_OFF = (MAX_LOCAL_LEVEL + 1 + 0.5**0.5) * GRID_SIZE  # in Angstrom
    # has to be smaller than CUT_OFF
    INNER_CUT_OFF = CUT_OFF - 0.2  # in Angstrom
    SHUFFLE_BUFFER_SIZE = 1000
    BATCH_SIZE = 64
    use_charge_number = True
    sort_matrix = False


@dataclass(frozen=True)
class Hyperparameters:
    initial_learning_rate = 5e-5  # paper: 5e-4
    # paper: 32e+5 (but uses 2e+7 training samples => updates 6 times per epoch)
    # currently best: 1e+3
    lr_decay_steps = 8.727e+4  # every 4 epochs (20.8e+4; Stachyose: 8.727e+4)
    decay_steps = 7.5e+3  # 3 times per epoch
    decay_rate = 0.97  # paper: 0.96
    epochs = 200
    # 0.02
    p_energy_start = 1
    p_energy_limit = 1
    p_force_start = 1
    p_force_limit = 0
    M1_local = 30
    M2_local = 4
    M1_long = 6
    M2_long = 2
    embedding_dims = [64, 128, 128, 128]


@dataclass(frozen=True)
class Model:
    activation = "elu"  # "relu" or "elu"
    # aspirin: 20 (grid size 1, max local level 6)
    # Azobenzene: 23 (grid size 1, max local level 6)
    # Paracetamol: 19 (grid size 1, max local level 6)
    # stachyose: 86 (grid size 2, max local level 4)
    # Ac-Ala3-NHMe: 41 (grid size 2, max local level 2)
    # DHA: 55 (grid size 2, max local level 4)
    n_max_local = 86
    # aspirin: 4 (grid size 1, max local level 6)
    # Azobenzene: 11 (grid size 1, max local level 6)
    # Paracetamol: 6 (grid size 1, max local level 6)
    # stachyose: 28 (grid size 2, max local level 4)
    # Ac-Ala3-NHMe: 20 (grid size 2, max local level 2)
    # DHA: 32 (grid size 2, max local level 4)
    n_max_long_range = 28
    input_shape_local_matrix = (n_max_local, 4)
    input_shape_atomic_numbers = (n_max_local, 2)
    input_shape_long_range_matrix = (n_max_long_range, 4)
    input_shape_long_range_atomic_features = (n_max_long_range, Dataset.MAX_ATOM_ELEMENTS + 1)
    predict_only_energy = True
    use_long_range = True


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
