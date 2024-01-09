from dataclasses import dataclass


# Hyperparameters
@dataclass(frozen=True)
class Keys:
    ATOMS_KEY = "ase_atoms"
    ENERGY_KEY = "energy_corrected_per_atom"
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
    MAX_ATOM_ELEMENTS = 100
    SHUFFLE_BUFFER_SIZE = 1000
    BATCH_SIZE = 64


@dataclass(frozen=True)
class Model:
    input_shape_local_matrix = (8, 4)
    input_shape_atomic_numbers = (8, 2)
    input_shape_long_range_matrix = (124, 4)
    input_shape_long_range_atomic_features = (124, Dataset.MAX_ATOM_ELEMENTS + 1)
    M1 = 32
    M2 = 32
    embedding_size = 10
