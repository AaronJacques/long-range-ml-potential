import os
import numpy as np
import pandas as pd
from Constants import Dataset, Keys
from tqdm import tqdm
import tensorflow as tf

class GridCell:
    def __init__(self, grid_index, positions, atomic_numbers):
        self.grid_index = grid_index
        self.positions = positions
        self.atomic_numbers = atomic_numbers
        # each grid cell has a dictionary of neighbour trees
        # the key is the level of the neighbour tree and the value
        # is a list of grid cells at that level
        # level: 0 is direct neighbours, 1 is second-degree neighbours, etc.
        self.neighbour_tree = {}
        self.max_atom_elements = Dataset.MAX_ATOM_ELEMENTS
        self.atomic_features = self.__create_atomic_features()

    def __eq__(self, other):
        return self.grid_index == other.grid_index

    def __hash__(self):
        return hash(self.grid_index)

    def __str__(self):
        return f"GridCell(index: {self.grid_index})"

    def __repr__(self):
        return self.__str__()

    def __create_atomic_features(self):
        atomic_features = np.zeros(self.max_atom_elements, dtype=int)
        for atomic_number in self.atomic_numbers:
            if atomic_number > self.max_atom_elements:
                continue
            atomic_features[atomic_number - 1] += 1
        return atomic_features

    # add an atom to the grid cell
    def add_atom(self, position, atomic_number):
        # append this to the positions numpy array
        self.positions.append(np.array(position))
        self.atomic_numbers.append(atomic_number)
        # add the atomic number to the atomic features
        if atomic_number <= self.max_atom_elements:
            self.atomic_features[atomic_number - 1] += 1

    # generate neighbour tree
    # The node is the current grid cell.
    # The next level down are the direct neighbors of the current grid cell.
    # The level below are the direct neighbors of the direct neighbors, i.e. the second-degree neighbors.
    def add_neighbours(self, grid):
        # loop over the points in the grid and
        # assign them a neighbour level
        for coord, grid_cell in grid.items():
            diff = np.abs(np.array(coord) - np.array(self.grid_index))

            # get the level of the neighbour
            level = np.max(diff)

            # add the neighbour to the neighbour tree
            if level not in self.neighbour_tree:
                self.neighbour_tree[level] = []
            self.neighbour_tree[level].append(grid_cell)

    # get the sum of the atomic numbers of the grid cell
    def get_atomic_number_sum(self):
        return np.sum(self.atomic_numbers)

    # get the center of mass of the grid cell
    # weighted by the atomic numbers
    def get_center_of_mass(self):
        total_mass = self.get_atomic_number_sum()
        total_mass_position = np.sum(np.array(self.positions) * np.array(self.atomic_numbers).reshape(-1, 1), axis=0)
        return total_mass_position / total_mass

    # each index corresponds to the atomic number (0: H, 1: He, etc.)
    # the value is the number of atoms of that type in the grid cell
    # only the first max_atom_elements are considered
    def get_atomic_features(self):
        return self.atomic_features


def create_grid(positions, atomic_numbers, grid_size):
    # Determine the grid boundaries
    min_values = np.min(positions, axis=0)
    max_values = np.max(positions, axis=0)

    # Create an empty dictionary
    grid = {}

    # Assign each point to the corresponding grid box
    for i, (point, atomic_number) in enumerate(zip(positions, atomic_numbers)):
        grid_coordinates = tuple(((point - min_values) / grid_size).astype(int))
        if grid_coordinates not in grid:
            grid[grid_coordinates] = GridCell(grid_coordinates, [], [])
        grid[grid_coordinates].add_atom(point, atomic_number)

    # add neighbours
    for grid_point in grid.values():
        grid_point.add_neighbours(grid)

    return list(grid.values())


# create the expanded distance matrix
# the expanded distance matrix has the shape (N, 4) where N is the number of distance vectors
# it contains the normalized distance vector and the inverse of the distance
def create_expanded_distance_matrix(distance_matrices):
    expanded_distance_matrices = []
    for i, distance_matrix in enumerate(distance_matrices):

        if len(distance_matrix) == 0:
            expanded_distance_matrices.append([])
            continue

        # calculate the inverse of the distance
        inverse_distance = np.linalg.norm(distance_matrix, axis=1)
        inverse_distance[inverse_distance == 0] = 1

        # normalize the distance matrix
        normalized_distance_matrix = distance_matrix / inverse_distance[:, None]

        # append the normalized distance matrix and the inverse of the distance to the list
        expanded_distance_matrices.append(np.concatenate([inverse_distance[:, None], normalized_distance_matrix], axis=1))

    return expanded_distance_matrices


# creating the distance matrix of a list of grid cells
# the distance matrix for one atom of a grid cell has the shape (N, 3)
# the function returns a list of distance matrix for all atoms of all grid cells
# and the corresponding list of atomic numbers of the reference atom with shape (N)
# the
def create_matrices(grid_points):
    # create the distance matrix
    local_distance_matrices = []
    long_range_distance_matrices = []
    local_atomic_numbers = []
    long_range_atomic_features = []

    N_max_local = 0
    N_max_long_range = 0

    for i, grid_point in enumerate(grid_points):
        # loop over all atoms in the grid cell
        for position, atomic_number in zip(grid_point.positions, grid_point.atomic_numbers):
            # TODO: Use NumPy instead of Python lists and calculate the number of distance vectors beforehand
            local_distance_matrix = []
            long_range_distance_matrix = []
            # list of lists of the form [atom_number_1, atom_number_2
            # where atom_number_1 is the atomic number of the current atom and
            # atom_number_2 is the atomic number of the atom in the distance vector
            current_atomic_numbers = []
            current_atomic_features = []

            for level, neighbours in grid_point.neighbour_tree.items():
                if level < 2:
                    # for level 0 and 1 we calculate the distance to all atoms directly
                    for neighbour in neighbours:
                        for neighbour_pos, neighbour_atomic_number in zip(neighbour.positions, neighbour.atomic_numbers):
                            # skip the current atom
                            if (neighbour_pos == position).all(): # only compare the positions because no two atoms can have the same position
                                continue

                            # calculate the distance between the two atoms
                            local_distance_matrix.append(position - neighbour_pos)
                            # add the atomic numbers to the list
                            current_atomic_numbers.append([atomic_number, neighbour_atomic_number])

                else:
                    # for all other levels we calculate the distance to the center of mass
                    for neighbour in neighbours:
                        # calculate the distance between the two atoms
                        long_range_distance_matrix.append(position - neighbour.get_center_of_mass())
                        # add the atomic numbers to the list
                        current_atomic_features.append(np.concatenate((np.array([atomic_number]), neighbour.get_atomic_features())))

            # append
            local_distance_matrices.append(np.array(local_distance_matrix))
            local_atomic_numbers.append(np.array(current_atomic_numbers))
            long_range_distance_matrices.append(np.array(long_range_distance_matrix))
            long_range_atomic_features.append(np.array(current_atomic_features))

            # update the maximum length
            N_max_local = max(N_max_local, len(local_distance_matrix))
            N_max_long_range = max(N_max_long_range, len(long_range_distance_matrix))

    # expand the distance matrices
    local_distance_matrices = create_expanded_distance_matrix(local_distance_matrices)
    long_range_distance_matrices = create_expanded_distance_matrix(long_range_distance_matrices)

    return local_distance_matrices, local_atomic_numbers, long_range_distance_matrices, long_range_atomic_features, N_max_local, N_max_long_range


def pad_df_entry(entry, n_max, expected_width):
    # pad the entry with zeros to the maximum length
    if len(entry) == 0:
        return np.zeros((1, n_max, expected_width))

    padded = []
    for x in entry:
        if len(x) == 0:
            padded.append(np.zeros((n_max, expected_width)))
            continue

        padded.append(np.pad(x, ((0, n_max - len(x)), (0, 0)), mode="constant"))

    return np.array(padded)


def create_dataset(paths, grid_size, save_folder, val_split=0.1, n_samples_per=None):
    # create Pandas DataFrame
    df = None
    keys = [Keys.LOCAL_DISTANCE_MATRIX_KEY, Keys.LOCAL_ATOMIC_NUMBERS_KEY, Keys.LONG_RANGE_DISTANCE_MATRIX_KEY,
            Keys.LONG_RANGE_ATOMIC_FEATURES_KEY, Keys.N_MAX_LOCAL_KEY, Keys.N_MAX_LONG_RANGE_KEY]

    for path in tqdm(paths, total=len(paths), desc="Reading files", unit="file"):
        data = np.load(path)
        if n_samples_per is not None:
            size = min(n_samples_per, len(data['E']))
        else:
            size = len(data['E'])
        current_energy = data['E'][:size]  # shape (n_samples)
        current_forces = data['F'][:size]   # shape (n_samples, n_atoms, 3)
        current_positions = data['R'][:size]   # shape (n_samples, n_atoms, 3)
        current_atomic_numbers = data['z']  # shape (n_atoms)
        # convert atomic numbers to shape (n_samples, n_atoms)
        current_atomic_numbers = np.tile(current_atomic_numbers, (len(current_energy), 1))

        # add the data to the DataFrame
        current_df = pd.DataFrame(columns=['energy', 'forces', 'atomic_numbers', 'positions'])
        current_df['energy'] = current_energy.flatten()
        current_df['forces'] = current_forces.tolist()
        current_df['atomic_numbers'] = current_atomic_numbers.tolist()
        current_df['positions'] = current_positions.tolist()
        if df is None:
            df = current_df
        else:
            df = pd.concat([df, current_df], ignore_index=True)

    print("Creating grids...")
    # create grid for each sample
    df[Keys.GRID_KEY] = df.apply(lambda row: create_grid(row['positions'], row['atomic_numbers'], grid_size), axis=1)
    print("Created grids")

    print("Creating matrices...")
    # create matrices for each sample
    df[keys] = df[Keys.GRID_KEY].apply(
        lambda x: pd.Series(create_matrices(x), index=keys)
    )
    # drop the grid column
    df = df.drop(Keys.GRID_KEY, axis=1)
    print("Created matrices")

    print("Padding matrices...")
    # pad the matrices
    # pad the local distance matrices, local atomic numbers, long range distance matrices and long range atomic features
    # with zeros to the maximum length
    n_max_local = df[Keys.N_MAX_LOCAL_KEY].max()
    print(f"n_max_local: {n_max_local}")
    n_max_long_range = df[Keys.N_MAX_LONG_RANGE_KEY].max()
    print(f"n_max_long_range: {n_max_long_range}")
    df[Keys.LOCAL_DISTANCE_MATRIX_KEY] = df[Keys.LOCAL_DISTANCE_MATRIX_KEY].apply(
        lambda x: pad_df_entry(x, n_max_local, 4)
    )
    df[Keys.LOCAL_ATOMIC_NUMBERS_KEY] = df[Keys.LOCAL_ATOMIC_NUMBERS_KEY].apply(
        lambda x: pad_df_entry(x, n_max_local, 2)
    )
    df[Keys.LONG_RANGE_DISTANCE_MATRIX_KEY] = df[Keys.LONG_RANGE_DISTANCE_MATRIX_KEY].apply(
        lambda x: pad_df_entry(x, n_max_long_range, 4)
    )
    df[Keys.LONG_RANGE_ATOMIC_FEATURES_KEY] = df[Keys.LONG_RANGE_ATOMIC_FEATURES_KEY].apply(
        lambda x: pad_df_entry(x, n_max_long_range, Dataset.MAX_ATOM_ELEMENTS + 1)
    )
    print("Padded matrices")

    print("Splitting dataset...")
    # split the dataset
    train_df = df.sample(frac=1 - val_split, random_state=42)
    val_df = df.drop(train_df.index)
    print("Split dataset")

    print("Saving DataFrame...")
    # save the dataframes in the save_folder
    # create the folder if it does not exist
    save_folder = save_folder + f"_grid_size_{grid_size}_n_samples_{n_samples_per}"
    if os.path.exists(save_folder):
        raise FileExistsError(f"The folder {save_folder} already exists")
    os.makedirs(save_folder)
    train_df.to_pickle(os.path.join(save_folder, "train.pkl.gzip"), compression="gzip")
    val_df.to_pickle(os.path.join(save_folder, "val.pkl.gzip"), compression="gzip")
    print("Saved DataFrame")


def create_tf_dataset_force_only(path):
    df = pd.read_pickle(path, compression="gzip")

    local_distance_matrices = np.concatenate(np.array(df[Keys.LOCAL_DISTANCE_MATRIX_KEY]), axis=0)
    local_atomic_numbers = np.concatenate(np.array(df[Keys.LOCAL_ATOMIC_NUMBERS_KEY]), axis=0)
    long_range_distance_matrices = np.concatenate(np.array(df[Keys.LONG_RANGE_DISTANCE_MATRIX_KEY]), axis=0)
    long_range_atomic_features = np.concatenate(np.array(df[Keys.LONG_RANGE_ATOMIC_FEATURES_KEY]), axis=0)
    forces = np.concatenate(np.array(df[Keys.FORCE_KEY]))

    print("-" * 50)
    print("Dataset shapes:")
    print(local_distance_matrices.shape)
    print(local_atomic_numbers.shape)
    print(long_range_distance_matrices.shape)
    print(long_range_atomic_features.shape)
    # print(energies.shape)
    print(forces.shape)
    print("-" * 50)

    def generator():
        for i in range(local_distance_matrices.shape[0]):
            yield (
                local_distance_matrices[i],
                local_atomic_numbers[i],
                long_range_distance_matrices[i],
                long_range_atomic_features[i],
                forces[i]
            )

    dataset = tf.data.Dataset.from_generator(generator, output_signature=(
        tf.TensorSpec(shape=local_distance_matrices.shape[1:], dtype=tf.float32),
        tf.TensorSpec(shape=local_atomic_numbers.shape[1:], dtype=tf.int32),
        tf.TensorSpec(shape=long_range_distance_matrices.shape[1:], dtype=tf.float32),
        tf.TensorSpec(shape=long_range_atomic_features.shape[1:], dtype=tf.float32),
        tf.TensorSpec(shape=(3,), dtype=tf.float32),  # (fx, fy, fz)
    ))

    # forces are the labels and the rest is the input
    dataset = dataset.map(
        lambda x1, x2, x3, x4, y: ((x1, x2, x3, x4), y),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    # shuffle and batch
    dataset = dataset.shuffle(buffer_size=Dataset.SHUFFLE_BUFFER_SIZE)
    dataset = dataset.batch(Dataset.BATCH_SIZE, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


def create_tf_dataset(path):
    print(f"Creating dataset from {path}")
    df = pd.read_pickle(path, compression="gzip")

    # get the shapes of the matrices (all matrices have the same shape)
    local_distance_matrix_shape = df[Keys.LOCAL_DISTANCE_MATRIX_KEY][0].shape[1:]
    local_atomic_numbers_shape = df[Keys.LOCAL_ATOMIC_NUMBERS_KEY][0].shape[1:]
    long_range_distance_matrix_shape = df[Keys.LONG_RANGE_DISTANCE_MATRIX_KEY][0].shape[1:]
    long_range_atomic_features_shape = df[Keys.LONG_RANGE_ATOMIC_FEATURES_KEY][0].shape[1:]

    def generator():
        for _, row in df.iterrows():
            yield (
                row[Keys.LOCAL_DISTANCE_MATRIX_KEY],
                row[Keys.LOCAL_ATOMIC_NUMBERS_KEY],
                row[Keys.LONG_RANGE_DISTANCE_MATRIX_KEY],
                row[Keys.LONG_RANGE_ATOMIC_FEATURES_KEY],
                row[Keys.FORCE_KEY],
                np.array([row[Keys.ENERGY_KEY]])
            )

    dataset = tf.data.Dataset.from_generator(generator, output_signature=(
        tf.TensorSpec(shape=(None, *local_distance_matrix_shape,), dtype=tf.float32),
        tf.TensorSpec(shape=(None, *local_atomic_numbers_shape,), dtype=tf.int32),
        tf.TensorSpec(shape=(None, *long_range_distance_matrix_shape,), dtype=tf.float32),
        tf.TensorSpec(shape=(None, *long_range_atomic_features_shape,), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 3,), dtype=tf.float32),  # (fx, fy, fz)
        tf.TensorSpec(shape=(1,), dtype=tf.float32)  # (energy,)
    ))

    # Shuffle
    dataset = dataset.shuffle(buffer_size=Dataset.SHUFFLE_BUFFER_SIZE)

    # energies and forces are the labels and the rest is the input
    dataset = dataset.map(
        lambda x1, x2, x3, x4, y1, y2: ((x1, x2, x3, x4), (y1, y2)),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


if __name__ == "__main__":
    files = ['./../Datasets/md17_aspirin.npz',
             './../Datasets/md17_benzene2017.npz',
             './../Datasets/md17_ethanol.npz',
             './../Datasets/md17_malonaldehyde.npz',
             './../Datasets/md17_naphthalene.npz',
             './../Datasets/md17_salicylic.npz',
             './../Datasets/md17_toluene.npz',
             './../Datasets/md17_uracil.npz']
    files = [files[0]]  # asprin

    save_folder = './../Datasets/aspirin'  # "./../Datasets/df_8molecules"
    create_dataset(files, grid_size=1, save_folder=save_folder, n_samples_per=40000)
