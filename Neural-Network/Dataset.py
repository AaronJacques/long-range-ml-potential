import os
import numpy as np
import pandas as pd
from Constants import Dataset, Keys, Model
from tqdm import tqdm
import tensorflow as tf


class GridCell:
    def __init__(self, grid_index, positions, atomic_numbers, max_level):
        self.grid_index = grid_index
        self.positions = positions
        self.atomic_numbers = atomic_numbers
        # each grid cell has a dictionary of neighbour trees
        # the key is the level of the neighbour tree and the value
        # is a list of grid cells at that level
        # level: 0 is direct neighbours, 1 is second-degree neighbours, etc.
        self.neighbour_tree = [[] for _ in range(max_level + 1)]
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
        # calculate the level of the neighbour
        current_grid_index = np.array(self.grid_index * len(grid)).reshape(len(grid), -1)
        all_grid_indices = np.array([grid_cell.grid_index for grid_cell in grid])
        diff = np.abs(all_grid_indices - current_grid_index)
        levels = np.max(diff, axis=1)

        for i in range(len(grid)):
            self.neighbour_tree[levels[i]].append(grid[i])

    # get the sum of the atomic numbers of the grid cell
    def get_atomic_number_sum(self):
        return np.sum(self.atomic_numbers)

    def __get_main_group_number(self, atomic_number):
        # Main groups and their typical elements by atomic number
        group_1 = {1, 3, 11, 19, 37, 55, 87}
        group_2 = {4, 12, 20, 38, 56, 88}
        group_3 = {5, 13, 31, 49, 81, 21, 39, 71}
        group_4 = {6, 14, 32, 50, 82, 22, 40, 72}
        group_5 = {7, 15, 33, 51, 83, 23, 41, 73}
        group_6 = {8, 16, 34, 52, 84, 24, 42, 74}
        group_7 = {9, 17, 35, 53, 85, 25, 43, 75}
        group_8 = {2, 10, 18, 36, 54, 86,}

        groups = {
            1: group_1, 2: group_2,
            3: group_3, 4: group_4, 5: group_5, 6: group_6,
            7: group_7, 8: group_8
        }

        for group, elements in groups.items():
            if atomic_number in elements:
                return group
        return None


    # get the center of mass of the grid cell
    # weighted by the atomic numbers
    def get_center_of_mass(self):
        if Dataset.use_charge_number:
            total_mass = self.get_atomic_number_sum()
            total_mass_position = np.sum(
                np.array(self.positions) * np.array(self.atomic_numbers).reshape(-1, 1),
                axis=0
            )
            return total_mass_position / total_mass
        else:
            # weight by the main group number
            main_group_numbers = [self.__get_main_group_number(atomic_number) for atomic_number in self.atomic_numbers]
            total_mass = np.sum(main_group_numbers)
            total_mass_position = np.sum(
                np.array(self.positions) * np.array(main_group_numbers).reshape(-1, 1),
                axis=0
            )
            return total_mass_position / total_mass

    # each index corresponds to the atomic number (0: H, 1: He, etc.)
    # the value is the number of atoms of that type in the grid cell
    # only the first max_atom_elements are considered
    def get_atomic_features(self):
        return self.atomic_features


# create grid with Fast Multipole Method
# positions: list of positions of atoms
# atomic_numbers: list of atomic numbers of atoms
# min_grid_size: minimum grid size
def create_grid_multipole(positions, atomic_numbers, min_grid_size):
    # init arrays
    local_matrix = []
    long_range_matrix = []
    local_atomic_numbers = []
    long_range_atomic_features = []

    # get the max distance in each dimension
    max_values = np.max(positions, axis=0)
    min_values = np.min(positions, axis=0)
    max_length = np.max(max_values - min_values)

    # init grid size
    grid_size = max_length / 4

    # loop over all grid sizes
    while True:
        if grid_size < min_grid_size:
            break

        # get the grid indices
        grid_indices = np.floor((positions - min_values) / grid_size).astype(np.int32)
        unique_indices = np.unique(grid_indices, axis=0)
        # max index in each dimension
        max_index = np.max(grid_indices, axis=0)

        # loop over all atoms
        for i, (position, atomic_number) in enumerate(zip(positions, atomic_numbers)):
            # get the grid index of the atom
            grid_index = grid_indices[i]

            # filter out the cells which are direct neighbours
            long_range_neighbours = unique_indices[
                np.any(np.abs(unique_indices - grid_index) > 1, axis=1) & np.all(np.abs(unique_indices - grid_index) <= 1, axis=1)
            ]

            # TODO: filter out the cells that have been already added in the previous iteration

            # for each neighbour find the atoms in the neighbour cell
            cell_positions = []
            cell_atomic_numbers = []
            for neighbour in long_range_neighbours:
                cell_positions.append([positions[j] for j in range(len(positions)) if np.all(grid_indices[j] == neighbour)])
                cell_atomic_numbers.append([atomic_numbers[j] for j in range(len(atomic_numbers)) if np.all(grid_indices[j] == neighbour)])

            # calculate the center of mass of the neighbour cell
            center_of_mass = np.mean(cell_positions, axis=0)

            # calculate the feature vector of the neighbour cell
            features = np.zeros((len(cell_positions), Dataset.MAX_ATOM_ELEMENTS + 1))
            # first entry is the atomic number of the current atom
            features[:, 0] = atomic_number
            for j, cell in enumerate(cell_atomic_numbers):
                for an in cell:
                    if an > Dataset.MAX_ATOM_ELEMENTS:
                        continue
                    features[j, an] += 1

            # calculate the distance between the atom and the center of mass
            distance_vector = position - center_of_mass

            # append the distance vector to the long range matrix
            long_range_matrix.append(distance_vector)
            long_range_atomic_features.append(features)

        # update the grid size
        grid_size /= 2

    # TODO: calculate the local distance matrix


def create_grid(positions, atomic_numbers, grid_size):
    min_values = np.min(positions, axis=0)
    positions = np.array(positions)
    grid_indices = np.floor((positions - min_values) / grid_size).astype(np.int32)

    # Unique indices with inverse mapping to reconstruct original grid cells
    unique_indices, inverse = np.unique(grid_indices, axis=0, return_inverse=True)

    # Determine the grid dimensions dynamically from max indices in each direction
    max_index = np.max(unique_indices, axis=0)

    # Calculate the distances to the nearest (0) and farthest (max_index) boundaries
    dist_to_min = unique_indices  # Since min is always 0 in our adjusted grid index system
    dist_to_max = max_index - unique_indices

    # Maximum of distances in each dimension
    max_dist = np.maximum(dist_to_min, dist_to_max)

    # Calculate the maximum level for each cell (max value across dimensions)
    max_levels = np.max(max_dist, axis=1)

    # Create structured arrays or similar to hold data
    grid = [
        GridCell(
            tuple(index),
            [],
            [],
            max_levels[i]
        ) for i, index in enumerate(unique_indices)
    ]

    # Add atoms
    for i, (point, atomic_number) in enumerate(zip(positions, atomic_numbers)):
        grid[inverse[i]].add_atom(point, atomic_number)

    # add neighbours
    for grid_point in grid:
        grid_point.add_neighbours(grid)

    return grid


def get_expanded_distance_vector(distance_vector, norm):
    inverse_distance = 1 / norm if norm > 0 else 0
    return np.concatenate([np.array([inverse_distance]), distance_vector * inverse_distance])


def get_s(norm):
    # if the norm is smaller than INNER_CUT_OFF, use 1 / r
    if norm < Dataset.INNER_CUT_OFF:
        return 1 / norm

    # if the norm is INNNER_CUT_OFF < r < CUT_OFF,
    # use 1 / r * (0.5 * cos(pi * (r - INNER_CUT_OFF) / (CUT_OFF - INNER_CUT_OFF)) + 0.5)
    if Dataset.INNER_CUT_OFF <= norm < Dataset.CUT_OFF:
        return 1 / norm * (0.5 * np.cos(np.pi * (norm - Dataset.INNER_CUT_OFF) / (Dataset.CUT_OFF - Dataset.INNER_CUT_OFF))
                           + 0.5)

    # otherwise, use 0
    return 0


def get_expanded_distance_vector_local(distance_vector, norm, s):
    return np.concatenate([np.array([s]), distance_vector * s / norm])


def process_neighbour(neighbour, position, atomic_number, long_range_distance_matrix, current_atomic_features):
    # check if the neighbour is empty
    if len(neighbour.positions) == 0:
        return

    # calculate the distance between the two atoms
    distance_vector = position - neighbour.get_center_of_mass()
    # append the expanded distance vector to the list
    long_range_distance_matrix.append(
        get_expanded_distance_vector(distance_vector, np.linalg.norm(distance_vector, ord=2))
    )
    # add the atomic numbers to the list
    current_atomic_features.append(np.concatenate((np.array([atomic_number]), neighbour.get_atomic_features())))


# creating the distance matrix of a list of grid cells
def create_matrices(grid_points):
    # create the distance matrix
    local_distance_matrices = []
    long_range_distance_matrices = []
    local_atomic_numbers = []
    long_range_atomic_features = []

    N_max_local = 0
    N_max_long_range = 0

    for grid_point in grid_points:
        # loop over all atoms in the grid cell
        for position, atomic_number in zip(grid_point.positions, grid_point.atomic_numbers):
            local_distance_matrix = []
            long_range_distance_matrix = []
            # list of lists of the form [atom_number_1, atom_number_2]
            # where atom_number_1 is the atomic number of the current atom and
            # atom_number_2 is the atomic number of the atom in the distance vector
            current_atomic_numbers = []
            current_atomic_features = []

            for level, neighbours in enumerate(grid_point.neighbour_tree):
                for neighbour in neighbours:
                    # if not a neighbour => long range and process directly
                    if level > Dataset.MAX_LOCAL_LEVEL + 1:
                        process_neighbour(
                            neighbour,
                            position,
                            atomic_number,
                            long_range_distance_matrix,
                            current_atomic_features
                        )
                        continue

                    # initialize new neighbour
                    updated_neighbour = GridCell(neighbour.grid_index, [], [], max_level=1)

                    for (n_index, (neighbour_pos, neighbour_atomic_number)) in enumerate(zip(neighbour.positions, neighbour.atomic_numbers)):
                        # skip the current atom
                        if (neighbour_pos == position).all():
                            continue

                        # calculate the distance between the two atoms
                        distance_vector = position - neighbour_pos
                        norm = np.linalg.norm(distance_vector, ord=2)
                        s = get_s(norm)

                        # skip if outside the cut-off
                        if norm > Dataset.CUT_OFF:
                            # add it to the updated_neighbour
                            updated_neighbour.add_atom(neighbour_pos, neighbour_atomic_number)
                            continue

                        # append the expanded distance vector to the list
                        local_distance_matrix.append(get_expanded_distance_vector_local(distance_vector, norm, s))
                        current_atomic_numbers.append([atomic_number, neighbour_atomic_number])

                    # add the updated neighbour to the list of neighbours
                    # if the level is <= MAX_LOCAL_LEVEL do not add it to the neighbours list because it is a
                    # local neighbour and is already in the local_distance_matrix
                    if level > Dataset.MAX_LOCAL_LEVEL:
                        process_neighbour(
                            updated_neighbour,
                            position,
                            atomic_number,
                            long_range_distance_matrix,
                            current_atomic_features
                        )

            # append
            local_distance_matrices.append(np.array(local_distance_matrix))
            local_atomic_numbers.append(np.array(current_atomic_numbers))
            long_range_distance_matrices.append(np.array(long_range_distance_matrix))
            long_range_atomic_features.append(np.array(current_atomic_features))

            # update the maximum length
            N_max_local = max(N_max_local, len(local_distance_matrix))
            N_max_long_range = max(N_max_long_range, len(long_range_distance_matrix))

    return local_distance_matrices, local_atomic_numbers, long_range_distance_matrices, long_range_atomic_features, N_max_local, N_max_long_range


def pad_df_entry(entry, n_max, expected_width):
    # Check if the entry is empty, and return a zeros array directly
    if len(entry) == 0:
        return np.zeros((1, n_max, expected_width))

    # Preallocate a zeros array with the correct shape
    padded = np.zeros((len(entry), n_max, expected_width), dtype=np.float32)

    # Fill the pre-allocated array where data exists
    for i, x in enumerate(entry):
        if len(x) > 0:
            padded[i, :len(x), :] = x.astype(np.float32)

    return padded


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
    df[Keys.GRID_KEY] = df.apply(lambda row: create_grid(row['positions'], row['atomic_numbers'], grid_size), axis=1, )
    print("Created grids")

    print("Creating matrices...")
    # create matrices for each sample
    df[keys] = df[Keys.GRID_KEY].apply(
        lambda x: pd.Series(create_matrices(x), index=keys)
    )
    # Drop the positions, atomic_numbers and grids columns
    df.drop(['positions', 'atomic_numbers', Keys.GRID_KEY], axis=1, inplace=True)
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
    save_folder = save_folder + f"_max_local_level_{Dataset.MAX_LOCAL_LEVEL}_grid_size_{grid_size}_n_samples_{n_samples_per}"
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
    df_size = len(df.index)

    # get the shapes of the matrices (all matrices have the same shape)
    local_distance_matrix_shape = df[Keys.LOCAL_DISTANCE_MATRIX_KEY].iloc[0].shape[1:]
    local_atomic_numbers_shape = df[Keys.LOCAL_ATOMIC_NUMBERS_KEY].iloc[0].shape[1:]
    long_range_distance_matrix_shape = df[Keys.LONG_RANGE_DISTANCE_MATRIX_KEY].iloc[0].shape[1:]
    long_range_atomic_features_shape = df[Keys.LONG_RANGE_ATOMIC_FEATURES_KEY].iloc[0].shape[1:]

    def generator():
        for _, row in df.iterrows():
            if Model.predict_only_energy:
                yield (
                    row[Keys.LOCAL_DISTANCE_MATRIX_KEY],
                    row[Keys.LOCAL_ATOMIC_NUMBERS_KEY],
                    row[Keys.LONG_RANGE_DISTANCE_MATRIX_KEY],
                    row[Keys.LONG_RANGE_ATOMIC_FEATURES_KEY],
                    np.array([row[Keys.ENERGY_KEY]])
                )
            else:
                yield (
                    row[Keys.LOCAL_DISTANCE_MATRIX_KEY],
                    row[Keys.LOCAL_ATOMIC_NUMBERS_KEY],
                    row[Keys.LONG_RANGE_DISTANCE_MATRIX_KEY],
                    row[Keys.LONG_RANGE_ATOMIC_FEATURES_KEY],
                    row[Keys.FORCE_KEY],
                    np.array([row[Keys.ENERGY_KEY]])
                )

    if Model.predict_only_energy:
        dataset = tf.data.Dataset.from_generator(generator, output_signature=(
            tf.TensorSpec(shape=(None, *local_distance_matrix_shape), dtype=tf.float32),
            tf.TensorSpec(shape=(None, *local_atomic_numbers_shape), dtype=tf.int32),
            tf.TensorSpec(shape=(None, *long_range_distance_matrix_shape), dtype=tf.float32),
            tf.TensorSpec(shape=(None, *long_range_atomic_features_shape), dtype=tf.float32),
            tf.TensorSpec(shape=(1,), dtype=tf.float32)  # (energy,)
        ))
    else:
        dataset = tf.data.Dataset.from_generator(generator, output_signature=(
            tf.TensorSpec(shape=(None, *local_distance_matrix_shape,), dtype=tf.float32),
            tf.TensorSpec(shape=(None, *local_atomic_numbers_shape,), dtype=tf.int32),
            tf.TensorSpec(shape=(None, *long_range_distance_matrix_shape,), dtype=tf.float32),
            tf.TensorSpec(shape=(None, *long_range_atomic_features_shape,), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 3,), dtype=tf.float32),  # (fx, fy, fz)
            tf.TensorSpec(shape=(1,), dtype=tf.float32)  # (energy,)
        ))

    # energies and forces are the labels and the rest is the input
    if Model.predict_only_energy:
        dataset = dataset.map(
            lambda x1, x2, x3, x4, y: ((x1, x2, x3, x4), y),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
    else:
        dataset = dataset.map(
            lambda x1, x2, x3, x4, y1, y2: ((x1, x2, x3, x4), (y1, y2)),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

    # cache the dataset
    dataset = dataset.cache()

    # Shuffle and prefetch
    dataset = dataset.shuffle(buffer_size=Dataset.SHUFFLE_BUFFER_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset, df_size


if __name__ == "__main__":
    fs = [
        './../Datasets/md17_aspirin.npz',
        './../Datasets/md17_benzene2017.npz',
        './../Datasets/md17_ethanol.npz',
        './../Datasets/md17_malonaldehyde.npz',
        './../Datasets/md17_naphthalene.npz',
        './../Datasets/md17_salicylic.npz',
        './../Datasets/md17_toluene.npz',
        './../Datasets/md17_uracil.npz'
    ]
    # files = ['./../Datasets/md17_aspirin.npz']  # asprin
    files = ['./../Datasets/md22_double-walled_nanotube.npz']

    save_folder = './../Datasets/double-walled_nanotube'   #'./../Datasets/aspirin'  # "./../Datasets/df_8molecules"
    create_dataset(files, grid_size=Dataset.GRID_SIZE, save_folder=save_folder, n_samples_per=50000)
