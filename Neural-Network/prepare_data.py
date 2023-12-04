import pandas as pd
import numpy as np
import ase
from ase import Atoms, Atom


def deserialize_df(df):
    df["ase_atoms"] = [Atoms(atom) for atom in df["ase_atoms"]]
    return df


def filter_out_single_atom_molecules(df):
    # filter out single atom molecules
    df = df[df["ase_atoms"].apply(lambda atoms: len(atoms) > 1)]
    # reset the index
    df = df.reset_index(drop=True)
    return df


class GridCell:
    def __init__(self, grid_index, atoms, max_atom_elements=100):
        self.grid_index = grid_index
        self.atoms = atoms
        # each grid cell has a dictionary of neighbour trees
        # the key is the level of the neighbour tree and the value
        # is a list of grid cells at that level
        # level: 0 is direct neighbours, 1 is second-degree neighbours, etc.
        self.neighbour_tree = {}
        self.max_atom_elements = max_atom_elements
        self.atomic_features = self.__create_atomic_features()

    def __eq__(self, other):
        return self.grid_index == other.grid_index

    def __hash__(self):
        return hash(self.grid_index)

    def __str__(self):
        atoms_str = ", ".join([str(atom) for atom in self.atoms])
        return f"GridPoint(index: {self.grid_index}, atoms: {atoms_str})"

    def __repr__(self):
        return self.__str__()

    def __create_atomic_features(self):
        atomic_features = np.zeros(self.max_atom_elements)
        for i, atom in enumerate(self.atoms):
            if i >= self.max_atom_elements:
                continue
            atomic_features[atom.number - 1] += 1
        return atomic_features

    # add an atom to the grid cell
    def add_atom(self, atom):
        self.atoms.append(atom)

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
        return np.sum([atom.number for atom in self.atoms])

    # get the center of mass of the grid cell
    # weighted by the atomic numbers
    def get_center_of_mass(self):
        total_mass = self.get_atomic_number_sum()
        total_mass_position = np.sum([atom.number * atom.position for atom in self.atoms], axis=0)
        return total_mass_position / total_mass

    # each index corresponds to the atomic number (0: H, 1: He, etc.)
    # the value is the number of atoms of that type in the grid cell
    # only the first max_atom_elements are considered
    def get_atomic_features(self):
        return self.atomic_features


def create_grid(atoms, grid_size):
    # Determine the grid boundaries
    min_values = np.min(atoms.positions, axis=0)
    max_values = np.max(atoms.positions, axis=0)

    # Create an empty dictionary
    grid = {}

    # Assign each point to the corresponding grid box
    for i, point in enumerate(atoms.positions):
        grid_coordinates = tuple(((point - min_values) / grid_size).astype(int))
        if grid_coordinates not in grid:
            grid[grid_coordinates] = GridCell(grid_coordinates, [])
        grid[grid_coordinates].add_atom(atoms[i])

    # add neighbours
    for grid_point in grid.values():
        grid_point.add_neighbours(grid)

    return list(grid.values())


# creating the distance matrix of a list of grid cells
# the distance matrix for one atom of a grid cell has the shape (N, 3)
# the function returns a list of distance matrix for all atoms of all grid cells
# and the corresponding list of atomic numbers of the reference atom with shape (N)
# the
def create_distance_matrices(grid_points):
    # create the distance matrix
    distance_matrices = []
    atomic_features = []
    atomic_numbers = []
    for i, grid_point in enumerate(grid_points):
        # loop over all atoms in the grid cell
        for current_atom in grid_point.atoms:
            # TODO: Use NumPy instead of Python lists and calculate the number of distance vectors beforehand
            distance_matrix = []
            # list of lists of the form [atom_number_1, atom_number_2
            # where atom_number_1 is the atomic number of the current atom and
            # atom_number_2 is the atomic number of the atom in the distance vector
            current_atom_numbers = []

            for level, neighbours in grid_point.neighbour_tree.items():
                if level < 2:
                    # for level 0 and 1 we calculate the distance to all atoms directly
                    for neighbour in neighbours:
                        for atom in neighbour.atoms:
                            # skip the current atom
                            if current_atom == atom:
                                continue

                            # calculate the distance between the two atoms
                            distance_matrix.append(current_atom.position - atom.position)
                            # add the atomic numbers to the list
                            current_atom_numbers.append([current_atom.number, atom.number])

                else:
                    # for all other levels we calculate the distance to the center of mass
                    for neighbour in neighbours:
                        # calculate the distance between the two atoms
                        distance_matrix.append(current_atom.position - neighbour.get_center_of_mass())
                        # add the atomic numbers to the list
                        current_atom_numbers.append([current_atom.number, neighbour.get_atomic_number_sum()])

            # append the distance matrix to the list of distance matrices
            distance_matrices.append(np.array(distance_matrix))
            # append the atomic numbers to the list of atomic numbers
            atomic_numbers.append(np.array(current_atom_numbers))

    return distance_matrices, np.array(atomic_numbers)


# create the expanded distance matrix
# the expanded distance matrix has the shape (N, 4) where N is the number of distance vectors
# it contains the normalized distance vector and the inverse of the distance
def create_expanded_distance_matrix(distance_matrices):
    expanded_distance_matrices = []
    for i, distance_matrix in enumerate(distance_matrices):

        # calculate the inverse of the distance
        inverse_distance = np.linalg.norm(distance_matrix, axis=1)
        inverse_distance[inverse_distance == 0] = 1

        # normalize the distance matrix
        normalized_distance_matrix = distance_matrix / inverse_distance[:, None]

        # append the normalized distance matrix and the inverse of the distance to the list
        expanded_distance_matrices.append(np.concatenate([inverse_distance[:, None], normalized_distance_matrix], axis=1))

    return np.array(expanded_distance_matrices)


def get_expanded_distance_matrix(atoms, grid_size):
    # create the grid
    grid_points = create_grid(atoms, grid_size)

    # create the distance matrix
    distance_matrices, atomic_numbers = create_distance_matrices(grid_points)

    # create the expanded distance matrix
    expanded_distance_matrices = create_expanded_distance_matrix(distance_matrices)

    return expanded_distance_matrices, atomic_numbers
