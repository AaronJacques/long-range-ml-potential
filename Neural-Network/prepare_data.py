import pandas as pd
import numpy as np
from tqdm import tqdm
import ase
from ase import Atoms, Atom


def deserialize_df(df):
    df["ase_atoms"] = [Atoms(atom) for atom in df["ase_atoms"]]
    return df


class GridPoint:
    def __init__(self, grid_index, atoms):
        self.grid_index = grid_index
        self.atoms = atoms
        self.neighbours = []
        # each grid point has a dictionary of neighbour trees
        # the key is the level of the neighbour tree and the value
        # is a list of grid points at that level
        # level: 0 is direct neighbours, 1 is second-degree neighbours, etc.
        self.neighbour_tree = {}

    def __eq__(self, other):
        return self.grid_index == other.grid_index

    def __hash__(self):
        return hash(self.grid_index)

    def __str__(self):
        atoms_str = ", ".join([str(atom) for atom in self.atoms])
        return f"GridPoint(index: {self.grid_index}, atoms: {atoms_str})"

    def __repr__(self):
        return self.__str__()

    # add an atom to the grid point
    def add_atom(self, atom):
        self.atoms.append(atom)

    # generate neighbour tree
    # The node is the current grid point.
    # The next level down are the direct neighbors of the current grid point.
    # The level below are the direct neighbors of the direct neighbors, i.e. the second-degree neighbors.
    def add_neighbours(self, grid):
        # loop over the points in the grid and
        # assign them a neighbour level
        for coord, grid_point in grid.items():
            diff = np.abs(np.array(coord) - np.array(self.grid_index))

            # get the level of the neighbour
            level = np.max(diff)

            # add the neighbour to the neighbour tree
            if level not in self.neighbour_tree:
                self.neighbour_tree[level] = []
            self.neighbour_tree[level].append(grid_point)


def create_grid(atoms, grid_size):
    # Determine the grid boundaries
    min_values = np.min(atoms.positions, axis=0)
    max_values = np.max(atoms.positions, axis=0)

    # Create an empty dictionary
    grid = {}

    # Assign each point to the corresponding grid box
    for i, point in tqdm(enumerate(atoms.positions), total=len(atoms.positions), desc="Creating grid"):
        grid_coordinates = tuple(((point - min_values) / grid_size).astype(int))
        if grid_coordinates not in grid:
            grid[grid_coordinates] = GridPoint(grid_coordinates, [])
        grid[grid_coordinates].add_atom(atoms[i])

    # add neighbours
    for grid_point in tqdm(grid.values(), total=len(grid), desc="Adding neighbours"):
        grid_point.add_neighbours(grid)

    return list(grid.values())

