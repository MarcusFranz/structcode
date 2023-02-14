import datetime as dt
import concurrent.futures
import numpy as np
import pandas as pd


class TrussStructure:
    def __init__(self, analysis_name: str, directory: str):
        """
        Creates new object in the class TrussStructure. Contains all the truss data and methods.

        :param analysis_name: Name of analysis.
        :param directory: Directory where truss files are located.
        """
        self.name = analysis_name
        self.date_created = dt.datetime.now()
        self.n_displacements = None
        self.n_forces = None
        self.forces = None
        self.n_elements = None
        self.elements = None
        self.nodes = None
        self.displacements = None
        self.n_nodes = None
        self.n_dimensions = None
        self.global_connector = None
        self.k_element = None
        self.n_dof = None
        self.u = None
        self.k_reduced = None
        self.forces_reduced = None
        self.u_reduced = None
        self.import_data(directory)
        self.create_global_connector_fast()
        self.create_k_element_2d_multi()
        self.create_k_reduced_2d()
        self.solve()
        self.u_remap()


    def __str__(self):
        return f"{self.name} analysis, created on {self.date_created}"

    def import_data(self, directory: str):
        """
        Imports data from files called: displacements, forces, nodes, and elements that are in directory.

        :param directory: Directory where truss files are located.
        """
        # Checking if the directory string given ends with a /
        if directory[-1] == '/':
            directory = directory[:-1]

        # Importing the data from the files in directory
        self.displacements = pd.read_csv(f"{directory}/displacements", delimiter='\t', header=None,
                                         names=['Node', 'DOF', 'Value'], skiprows=1, keep_default_na=False,
                                         index_col=False)
        self.nodes = pd.read_csv(f"{directory}/nodes", delimiter='\t', header=None, names=['Node', 'X', 'Y', 'Z'],
                                 skiprows=1, keep_default_na=True, index_col=False,
                                 dtype={'Node': int, 'X': float, 'Y': float, 'Z': float})
        self.n_dimensions = 3
        if self.nodes['Z'].isnull().values.any():
            """
            Checks if the Z values are NaN if they are it replaces them with zero and sets number of dimensions to 2. 
            """
            self.nodes.loc[:, 'Z'] = 0
            self.n_dimensions = 2

        self.elements = pd.read_csv(f"{directory}/elements", delimiter='\t', header=None,
                                    names=['Element', 'Node 1', 'Node 2', 'Youngs Modulus', 'Cross Sectional Area'],
                                    skiprows=1, keep_default_na=False, index_col=False,
                                    dtype={'Element': int, 'Node 1': int, 'Node 2': int, 'Youngs Modulus': float,
                                           'Cross Sectional Area': float})

        self.forces = pd.read_csv(f"{directory}/forces", delimiter='\t', header=None, names=['Node', 'DOF', 'Value'],
                                  skiprows=1, keep_default_na=False, index_col=False)

        # Determining the number of nodes, elements, forces, and displacements.
        self.n_nodes = self.nodes.shape[0]
        self.n_elements = self.elements.shape[0]
        self.n_forces = self.forces.shape[0]
        self.n_displacements = self.displacements.shape[0]

    def create_global_connector_fast(self):
        """
        Creates the global connector array, using a 0 index system.
        """
        self.n_dof = self.n_dimensions * self.n_nodes - self.n_displacements
        self.displacements.sort_values(by=['Node', 'DOF'], inplace=True)
        global_connector = np.arange(self.n_nodes * self.n_dimensions).reshape((self.n_nodes, self.n_dimensions))
        for row_of_displacements in self.displacements.itertuples():
            dof_num = global_connector[row_of_displacements[1] - 1, row_of_displacements[2] - 1]
            mask = global_connector > dof_num
            global_connector[mask] -= 1
            global_connector[
                row_of_displacements[1] - 1, row_of_displacements[2] - 1] = self.n_nodes * self.n_dimensions - 1
        self.global_connector = global_connector

    def create_k_element_2d(self):
        k_elements = np.empty((4, 4, 0))
        for row in self.elements.itertuples():
            E = row[4]
            A = row[5]
            node_1_pos = self.nodes.loc[self.nodes['Node'] == row[2], ('X', 'Y')]
            node_2_pos = self.nodes.loc[self.nodes['Node'] == row[3], ('X', 'Y')]
            dx = node_2_pos['X'].item() - node_1_pos['X'].item()
            dy = node_2_pos['Y'].item() - node_1_pos['Y'].item()
            c2 = dx ** 2 / (dx ** 2 + dy ** 2)
            s2 = dy ** 2 / (dx ** 2 + dy ** 2)
            cs = (dy * dx) / (dx ** 2 + dy ** 2)
            k_elements_iter = np.array(
                [[c2, cs, -c2, -cs],
                 [cs, s2, -cs, -s2],
                 [-c2, -cs, c2, cs],
                 [-cs, -s2, cs, s2]]) * (E * A) / np.linalg.norm([dx, dy])
            k_elements = np.append(k_elements, np.expand_dims(k_elements_iter, axis=2), axis=2)
        self.k_element = k_elements

    def create_k_element_2d_multi(self):
        def process_row(row):
            E = row[4]
            A = row[5]
            node_1_pos = self.nodes.loc[self.nodes['Node'] == row[2], ('X', 'Y')]
            node_2_pos = self.nodes.loc[self.nodes['Node'] == row[3], ('X', 'Y')]
            dx = node_2_pos['X'].item() - node_1_pos['X'].item()
            dy = node_2_pos['Y'].item() - node_1_pos['Y'].item()
            c2 = dx ** 2 / (dx ** 2 + dy ** 2)
            s2 = dy ** 2 / (dx ** 2 + dy ** 2)
            cs = (dy * dx) / (dx ** 2 + dy ** 2)
            k_elements_iter = np.array(
                [[c2, cs, -c2, -cs],
                 [cs, s2, -cs, -s2],
                 [-c2, -cs, c2, cs],
                 [-cs, -s2, cs, s2]]) * (E * A) / np.linalg.norm([dx, dy])
            return k_elements_iter

        k_elements = np.empty((4, 4, 0))

        def make_array(row):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                return executor.map(process_row, row)

        results = make_array(self.elements.itertuples())
        for result in results:
            k_elements = np.append(k_elements, np.expand_dims(result, axis=2), axis=2)

        self.k_element = k_elements

    def create_k_reduced_2d(self):
        self.u = np.zeros((self.n_nodes, self.n_dimensions), dtype=float)
        self.u[self.displacements.loc[:, 'Node'] - 1, self.displacements.loc[:, 'DOF'] - 1] = self.displacements.loc[
                :, 'Value']
        self.forces_reduced = np.zeros(self.n_dof)
        self.forces_reduced[self.global_connector[self.forces['Node'] - 1, self.forces['DOF'] - 1]] += self.forces['Value']

        k_reduced = np.zeros((self.n_dof, self.n_dof))
        for element in range(self.n_elements):
            for local_node_1 in range(2):
                for local_xy_1 in range(self.n_dimensions):
                    local_dof_1 = self.n_dimensions * local_node_1 + local_xy_1
                    global_node_1 = int(self.elements.to_numpy()[element, local_node_1 + 1] - 1)
                    global_dof_1 = self.global_connector[global_node_1, local_xy_1]
                    if global_dof_1 > self.n_dof - 1:
                        continue
                    for local_node_2 in range(2):
                        for local_xy_2 in range(self.n_dimensions):
                            local_dof_2 = self.n_dimensions * local_node_2 + local_xy_2
                            global_node_2 = int(self.elements.to_numpy()[element, local_node_2 + 1] - 1)
                            global_dof_2 = self.global_connector[global_node_2, local_xy_2]
                            if global_dof_2 > self.n_dof - 1:
                                self.forces_reduced[global_dof_1] = self.forces_reduced[global_dof_1] - self.k_element[
                                    local_dof_1, local_dof_2, element] * self.u[global_node_2, local_xy_2]
                            else:
                                k_reduced[global_dof_1, global_dof_2] = k_reduced[global_dof_1, global_dof_2] + \
                                                                       self.k_element[
                                                                           local_dof_1, local_dof_2, element]

        self.k_reduced = k_reduced

    def solve(self):
        self.u_reduced = np.linalg.solve(self.k_reduced, self.forces_reduced)

    def u_remap(self):
        for i in range(self.n_nodes):
            for j in range(self.n_dimensions):
                dof = self.global_connector[i, j]
                if dof < self.n_dof:
                    self.u[i, j] = self.u_reduced[dof]


