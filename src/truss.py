import datetime as dt
import concurrent.futures
import numpy as np
import pandas as pd
import uuid


class TrussStructure:
    def __init__(self, analysis_name: str, directory: str):
        """
        Creates new object in the class TrussStructure. Contains all the truss data and methods.

        :param analysis_name: Name of analysis.
        :param directory: Directory where truss files are located.
        """
        self.name = analysis_name
        self.date_created = dt.datetime.now()
        self.directory = directory
        if self.directory[-1] == '/':
            self.directory = self.directory[:-1]
        self.n_displacements, self.n_forces, self.forces, self.n_elements, self.elements, self.nodes, \
            self.displacements, self.n_nodes, self.n_dimensions, self.global_connector, self.k_element, \
            self.n_dof, self.u, self.k_reduced, self.forces_reduced, self.u_reduced = (None,) * 16
        self.import_data()
        self.perform_calculations()

    def __str__(self):
        return f"{self.name} analysis, created on {self.date_created}"

    def import_data(self):
        """
        Imports data from files called: displacements, forces, nodes, and elements that are in directory.
        """
        self.displacements = pd.read_csv(f"{self.directory}/displacements", delimiter='\t', header=None,
                                         names=['Node', 'DOF', 'Value'], skiprows=1, keep_default_na=False,
                                         index_col=False)
        self.nodes = pd.read_csv(f"{self.directory}/nodes", delimiter='\t', header=None, names=['Node', 'X', 'Y', 'Z'],
                                 skiprows=1, keep_default_na=True, index_col=False,
                                 dtype={'Node': int, 'X': float, 'Y': float, 'Z': float})
        self.n_dimensions = 3
        if self.nodes['Z'].isnull().values.any():
            """
            Checks if the Z values are NaN if they are it replaces them with zero and sets number of dimensions to 2. 
            """
            self.nodes.loc[:, 'Z'] = 0
            self.n_dimensions = 2

        self.elements = pd.read_csv(f"{self.directory}/elements", delimiter='\t', header=None,
                                    names=['Element', 'Node 1', 'Node 2', 'Youngs Modulus', 'Cross Sectional Area'],
                                    skiprows=1, keep_default_na=False, index_col=False,
                                    dtype={'Element': int, 'Node 1': int, 'Node 2': int, 'Youngs Modulus': float,
                                           'Cross Sectional Area': float})

        self.forces = pd.read_csv(f"{self.directory}/forces", delimiter='\t', header=None,
                                  names=['Node', 'DOF', 'Value'],
                                  skiprows=1, keep_default_na=False, index_col=False)

        # Determining the number of nodes, elements, forces, and displacements.
        self.n_nodes = self.nodes.shape[0]
        self.n_elements = self.elements.shape[0]
        self.n_forces = self.forces.shape[0]
        self.n_displacements = self.displacements.shape[0]

    def create_global_connector_fast(self):
        """
        Creates the global connector array, using a 0 index system. This is vectorized and gives the same output for the
        limited test cases I've tried as the regular function.
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

    # def create_k_element_2d(self):
    #     """
    #     Original method of making the k_element matrix, does them one by one.
    #     :return: updates the self.k_elements array which is 4 x 4 * n_elements in size that was initialized as =None
    #     """
    #     k_elements = np.empty((4, 4, 0))
    #     for row in self.elements.itertuples():
    #         e = row[4]
    #         a = row[5]
    #         node_1_pos = self.nodes.loc[self.nodes['Node'] == row[2], ('X', 'Y')]
    #         node_2_pos = self.nodes.loc[self.nodes['Node'] == row[3], ('X', 'Y')]
    #         dx = node_2_pos['X'].item() - node_1_pos['X'].item()
    #         dy = node_2_pos['Y'].item() - node_1_pos['Y'].item()
    #         c2 = dx ** 2 / (dx ** 2 + dy ** 2)
    #         s2 = dy ** 2 / (dx ** 2 + dy ** 2)
    #         cs = (dy * dx) / (dx ** 2 + dy ** 2)
    #         k_elements_iter = np.array(
    #             [[c2, cs, -c2, -cs],
    #              [cs, s2, -cs, -s2],
    #              [-c2, -cs, c2, cs],
    #              [-cs, -s2, cs, s2]]) * (e * a) / np.linalg.norm([dx, dy])
    #         k_elements = np.append(k_elements, np.expand_dims(k_elements_iter, axis=2), axis=2)
    #     self.k_element = k_elements

    def create_k_element_2d_multi(self):
        """
        Uses concurrent futures to do all k_elements at the same time, seems to be a small improvement on the small test
        data from lecture.
        :return: updates the self.k_elements 3d array which is 4 x 4 x n_elements which was initialized as =None
        """

        def process_row(row):
            """
            Actually creates the k_element matrix.
            :param row: tuple from the .itertuple() method of a pandas df.
            :return: np.array of the k_elements for the given row.
            """
            e = row[4]
            a = row[5]
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
                 [-cs, -s2, cs, s2]]) * (e * a) / np.linalg.norm([dx, dy])
            return k_elements_iter

        k_elements = np.empty((4, 4, 0))

        def make_array(row):
            """
            This is what sets up the concurrent part, so they are calculated at the same time.
            :param row: tuple from the .itertuple() method of a pandas df.
            :return: Object of class executor, if you need the data from this you have to loop over it.
            """
            with concurrent.futures.ThreadPoolExecutor() as executor:
                return executor.map(process_row, row)

        results = make_array(self.elements.itertuples())
        for result in results:
            k_elements = np.append(k_elements, np.expand_dims(result, axis=2), axis=2)

        self.k_element = k_elements

    def create_k_reduced_2d(self):
        """
        Followed the algorithm from lecture 11 to create this, still trying to understand fully how it works but seems
        the gist of it is taking the k_elements and mapping them to a k_global if they are going to be kept post
        reduction and as directed by the global_connector array, hopefully I can vectorize this in the future.
        :return: updates the self.k_reduced matrix that was initialized as =None
        """
        self.u = np.zeros((self.n_nodes, self.n_dimensions), dtype=float)
        self.u[self.displacements.loc[:, 'Node'] - 1, self.displacements.loc[:, 'DOF'] - 1] = self.displacements.loc[
                                                                                              :, 'Value']
        self.forces_reduced = np.zeros(self.n_dof)
        self.forces_reduced[self.global_connector[self.forces['Node'] - 1, self.forces['DOF'] - 1]] += self.forces[
            'Value']

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
        """
        Solves for the displacements requires that some version of create_k_reduced has been completed.
        :return: Updates self.u_reduced.
        """
        self.u_reduced = np.linalg.solve(self.k_reduced, self.forces_reduced)

    def u_remap(self):
        """
        Makes the u into a more readable format, this could almost definitely be vectorized.
        :return: Updates self.u
        """
        for i in range(self.n_nodes):
            for j in range(self.n_dimensions):
                dof = self.global_connector[i, j]
                if dof < self.n_dof:
                    self.u[i, j] = self.u_reduced[dof]

    def perform_calculations(self):
        """
        Just made this to group the functions in here to clean up the __init__ method some.
        :return:
        """
        self.create_global_connector_fast()
        self.create_k_element_2d_multi()
        self.create_k_reduced_2d()
        self.solve()
        self.u_remap()

    def generate_output(self, output='print', directory=None):
        """
        Call this to output results, in either LaTeX, .txt, or printed in the console.
        :param output: str either latex, txt, if none it will print the data. Default: print
        :param directory: directory you want output to goto. Default: Same as import directory
        :return:
        """
        if output.lower() == 'latex':
            headers = ["Node \\#", "Displacement u", "Displacement v"]
            table = [headers]
            for i, row in enumerate(self.u):
                table.append([f"{i + 1}", f"{row[0]:.4f}", f"{row[1]:.4f}"])
            latex = '\\begin{table}\n\\centering\n\\caption{Nodal Displacements (units)}\n\\label{' \
                    'tab:node_disp}\n\\small\n'
            latex += '\\begin{tabular}{|c|c|c|}\n\\hline\n'
            for row in table:
                latex += ' & '.join(map(str, row)) + ' \\\\ \\hline\n'
            latex += '\\end{tabular}\n\\end{table}\n'
            if not directory:
                path = self.directory + "/latex_" + f"{uuid.uuid4()}"
                with open(path, 'w') as file:
                    file.write(latex)
        elif output.lower() == 'txt':
            pass
        else:
            pass
