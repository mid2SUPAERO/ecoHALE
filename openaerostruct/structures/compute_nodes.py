from __future__ import print_function, division
import numpy as np

from openmdao.api import ExplicitComponent

try:
    import OAS_API
    fortran_flag = True
    data_type = float
except:
    fortran_flag = False
    data_type = complex


class ComputeNodes(ExplicitComponent):
    """
    Compute FEM nodes based on aerodynamic mesh.

    The FEM nodes are placed at fem_origin * chord,
    with the default fem_origin = 0.35.

    Parameters
    ----------
    mesh[nx, ny, 3] : numpy array
        Array defining the nodal points of the lifting surface.

    Returns
    -------
    nodes[ny, 3] : numpy array
        Flattened array with coordinates for each FEM node.

    """

    def initialize(self):
        self.metadata.declare('surface', type_=dict)

    def initialize_variables(self):
        surface = self.metadata['surface']

        self.ny = surface['num_y']
        self.nx = surface['num_x']
        self.fem_origin = surface['fem_origin']

        self.add_input('mesh', val=np.random.random_sample((self.nx, self.ny, 3)))#, dtype=data_type))
        self.add_output('nodes', val=np.random.random_sample((self.ny, 3)))#, dtype=data_type))

    def compute(self, inputs, outputs):
        w = self.fem_origin
        mesh = inputs['mesh']
        outputs['nodes'] = (1-w) * mesh[0, :, :] + w * mesh[-1, :, :]

    def initialize_partials(self):
        w = self.fem_origin
        n = self.ny * 3

        nodes_mesh = np.zeros((n, n * self.nx))

        nodes_mesh[:n, :n] = np.eye(n) * (1-w)
        nodes_mesh[:n, -n:] = np.eye(n) * w

        self.declare_partials('nodes', 'mesh', val=nodes_mesh)
