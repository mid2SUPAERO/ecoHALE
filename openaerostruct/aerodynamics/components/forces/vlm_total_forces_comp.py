from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent

from openaerostruct.utils.vector_algebra import compute_cross, compute_cross_deriv1, compute_cross_deriv2

from openaerostruct.utils.misc_utils import get_array_indices, tile_sparse_jac


panel_forces_name = 'panel_forces_rotated'

class VLMTotalForcesComp(ExplicitComponent):
    """
    Total lift and drag.
    """

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)
        self.metadata.declare('lifting_surfaces', types=list)

    def setup(self):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']

        system_size = 0

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            system_size += (num_points_x - 1) * (num_points_z - 1)

        self.system_size = system_size

        self.add_input(panel_forces_name, shape=(num_nodes, system_size, 3))
        self.add_output('lift', shape=num_nodes)
        self.add_output('drag', shape=num_nodes)

        rows = np.zeros(system_size, int)
        cols = np.arange(3 * system_size).reshape((system_size, 3))[:, 1]
        _, rows, cols = tile_sparse_jac(1., rows, cols,
            1, system_size * 3, num_nodes)
        self.declare_partials('lift', panel_forces_name, val=1., rows=rows, cols=cols)

        rows = np.zeros(system_size, int)
        cols = np.arange(3 * system_size).reshape((system_size, 3))[:, 0]
        _, rows, cols = tile_sparse_jac(1., rows, cols,
            1, system_size * 3, num_nodes)
        self.declare_partials('drag', panel_forces_name, val=1., rows=rows, cols=cols)

        self.set_check_partial_options('*', method='cs')

    def compute(self, inputs, outputs):
        outputs['lift'] = np.sum(inputs[panel_forces_name][:, :, 1], axis=1)
        outputs['drag'] = np.sum(inputs[panel_forces_name][:, :, 0], axis=1)
