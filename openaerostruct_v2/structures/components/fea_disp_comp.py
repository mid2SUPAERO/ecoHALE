from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent

from openaerostruct_v2.utils.misc_utils import get_array_indices, tile_sparse_jac


class FEADispComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)
        self.metadata.declare('lifting_surfaces', types=list)

    def setup(self):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            size = 6 * num_points_z + 6

            states_name = '{}_states'.format(lifting_surface_name)
            disp_name = '{}_disp'.format(lifting_surface_name)

            self.add_input(states_name, shape=(num_nodes, size))
            self.add_output(disp_name, shape=(num_nodes, num_points_z, 6))

            arange = np.arange(6 * num_points_z)
            _, rows, cols = tile_sparse_jac(1., arange, arange,
                num_points_z * 6, size, num_nodes)
            self.declare_partials(disp_name, states_name, val=1., rows=rows, cols=cols)

        self.set_check_partial_options('*', method='cs')

    def compute(self, inputs, outputs):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            size = 6 * num_points_z + 6

            states_name = '{}_states'.format(lifting_surface_name)
            disp_name = '{}_disp'.format(lifting_surface_name)

            outputs[disp_name] = inputs[states_name][:, :6 * num_points_z].reshape(
                (num_nodes, num_points_z, 6))
