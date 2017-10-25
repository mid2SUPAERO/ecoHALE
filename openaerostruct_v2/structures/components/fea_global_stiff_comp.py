from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent

from openaerostruct_v2.utils.vector_algebra import add_ones_axis
from openaerostruct_v2.utils.vector_algebra import compute_norm, compute_norm_deriv
from openaerostruct_v2.utils.vector_algebra import compute_cross, compute_cross_deriv1, compute_cross_deriv2


class FEAGlobalStiffComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('lifting_surfaces', type_=list)

    def setup(self):
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            size = 6 * num_points_z + 6

            local_name = '{}_local_stiff'.format(lifting_surface_name)
            global_name = '{}_global_stiff'.format(lifting_surface_name)

            self.add_input(local_name, shape=(num_points_z - 1, 12, 12))
            self.add_output(global_name, shape=(size, size))

            arange = np.arange(num_points_z - 1)

            rows = np.empty((num_points_z - 1, 12, 12), int)
            for i in range(12):
                for j in range(12):
                    mtx_i = 6 * arange + i
                    mtx_j = 6 * arange + j
                    rows[:, i, j] = size * mtx_i + mtx_j
            rows = rows.flatten()
            cols = np.arange(144 * (num_points_z - 1))
            self.declare_partials(global_name, local_name, val=1., rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            size = 6 * num_points_z + 6

            local_name = '{}_local_stiff'.format(lifting_surface_name)
            global_name = '{}_global_stiff'.format(lifting_surface_name)

            arange = np.arange(num_points_z - 1)

            outputs[global_name] = 0.
            for i in range(12):
                for j in range(12):
                    outputs[global_name][6 * arange + i, 6 * arange + j] += inputs[local_name][:, i, j]

            mid_node_index = lifting_surface_data['num_points_z_half'] - 1
            index = 6 * mid_node_index
            num_dofs = 6 * num_points_z

            arange = np.arange(6)

            outputs[global_name][index + arange, num_dofs + arange] = 1.
            outputs[global_name][num_dofs + arange, index + arange] = 1.
