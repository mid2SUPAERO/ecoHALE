from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent

from openaerostruct_v2.utils.vector_algebra import add_ones_axis
from openaerostruct_v2.utils.vector_algebra import compute_norm, compute_norm_deriv
from openaerostruct_v2.utils.vector_algebra import compute_cross, compute_cross_deriv1, compute_cross_deriv2


row_indices = np.arange(12)
col_indices = np.array([
    0, 6,
    3, 9,
    2, 4, 8, 10,
    1, 5, 7, 11,
])
mtx = np.zeros((12, 12))
mtx[row_indices, col_indices] = 1.


class FEALocalStiffPermutedComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('lifting_surfaces', type_=list)

    def setup(self):
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            in_name = '{}_local_stiff'.format(lifting_surface_name)
            out_name = '{}_local_stiff_permuted'.format(lifting_surface_name)

            self.add_input(in_name, shape=(num_points_z - 1, 12, 12))
            self.add_output(out_name, shape=(num_points_z - 1, 12, 12))

            indices = np.arange(144 * (num_points_z - 1)).reshape((num_points_z - 1, 12, 12))
            ones = np.ones((12, 12), int)

            data = np.empty((num_points_z - 1, 12, 12, 12, 12))
            # for row1 in range(12):
            #     for col1 in range(12):
            #         for row2 in range(12):
            #             for col2 in range(12):
            #                 data[:, row1, col1, row2, col2] = mtx[row1, row2] * mtx[col2, col1]
            data = np.einsum('i,jl,mk->ijklm', np.ones(num_points_z - 1), mtx.T, mtx)

            data = data.flatten()
            rows = np.einsum('ijk,lm->ijklm', indices, ones).flatten()
            cols = np.einsum('ilm,jk->ijklm', indices, ones).flatten()
            self.declare_partials(out_name, in_name, val=data, rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            in_name = '{}_local_stiff'.format(lifting_surface_name)
            out_name = '{}_local_stiff_permuted'.format(lifting_surface_name)

            outputs[out_name] = np.einsum('jl,ilm,mk->ijk', mtx.T, inputs[in_name], mtx)
