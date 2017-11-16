from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent

from openaerostruct_v2.utils.vector_algebra import add_ones_axis
from openaerostruct_v2.utils.vector_algebra import compute_norm, compute_norm_deriv
from openaerostruct_v2.utils.vector_algebra import compute_cross, compute_cross_deriv1, compute_cross_deriv2

from openaerostruct_v2.utils.misc_utils import get_array_indices, tile_sparse_jac


coeffs_2 = np.array([
    [ 1., -1.],
    [-1.,  1.],
])

coeffs_y = np.array([
    [ 12.,  -6., -12.,  -6.],
    [ -6.,   4.,   6.,   2.],
    [-12.,   6.,  12.,   6.],
    [ -6.,   2.,   6.,   4.],
])

coeffs_z = np.array([
    [ 12.,   6., -12.,   6.],
    [  6.,   4.,  -6.,   2.],
    [-12.,  -6.,  12.,  -6.],
    [  6.,   2.,  -6.,   4.],
])


class FEALocalStiffComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)
        self.metadata.declare('lifting_surfaces', types=list)

    def setup(self):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            local_name = '{}_local_stiff'.format(lifting_surface_name)

            for in_name_ in ['A', 'Iy', 'Iz', 'J']:
                in_name = '{}_element_{}'.format(lifting_surface_name, in_name_)
                self.add_input(in_name, shape=(num_nodes, num_points_z - 1))

            in_name = '{}_element_{}'.format(lifting_surface_name, 'L')
            self.add_input(in_name, shape=(num_nodes, num_points_z - 1))

            self.add_output(local_name, shape=(num_nodes, num_points_z - 1, 12, 12))

            rows = np.arange(144 * (num_points_z - 1))
            cols = np.outer(np.arange(num_points_z - 1), np.ones(144, int)).flatten()
            _, rows, cols = tile_sparse_jac(1., rows, cols,
                (num_points_z - 1) * 12 * 12, num_points_z - 1, num_nodes)

            for in_name_ in ['A', 'Iy', 'Iz', 'J']:
                in_name = '{}_element_{}'.format(lifting_surface_name, in_name_)
                self.declare_partials(local_name, in_name, rows=rows, cols=cols)

            in_name = '{}_element_{}'.format(lifting_surface_name, 'L')
            self.declare_partials(local_name, in_name, rows=rows, cols=cols)

        self.set_check_partial_options('*', method='cs')

    def compute(self, inputs, outputs):
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            E = lifting_surface_data['E']
            G = lifting_surface_data['G']

            A  = inputs['{}_element_{}'.format(lifting_surface_name, 'A')]
            Iy = inputs['{}_element_{}'.format(lifting_surface_name, 'Iy')]
            Iz = inputs['{}_element_{}'.format(lifting_surface_name, 'Iz')]
            J  = inputs['{}_element_{}'.format(lifting_surface_name, 'J')]
            L  = inputs['{}_element_{}'.format(lifting_surface_name, 'L')]

            local_name = '{}_local_stiff'.format(lifting_surface_name)

            outputs[local_name] = 0.

            for i in range(2):
                for j in range(2):
                    outputs[local_name][:, :, 0 + i, 0 + j] = E * A / L * coeffs_2[i, j]
                    outputs[local_name][:, :, 2 + i, 2 + j] = G * J / L * coeffs_2[i, j]

            for i in range(4):
                for j in range(4):
                    outputs[local_name][:, :, 4 + i, 4 + j] = E * Iy / L ** 3 * coeffs_y[i, j]
                    outputs[local_name][:, :, 8 + i, 8 + j] = E * Iz / L ** 3 * coeffs_z[i, j]

            for i in [1, 3]:
                for j in range(4):
                    outputs[local_name][:, :, 4 + i, 4 + j] *= L
                    outputs[local_name][:, :, 8 + i, 8 + j] *= L
            for i in range(4):
                for j in [1, 3]:
                    outputs[local_name][:, :, 4 + i, 4 + j] *= L
                    outputs[local_name][:, :, 8 + i, 8 + j] *= L

    def compute_partials(self, inputs, partials):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            E = lifting_surface_data['E']
            G = lifting_surface_data['G']

            A_name = '{}_element_{}'.format(lifting_surface_name, 'A')
            Iy_name = '{}_element_{}'.format(lifting_surface_name, 'Iy')
            Iz_name = '{}_element_{}'.format(lifting_surface_name, 'Iz')
            J_name = '{}_element_{}'.format(lifting_surface_name, 'J')
            L_name = '{}_element_{}'.format(lifting_surface_name, 'L')

            A  = inputs[A_name]
            Iy = inputs[Iy_name]
            Iz = inputs[Iz_name]
            J  = inputs[J_name]
            L  = inputs[L_name]

            local_name = '{}_local_stiff'.format(lifting_surface_name)

            derivs_A = partials[local_name, A_name].reshape((num_nodes, num_points_z - 1, 12, 12))
            derivs_Iy = partials[local_name, Iy_name].reshape((num_nodes, num_points_z - 1, 12, 12))
            derivs_Iz = partials[local_name, Iz_name].reshape((num_nodes, num_points_z - 1, 12, 12))
            derivs_J = partials[local_name, J_name].reshape((num_nodes, num_points_z - 1, 12, 12))
            derivs_L = partials[local_name, L_name].reshape((num_nodes, num_points_z - 1, 12, 12))

            derivs_A[:] = 0.
            derivs_Iy[:] = 0.
            derivs_Iz[:] = 0.
            derivs_J[:] = 0.
            derivs_L[:] = 0.

            for i in range(2):
                for j in range(2):
                    derivs_A[:, :, 0 + i, 0 + j] = E / L * coeffs_2[i, j]
                    derivs_L[:, :, 0 + i, 0 + j] = -E * A / L ** 2 * coeffs_2[i, j]

                    derivs_J[:, :, 2 + i, 2 + j] = G / L * coeffs_2[i, j]
                    derivs_L[:, :, 2 + i, 2 + j] = -G * J / L ** 2 * coeffs_2[i, j]

            for i in range(4):
                for j in range(4):
                    derivs_Iy[:, :, 4 + i, 4 + j] = E / L ** 3 * coeffs_y[i, j]
                    derivs_L [:, :, 4 + i, 4 + j] = -3 * E * Iy / L ** 4 * coeffs_y[i, j]

                    derivs_Iz[:, :, 8 + i, 8 + j] = E / L ** 3 * coeffs_z[i, j]
                    derivs_L [:, :, 8 + i, 8 + j] = -3 * E * Iz / L ** 4 * coeffs_z[i, j]

            for i in [1, 3]:
                for j in range(4):
                    derivs_Iy[:, :, 4 + i, 4 + j] = E / L ** 2 * coeffs_y[i, j]
                    derivs_L [:, :, 4 + i, 4 + j] = -2 * E * Iy / L ** 3 * coeffs_y[i, j]

                    derivs_Iz[:, :, 8 + i, 8 + j] = E / L ** 2 * coeffs_z[i, j]
                    derivs_L [:, :, 8 + i, 8 + j] = -2 * E * Iz / L ** 3 * coeffs_z[i, j]
            for i in range(4):
                for j in [1, 3]:
                    derivs_Iy[:, :, 4 + i, 4 + j] = E / L ** 2 * coeffs_y[i, j]
                    derivs_L [:, :, 4 + i, 4 + j] = -2 * E * Iy / L ** 3 * coeffs_y[i, j]

                    derivs_Iz[:, :, 8 + i, 8 + j] = E / L ** 2 * coeffs_z[i, j]
                    derivs_L [:, :, 8 + i, 8 + j] = -2 * E * Iz / L ** 3 * coeffs_z[i, j]

            for i in [1, 3]:
                for j in [1, 3]:
                    derivs_Iy[:, :, 4 + i, 4 + j] = E / L * coeffs_y[i, j]
                    derivs_L [:, :, 4 + i, 4 + j] = -E * Iy / L ** 2 * coeffs_y[i, j]

                    derivs_Iz[:, :, 8 + i, 8 + j] = E / L * coeffs_z[i, j]
                    derivs_L [:, :, 8 + i, 8 + j] = -E * Iz / L ** 2 * coeffs_z[i, j]
