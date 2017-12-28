from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent

from openaerostruct_v2.utils.vector_algebra import compute_cross, compute_cross_deriv1, compute_cross_deriv2

from openaerostruct_v2.utils.misc_utils import get_array_indices


class VLMPanelCoeffsFactorComp(ExplicitComponent):
    """
    Lift and drag coefficients by section, by surface.
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

        sec_C_L_factor_name = 'sec_C_L_factor'
        self.add_output(sec_C_L_factor_name, shape=(num_nodes, system_size, 3))

        out_indices = get_array_indices(num_nodes, system_size, 3)

        ind1, ind2 = 0, 0
        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            ind2 += (num_points_x - 1) * (num_points_z - 1)

            sec_C_L_name = '{}_sec_C_L'.format(lifting_surface_name)
            sec_C_L_capped_name = '{}_sec_C_L_capped'.format(lifting_surface_name)

            self.add_input(sec_C_L_name, shape=(num_nodes, num_points_z - 1))
            self.add_input(sec_C_L_capped_name, shape=(num_nodes, num_points_z - 1))

            in_indices = np.einsum('ik,jl->ijkl',
                get_array_indices(num_nodes, num_points_z - 1), np.ones((num_points_x - 1, 3), int))

            rows = out_indices[:, ind1:ind2, :].flatten()
            cols = in_indices.flatten()
            self.declare_partials(sec_C_L_factor_name, sec_C_L_name, rows=rows, cols=cols)
            self.declare_partials(sec_C_L_factor_name, sec_C_L_capped_name, rows=rows, cols=cols)

            ind1 += (num_points_x - 1) * (num_points_z - 1)

        self.set_check_partial_options('*', method='cs')

    def compute(self, inputs, outputs):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']

        sec_C_L_factor_name = 'sec_C_L_factor'

        ind1, ind2 = 0, 0
        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            ind2 += (num_points_x - 1) * (num_points_z - 1)

            sec_C_L_name = '{}_sec_C_L'.format(lifting_surface_name)
            sec_C_L_capped_name = '{}_sec_C_L_capped'.format(lifting_surface_name)

            outputs[sec_C_L_factor_name][:, ind1:ind2, :] = 1.
            outputs[sec_C_L_factor_name][:, ind1:ind2, 1] = np.einsum('ik,j->ijk',
                inputs[sec_C_L_capped_name] / inputs[sec_C_L_name], np.ones(num_points_x - 1),
            ).reshape((num_nodes, (num_points_x - 1) * (num_points_z - 1)))

            ind1 += (num_points_x - 1) * (num_points_z - 1)

    def compute_partials(self, inputs, partials):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']

        sec_C_L_factor_name = 'sec_C_L_factor'

        ind1, ind2 = 0, 0
        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            ind2 += (num_points_x - 1) * (num_points_z - 1)

            sec_C_L_name = '{}_sec_C_L'.format(lifting_surface_name)
            sec_C_L_capped_name = '{}_sec_C_L_capped'.format(lifting_surface_name)

            derivs = partials[sec_C_L_factor_name, sec_C_L_name].reshape(
                (num_nodes, num_points_x - 1, num_points_z - 1, 3)
            )
            derivs[:, :, :, :] = 0.
            derivs[:, :, :, 1] = np.einsum('ik,j->ijk',
                -inputs[sec_C_L_capped_name] / inputs[sec_C_L_name] ** 2, np.ones(num_points_x - 1),
            )

            derivs = partials[sec_C_L_factor_name, sec_C_L_capped_name].reshape(
                (num_nodes, num_points_x - 1, num_points_z - 1, 3)
            )
            derivs[:, :, :, :] = 0.
            derivs[:, :, :, 1] = np.einsum('ik,j->ijk',
                1. / inputs[sec_C_L_name], np.ones(num_points_x - 1),
            )

            ind1 += (num_points_x - 1) * (num_points_z - 1)
