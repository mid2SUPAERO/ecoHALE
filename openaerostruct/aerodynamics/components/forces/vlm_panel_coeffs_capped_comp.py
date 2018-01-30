from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent

from openaerostruct.utils.vector_algebra import compute_cross, compute_cross_deriv1, compute_cross_deriv2

from openaerostruct.utils.misc_utils import get_array_indices, tile_sparse_jac


r = -10.

class VLMPanelCoeffsCappedComp(ExplicitComponent):
    """
    Lift and drag coefficients by section, by surface.
    """

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)
        self.metadata.declare('lifting_surfaces', types=list)

    def setup(self):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']

        self.add_input('C_l_max', shape=num_nodes, val=1.)

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            sec_C_L_name = '{}_sec_C_L'.format(lifting_surface_name)
            sec_C_L_capped_name = '{}_sec_C_L_capped'.format(lifting_surface_name)

            self.add_input(sec_C_L_name, shape=(num_nodes, num_points_z - 1))
            self.add_output(sec_C_L_capped_name, shape=(num_nodes, num_points_z - 1))

            indices = get_array_indices(num_nodes, num_points_z - 1).flatten()
            C_l_max_indices = np.outer(get_array_indices(num_nodes), np.ones(num_points_z - 1, int)).flatten()

            self.declare_partials(sec_C_L_capped_name, sec_C_L_name, rows=indices, cols=indices)
            self.declare_partials(sec_C_L_capped_name, 'C_l_max', rows=indices, cols=C_l_max_indices)

        self.set_check_partial_options('*', method='cs')

    def compute(self, inputs, outputs):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            C_l_max = np.outer(inputs['C_l_max'], np.ones(num_points_z - 1))

            sec_C_L_name = '{}_sec_C_L'.format(lifting_surface_name)
            sec_C_L_capped_name = '{}_sec_C_L_capped'.format(lifting_surface_name)

            CL_max = np.minimum(inputs[sec_C_L_name], C_l_max)

            outputs[sec_C_L_capped_name] = CL_max + 1. / r \
                * np.log(np.exp(r * (inputs[sec_C_L_name] - CL_max)) + np.exp(r * (C_l_max - CL_max)))

    def compute_partials(self, inputs, partials):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            C_l_max = np.outer(inputs['C_l_max'], np.ones(num_points_z - 1))

            sec_C_L_name = '{}_sec_C_L'.format(lifting_surface_name)
            sec_C_L_capped_name = '{}_sec_C_L_capped'.format(lifting_surface_name)

            CL_max = np.minimum(inputs[sec_C_L_name], C_l_max)

            partials[sec_C_L_capped_name, sec_C_L_name] = (1. / r \
                / (np.exp(r * (inputs[sec_C_L_name] - CL_max)) + np.exp(r * (C_l_max - CL_max))) \
                * (np.exp(r * (inputs[sec_C_L_name] - CL_max)) * r)
            ).flatten()

            partials[sec_C_L_capped_name, 'C_l_max'] = (1. / r \
                / (np.exp(r * (inputs[sec_C_L_name] - CL_max)) + np.exp(r * (C_l_max - CL_max))) \
                * (np.exp(r * (C_l_max - CL_max)) * r)
            ).flatten()
