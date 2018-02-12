from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent

from openaerostruct.utils.misc_utils import get_array_indices, tile_sparse_jac


class FEAVolumeComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)
        self.metadata.declare('lifting_surfaces', types=list)

    def setup(self):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']

        self.add_output('structural_volume', shape=num_nodes)

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_z = 2 * lifting_surface_data.num_points_z_half - 1

            A_name = '{}_element_{}'.format(lifting_surface_name, 'A')
            L_name = '{}_element_{}'.format(lifting_surface_name, 'L')

            self.add_input(A_name, shape=(num_nodes, num_points_z - 1))
            self.add_input(L_name, shape=(num_nodes, num_points_z - 1))

            rows = np.zeros(num_points_z - 1, int)
            cols = get_array_indices(num_points_z - 1)
            _, rows, cols = tile_sparse_jac(1., rows, cols,
                1, num_points_z - 1, num_nodes)

            self.declare_partials('structural_volume', A_name, rows=rows, cols=cols)
            self.declare_partials('structural_volume', L_name, rows=rows, cols=cols)

        self.set_check_partial_options('*', method='cs')

    def compute(self, inputs, outputs):
        lifting_surfaces = self.metadata['lifting_surfaces']

        outputs['structural_volume'] = 0.

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            A_name = '{}_element_{}'.format(lifting_surface_name, 'A')
            L_name = '{}_element_{}'.format(lifting_surface_name, 'L')

            outputs['structural_volume'] += np.sum(inputs[A_name] * inputs[L_name], axis=1)

    def compute_partials(self, inputs, partials):
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            A_name = '{}_element_{}'.format(lifting_surface_name, 'A')
            L_name = '{}_element_{}'.format(lifting_surface_name, 'L')

            partials['structural_volume', A_name] = inputs[L_name].flatten()
            partials['structural_volume', L_name] = inputs[A_name].flatten()
