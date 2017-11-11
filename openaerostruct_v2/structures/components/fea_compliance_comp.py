from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent

from openaerostruct_v2.utils.misc_utils import get_array_indices, tile_sparse_jac


class FEAComplianceComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)
        self.metadata.declare('lifting_surfaces', types=list)

    def setup(self):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']

        self.add_output('compliance', shape=num_nodes)

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            size = 6 * num_points_z + 6

            forces_name = '{}_forces'.format(lifting_surface_name)
            states_name = '{}_states'.format(lifting_surface_name)

            self.add_input(forces_name, shape=(num_nodes, size))
            self.add_input(states_name, shape=(num_nodes, size))

            rows = np.zeros(size, int)
            cols = get_array_indices(size)
            _, rows, cols = tile_sparse_jac(1., rows, cols,
                1, size, num_nodes)

            self.declare_partials('compliance', forces_name, rows=rows, cols=cols)
            self.declare_partials('compliance', states_name, rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        lifting_surfaces = self.metadata['lifting_surfaces']

        outputs['compliance'] = 0.

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            forces_name = '{}_forces'.format(lifting_surface_name)
            states_name = '{}_states'.format(lifting_surface_name)

            outputs['compliance'] += np.sum(inputs[forces_name] * inputs[states_name], axis=1)

    def compute_partials(self, inputs, partials):
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            forces_name = '{}_forces'.format(lifting_surface_name)
            states_name = '{}_states'.format(lifting_surface_name)

            partials['compliance', forces_name] = inputs[states_name].flatten()
            partials['compliance', states_name] = inputs[forces_name].flatten()
