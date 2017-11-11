from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent

from openaerostruct_v2.utils.misc_utils import tile_sparse_jac


class VLMInflowVelocitiesComp(ExplicitComponent):

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

        self.add_input('alpha_rad', shape=num_nodes)
        self.add_input('v_m_s', shape=num_nodes)
        self.add_output('inflow_velocities', shape=(num_nodes, system_size, 3))

        rows = np.arange(3 * system_size)
        cols = np.zeros(3 * system_size)

        _, rows, cols = tile_sparse_jac(1., rows, cols, 3 * system_size, 1, num_nodes)
        self.declare_partials('inflow_velocities', 'alpha_rad', rows=rows, cols=cols)
        self.declare_partials('inflow_velocities', 'v_m_s', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        num_nodes = self.metadata['num_nodes']

        system_size = self.system_size

        alpha_rad = inputs['alpha_rad']
        v_m_s = inputs['v_m_s']

        ones = np.ones(system_size)

        outputs['inflow_velocities'][:, :, 0] = np.outer(v_m_s * np.cos(alpha_rad), ones)
        outputs['inflow_velocities'][:, :, 1] = np.outer(v_m_s * np.sin(alpha_rad), ones)
        outputs['inflow_velocities'][:, :, 2] = 0.

    def compute_partials(self, inputs, partials):
        num_nodes = self.metadata['num_nodes']

        system_size = self.system_size

        alpha_rad = inputs['alpha_rad'][0]
        v_m_s = inputs['v_m_s'][0]

        partials['inflow_velocities', 'v_m_s'] = np.einsum('ij,k->ijk',
            np.ones((num_nodes, system_size)),
            np.array([ np.cos(alpha_rad) , np.sin(alpha_rad) , 0. ]),
        ).flatten()

        partials['inflow_velocities', 'alpha_rad'] = np.outer(
            v_m_s * np.ones((num_nodes, system_size)),
            np.array([ -np.sin(alpha_rad) , np.cos(alpha_rad) , 0. ]),
        ).flatten()
