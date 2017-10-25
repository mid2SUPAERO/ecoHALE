from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent


class VLMInflowVelocitiesComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('lifting_surfaces', type_=list)

    def setup(self):
        lifting_surfaces = self.metadata['lifting_surfaces']

        system_size = 0

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            system_size += (num_points_x - 1) * (num_points_z - 1)

        self.system_size = system_size

        self.add_input('alpha_rad')
        self.add_input('v_m_s')
        self.add_output('inflow_velocities', shape=(system_size, 3))

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        system_size = self.system_size

        alpha_rad = inputs['alpha_rad'][0]
        v_m_s = inputs['v_m_s'][0]

        outputs['inflow_velocities'][:, 0] = v_m_s * np.cos(alpha_rad)
        outputs['inflow_velocities'][:, 1] = v_m_s * np.sin(alpha_rad)
        outputs['inflow_velocities'][:, 2] = 0.

    def compute_partials(self, inputs, partials):
        system_size = self.system_size

        alpha_rad = inputs['alpha_rad'][0]
        v_m_s = inputs['v_m_s'][0]

        partials['inflow_velocities', 'v_m_s'] = np.outer(
            np.ones(system_size),
            np.array([ np.cos(alpha_rad) , np.sin(alpha_rad) , 0. ]),
        ).reshape((3 * system_size, 1))

        partials['inflow_velocities', 'alpha_rad'] = np.outer(
            v_m_s * np.ones(system_size),
            np.array([ -np.sin(alpha_rad) , np.cos(alpha_rad) , 0. ]),
        ).reshape((3 * system_size, 1))
