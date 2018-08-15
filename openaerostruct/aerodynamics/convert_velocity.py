from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent


class ConvertVelocity(ExplicitComponent):

    def initialize(self):
        self.options.declare('surfaces', types=list)

    def setup(self):
        surfaces = self.options['surfaces']

        system_size = 0

        for surface in surfaces:
            nx = surface['num_x']
            ny = surface['num_y']
            name = surface['name']

            system_size += (nx - 1) * (ny - 1)

        self.system_size = system_size

        self.add_input('alpha', val=0., units='deg')
        self.add_input('v', val=1., units='m/s')

        self.add_output('inflow_velocities', shape=(system_size, 3), units='m/s')

        self.declare_partials('inflow_velocities', 'alpha')
        self.declare_partials('inflow_velocities', 'v')

    def compute(self, inputs, outputs):
        alpha = inputs['alpha'][0] * np.pi / 180.
        cosa = np.cos(alpha)
        sina = np.sin(alpha)
        v_inf = inputs['v'][0] * np.array([cosa, 0., sina])
        outputs['inflow_velocities'][:, :] = v_inf
    def compute_partials(self, inputs, J):
        alpha = inputs['alpha'][0] *np.pi / 180.
        J['inflow_velocities','alpha'] = 0
        J['inflow_velocities','v'] = 0
        cosa = np.cos(alpha)
        sina = np.sin(alpha)

        Jv_v =  np.tile(np.array([cosa, 0., sina]), self.system_size)
        Jv_alpha = np.tile(inputs['v'][0] * np.array([-sina, 0., cosa]) * np.pi/180.,self.system_size)
        J['inflow_velocities','v'] = Jv_v
        J['inflow_velocities','alpha'] = Jv_alpha
    