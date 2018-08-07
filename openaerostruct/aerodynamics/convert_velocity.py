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

        self.add_input('alpha', val=0.)
        self.add_input('v', val=1.)

        self.add_output('inflow_velocities', shape=(system_size, 3))

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        alpha = inputs['alpha'][0] * np.pi / 180.
        cosa = np.cos(alpha)
        sina = np.sin(alpha)
        v_inf = inputs['v'][0] * np.array([cosa, 0., sina])
        outputs['inflow_velocities'][:, :] = v_inf
