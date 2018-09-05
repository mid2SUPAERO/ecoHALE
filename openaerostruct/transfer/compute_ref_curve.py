from __future__ import division, print_function
import numpy as np

from openmdao.api import ExplicitComponent

try:
    from openaerostruct.fortran import OAS_API
    fortran_flag = True
    data_type = float
except:
    fortran_flag = False
    data_type = complex

np.random.seed(314)

class ComputeRefCurve(ExplicitComponent):

    def initialize(self):
        self.options.declare('surface', types=dict)

    def setup(self):
        self.surface = surface = self.options['surface']

        self.ny = surface['num_y']
        self.nx = surface['num_x']

        if surface['fem_model_type'] == 'tube':
            self.fem_origin = surface['fem_origin']
        else:
            y_upper = surface['data_y_upper']
            x_upper = surface['data_x_upper']
            y_lower = surface['data_y_lower']

            self.fem_origin = (x_upper[0]  * (y_upper[0]  - y_lower[0]) +
                               x_upper[-1] * (y_upper[-1] - y_lower[-1])) / \
                             ((y_upper[0]  -  y_lower[0]) + (y_upper[-1] - y_lower[-1]))

        self.add_input('mesh', val=np.zeros((self.nx, self.ny, 3)), units='m')
        self.add_output('ref_curve', val=np.zeros((self.ny, 3)), units='m')

        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        mesh = inputs['mesh']

        # Get the location of the spar within the wing and save as w
        w = self.fem_origin

        # Get the location of the spar
        ref_curve = (1-w) * mesh[0, :, :] + w * mesh[-1, :, :]

        outputs['ref_curve'] = ref_curve
