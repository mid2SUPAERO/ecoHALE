from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent

from openaerostruct_v2.utils.misc_utils import get_array_indices, get_airfoils, tile_sparse_jac
from openaerostruct_v2.utils.vector_algebra import compute_cross, compute_cross_deriv1, compute_cross_deriv2


g = 9.81

class ASFuelburnComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)
        self.metadata.declare('lifting_surfaces', types=list)

    def setup(self):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']

        if 'W0' in lifting_surfaces[0][1].keys():
            self.W0 = lifting_surfaces[0][1]['W0']
        else:
            self.W0 = 0.
        if 'a' in lifting_surfaces[0][1].keys():
            self.a = lifting_surfaces[0][1]['a']
        else:
            self.a = 0.
        if 'R' in lifting_surfaces[0][1].keys():
            self.R = lifting_surfaces[0][1]['R']
        else:
            self.R = 0.
        if 'M' in lifting_surfaces[0][1].keys():
            self.M = lifting_surfaces[0][1]['M']
        else:
            self.M = 0.
        if 'CT' in lifting_surfaces[0][1].keys():
            self.CT = lifting_surfaces[0][1]['CT']
        else:
            self.CT = 0.

        self.add_input('C_L', shape=num_nodes)
        self.add_input('C_D', shape=num_nodes)
        self.add_input('structural_weight', shape=num_nodes)

        self.add_output('fuelburn', shape=num_nodes)

        arange = np.arange(num_nodes)
        self.declare_partials('*', '*', rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        W0 = self.W0
        a = self.a
        R = self.R
        M = self.M
        CT = self.CT

        # Loop through the surfaces and add up the structural weights
        # to get the total structural weight.
        Ws = inputs['structural_weight']

        CL = inputs['C_L']
        CD = inputs['C_D']

        fuelburn = (W0 + Ws) * (np.exp(R * CT / a / M * CD / CL) - 1)

        # Convert fuelburn from N to kg
        outputs['fuelburn'] = fuelburn / g
        print('fb:', outputs['fuelburn'])

    def compute_partials(self, inputs, partials):
        W0 = self.W0
        a = self.a
        R = self.R
        M = self.M
        CT = self.CT

        Ws = inputs['structural_weight']

        CL = inputs['C_L']
        CD = inputs['C_D']

        dfb_dCL = -(W0 + Ws) * np.exp(R * CT / a / M * CD / CL) \
            * R * CT / a / M * CD / CL ** 2
        dfb_dCD = (W0 + Ws) * np.exp(R * CT / a / M * CD / CL) \
            * R * CT / a / M / CL
        dfb_dCT = (W0 + Ws) * np.exp(R * CT / a / M * CD / CL) \
            * R / a / M / CL * CD
        dfb_dR = (W0 + Ws) * np.exp(R * CT / a / M * CD / CL) \
            / a / M / CL * CD * CT
        dfb_da = -(W0 + Ws) * np.exp(R * CT / a / M * CD / CL) \
            * R * CT / a**2 / M * CD / CL
        dfb_dM = -(W0 + Ws) * np.exp(R * CT / a / M * CD / CL) \
            * R * CT / a / M**2 * CD / CL

        dfb_dW = np.exp(R * CT / a / M * CD / CL) - 1

        partials['fuelburn', 'C_L'] = dfb_dCL / g
        partials['fuelburn', 'C_D'] = dfb_dCD / g
        # partials['fuelburn', 'CT'] = dfb_dCT / g
        # partials['fuelburn', 'a'] = dfb_da / g
        # partials['fuelburn', 'R'] = dfb_dR / g
        # partials['fuelburn', 'M'] = dfb_dM / g
        # partials['fuelburn', 'W0'] = dfb_dW

        partials['fuelburn', 'structural_weight'] = dfb_dW / g
