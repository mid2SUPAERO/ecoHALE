from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent

class VLMModifyCoeffsComp(ExplicitComponent):
    """
    Change lift and drag coefficients.
    """

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)
        self.metadata.declare('lifting_surfaces', types=list)

    def setup(self):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']

        if 'CD0' in lifting_surfaces[0][1].keys():
            self.CD0 = lifting_surfaces[0][1]['CD0']
        else:
            self.CD0 = 0.

        if 'CL0' in lifting_surfaces[0][1].keys():
            self.CL0 = lifting_surfaces[0][1]['CL0']
        else:
            self.CL0 = 0.

        if 'factor2' in lifting_surfaces[0][1].keys():
            self.factor2 = lifting_surfaces[0][1]['factor2']
        else:
            self.factor2 = 0.

        if 'factor4' in lifting_surfaces[0][1].keys():
            self.factor4 = lifting_surfaces[0][1]['factor4']
        else:
            self.factor4 = 0.

        if 'cl_factor' in lifting_surfaces[0][1].keys():
            self.cl_factor = lifting_surfaces[0][1]['cl_factor']
        else:
            self.cl_factor = 1.

        self.add_input('C_L_ind', shape=num_nodes)
        self.add_input('C_D_ind', shape=num_nodes)
        self.add_output('C_L', shape=num_nodes)
        self.add_output('C_D', shape=num_nodes)

        arange = np.arange(num_nodes)

        self.declare_partials('C_L', 'C_L_ind', rows=arange, cols=arange)
        self.declare_partials('C_D', 'C_D_ind', rows=arange, cols=arange)
        self.declare_partials('C_L', 'C_D_ind', rows=arange, cols=arange)
        self.declare_partials('C_D', 'C_L_ind', rows=arange, cols=arange)

        self.set_check_partial_options('*', method='cs')

    def compute(self, inputs, outputs):
        CL = inputs['C_L_ind'] + self.CL0
        CD = inputs['C_D_ind'] + self.CD0

        outputs['C_L'] = CL * self.cl_factor
        outputs['C_D'] = CD + self.factor2 * (CL - .2) ** 2 + self.factor4 * CL ** 4
        print('oas coeffs')
        print(outputs['C_L'])
        print(outputs['C_D'])
        print()

    def compute_partials(self, inputs, partials):
        CL = inputs['C_L_ind'] + self.CL0
        CD = inputs['C_D_ind'] + self.CD0

        partials['C_L', 'C_D_ind'] = 0.
        partials['C_D', 'C_L_ind'] = 4 * self.factor4 * CL ** 3 + 2 * self.factor2 * (CL - .2)
        partials['C_L', 'C_L_ind'] = 1. * self.cl_factor
        partials['C_D', 'C_D_ind'] = 1.
