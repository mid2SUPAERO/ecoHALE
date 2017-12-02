from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent

factor2 = 0.119
factor4 = -0.064
cl_factor = 1.05

# factor2 = 0.04
# factor4 = 0.
# cl_factor = 1.15

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
        outputs['C_L'] = inputs['C_L_ind'] * cl_factor
        outputs['C_D'] = inputs['C_D_ind'] + factor2 * (inputs['C_L_ind'] - .2) ** 2 + factor4 * inputs['C_L_ind'] ** 4

    def compute_partials(self, inputs, partials):
        partials['C_L', 'C_D_ind'] = 0.
        partials['C_D', 'C_L_ind'] = 4 * factor4 * inputs['C_L_ind'] ** 3 + 2 * factor2 * (inputs['C_L_ind'] - .2)
        partials['C_L', 'C_L_ind'] = 1. * cl_factor
        partials['C_D', 'C_D_ind'] = 1.
