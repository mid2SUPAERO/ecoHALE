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
        outputs['C_L'] = inputs['C_L_ind']
        outputs['C_D'] = inputs['C_D_ind'] + .01 * inputs['C_L_ind'] ** 2

    def compute_partials(self, inputs, partials):
        partials['C_L', 'C_D_ind'] = 0.
        partials['C_D', 'C_L_ind'] = 2 * .01 * inputs['C_L_ind']
        partials['C_L', 'C_L_ind'] = 1.
        partials['C_D', 'C_D_ind'] = 1.
