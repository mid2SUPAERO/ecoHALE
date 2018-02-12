from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent

from openaerostruct.utils.misc_utils import get_array_indices, get_airfoils, tile_sparse_jac
from openaerostruct.utils.vector_algebra import compute_cross, compute_cross_deriv1, compute_cross_deriv2


g = 9.81

class ASFuelburnComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)
        self.metadata.declare('lifting_surfaces', types=list)

    def setup(self):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']

        self.add_input('W0', shape=num_nodes)
        self.add_input('a_m_s', shape=num_nodes)
        self.add_input('design_range_m', shape=num_nodes)
        self.add_input('Mach', shape=num_nodes)
        self.add_input('SFC', shape=num_nodes)

        self.add_input('C_L', shape=num_nodes)
        self.add_input('C_D', shape=num_nodes)
        self.add_input('structural_weight', shape=num_nodes)

        self.add_output('fuelburn', shape=num_nodes)

        arange = np.arange(num_nodes)
        self.declare_partials('*', '*', rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        W0 = inputs['W0']
        a_m_s = inputs['a_m_s']
        design_range_m = inputs['design_range_m']
        Mach = inputs['Mach']
        SFC = inputs['SFC']

        Ws = inputs['structural_weight']

        CL = inputs['C_L']
        CD = inputs['C_D']

        fuelburn = (W0 + Ws) * (np.exp(design_range_m * SFC / a_m_s / Mach * CD / CL) - 1)

        # Convert fuelburn from N to kg
        outputs['fuelburn'] = fuelburn / g

    def compute_partials(self, inputs, partials):
        W0 = inputs['W0']
        a_m_s = inputs['a_m_s']
        design_range_m = inputs['design_range_m']
        Mach = inputs['Mach']
        SFC = inputs['SFC']

        Ws = inputs['structural_weight']

        CL = inputs['C_L']
        CD = inputs['C_D']

        dfb_dCL = -(W0 + Ws) * np.exp(design_range_m * SFC / a_m_s / Mach * CD / CL) \
            * design_range_m * SFC / a_m_s / Mach * CD / CL ** 2
        dfb_dCD = (W0 + Ws) * np.exp(design_range_m * SFC / a_m_s / Mach * CD / CL) \
            * design_range_m * SFC / a_m_s / Mach / CL
        dfb_dSFC = (W0 + Ws) * np.exp(design_range_m * SFC / a_m_s / Mach * CD / CL) \
            * design_range_m / a_m_s / Mach / CL * CD
        dfb_dR = (W0 + Ws) * np.exp(design_range_m * SFC / a_m_s / Mach * CD / CL) \
            / a_m_s / Mach / CL * CD * SFC
        dfb_da = -(W0 + Ws) * np.exp(design_range_m * SFC / a_m_s / Mach * CD / CL) \
            * design_range_m * SFC / a_m_s**2 / Mach * CD / CL
        dfb_dM = -(W0 + Ws) * np.exp(design_range_m * SFC / a_m_s / Mach * CD / CL) \
            * design_range_m * SFC / a_m_s / Mach**2 * CD / CL

        dfb_dW = np.exp(design_range_m * SFC / a_m_s / Mach * CD / CL) - 1

        partials['fuelburn', 'C_L'] = dfb_dCL / g
        partials['fuelburn', 'C_D'] = dfb_dCD / g
        partials['fuelburn', 'SFC'] = dfb_dSFC / g
        partials['fuelburn', 'a_m_s'] = dfb_da / g
        partials['fuelburn', 'design_range_m'] = dfb_dR / g
        partials['fuelburn', 'Mach'] = dfb_dM / g
        partials['fuelburn', 'W0'] = dfb_dW / g

        partials['fuelburn', 'structural_weight'] = dfb_dW / g
