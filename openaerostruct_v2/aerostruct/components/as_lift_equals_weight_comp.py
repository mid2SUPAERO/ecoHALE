from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent

from openaerostruct_v2.utils.misc_utils import get_array_indices, get_airfoils, tile_sparse_jac
from openaerostruct_v2.utils.vector_algebra import compute_cross, compute_cross_deriv1, compute_cross_deriv2


g = 9.81

class ASLiftEqualsWeightComp(ExplicitComponent):

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

        self.add_input('fuelburn', shape=num_nodes)
        self.add_input('C_L', shape=num_nodes)

        self.add_input('rho_kg_m3', shape=num_nodes)
        self.add_input('v_m_s', shape=num_nodes)
        self.add_input('wing_area_m2', shape=num_nodes)
        self.add_input('structural_weight', shape=num_nodes)

        self.add_output('L_equals_W', shape=num_nodes)

        arange = np.arange(num_nodes)
        self.declare_partials('*', '*', rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        W0 = self.W0
        a = self.a
        R = self.R
        M = self.M

        rho_kg_m3 = inputs['rho_kg_m3']
        v_m_s = inputs['v_m_s']
        wing_area_m2 = inputs['wing_area_m2']
        CL = inputs['C_L']
        structural_weight = inputs['structural_weight']

        lift = (0.5 * rho_kg_m3 * v_m_s ** 2 * wing_area_m2) * CL
        tot_weight = structural_weight + inputs['fuelburn'] * g + W0

        outputs['L_equals_W'] = (lift - tot_weight) / tot_weight

    def compute_partials(self, inputs, partials):
        W0 = self.W0
        a = self.a
        R = self.R
        M = self.M

        rho_kg_m3 = inputs['rho_kg_m3']
        v_m_s = inputs['v_m_s']
        wing_area_m2 = inputs['wing_area_m2']
        CL = inputs['C_L']
        structural_weight = inputs['structural_weight']

        lift = (0.5 * rho_kg_m3 * v_m_s ** 2 * wing_area_m2) * CL
        tot_weight = structural_weight + inputs['fuelburn'] * g + W0

        partials['L_equals_W', 'C_L'] = .5 * rho_kg_m3 * v_m_s**2 * wing_area_m2 / tot_weight
        partials['L_equals_W', 'wing_area_m2'] = .5 * rho_kg_m3 * v_m_s**2 * CL / tot_weight
        partials['L_equals_W', 'rho_kg_m3'] = .5 * CL * v_m_s**2 * wing_area_m2 / tot_weight
        partials['L_equals_W', 'v_m_s'] = rho_kg_m3 * v_m_s * wing_area_m2 * CL / tot_weight
        partials['L_equals_W', 'fuelburn'] = -g * lift / tot_weight**2
        partials['L_equals_W', 'structural_weight'] = -lift / tot_weight**2
