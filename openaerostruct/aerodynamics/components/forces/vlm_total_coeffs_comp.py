from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent

from openaerostruct.utils.misc_utils import get_airfoils, tile_sparse_jac


class VLMTotalCoeffsComp(ExplicitComponent):
    """
    Total lift and drag coefficients.
    """

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)
        self.metadata.declare('lifting_surfaces', types=list)

    def setup(self):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']

        self.add_input('rho_kg_m3', shape=num_nodes)
        self.add_input('v_m_s', shape=num_nodes)
        self.add_input('wing_area_m2', shape=num_nodes)
        self.add_input('lift', shape=num_nodes)
        self.add_input('drag', shape=num_nodes)
        self.add_output('C_L_ind', shape=num_nodes)
        self.add_output('C_D_ind', shape=num_nodes)

        arange = np.arange(num_nodes)

        self.declare_partials('C_L_ind', 'rho_kg_m3', rows=arange, cols=arange)
        self.declare_partials('C_L_ind', 'v_m_s', rows=arange, cols=arange)
        self.declare_partials('C_L_ind', 'wing_area_m2', rows=arange, cols=arange)
        self.declare_partials('C_L_ind', 'lift', rows=arange, cols=arange)

        self.declare_partials('C_D_ind', 'rho_kg_m3', rows=arange, cols=arange)
        self.declare_partials('C_D_ind', 'v_m_s', rows=arange, cols=arange)
        self.declare_partials('C_D_ind', 'wing_area_m2', rows=arange, cols=arange)
        self.declare_partials('C_D_ind', 'drag', rows=arange, cols=arange)

        self.set_check_partial_options('*', method='cs')

    def compute(self, inputs, outputs):
        lift = inputs['lift']
        drag = inputs['drag']
        rho_kg_m3 = inputs['rho_kg_m3']
        v_m_s = inputs['v_m_s']
        wing_area_m2 = inputs['wing_area_m2']

        outputs['C_L_ind'] = lift / (0.5 * rho_kg_m3 * v_m_s ** 2 * wing_area_m2)
        outputs['C_D_ind'] = drag / (0.5 * rho_kg_m3 * v_m_s ** 2 * wing_area_m2)

    def compute_partials(self, inputs, partials):
        lift = inputs['lift']
        drag = inputs['drag']
        rho_kg_m3 = inputs['rho_kg_m3']
        v_m_s = inputs['v_m_s']
        wing_area_m2 = inputs['wing_area_m2']

        partials['C_L_ind', 'lift'] = 1. / (0.5 * rho_kg_m3 * v_m_s ** 2 * wing_area_m2)
        partials['C_L_ind', 'rho_kg_m3'] = -lift / (0.5 * rho_kg_m3 ** 2 * v_m_s ** 2 * wing_area_m2)
        partials['C_L_ind', 'v_m_s'] = -2 * lift / (0.5 * rho_kg_m3 * v_m_s ** 3 * wing_area_m2)
        partials['C_L_ind', 'wing_area_m2'] = -lift / (0.5 * rho_kg_m3 * v_m_s ** 2 * wing_area_m2 ** 2)

        partials['C_D_ind', 'drag'] = 1. / (0.5 * rho_kg_m3 * v_m_s ** 2 * wing_area_m2)
        partials['C_D_ind', 'rho_kg_m3'] = -drag / (0.5 * rho_kg_m3 ** 2 * v_m_s ** 2 * wing_area_m2)
        partials['C_D_ind', 'v_m_s'] = -2 * drag / (0.5 * rho_kg_m3 * v_m_s ** 3 * wing_area_m2)
        partials['C_D_ind', 'wing_area_m2'] = -drag / (0.5 * rho_kg_m3 * v_m_s ** 2 * wing_area_m2 ** 2)
