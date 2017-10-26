from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent


class VLMTotalCoeffsComp(ExplicitComponent):
    """
    Total lift and drag coefficients.
    """

    def initialize(self):
        self.metadata.declare('lifting_surfaces', type_=list)

    def setup(self):
        lifting_surfaces = self.metadata['lifting_surfaces']

        system_size = 0

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            system_size += (num_points_x - 1) * (num_points_z - 1)

        self.system_size = system_size

        self.add_input('rho_kg_m3')
        self.add_input('v_m_s')
        self.add_input('wing_area_m2')
        self.add_input('lift')
        self.add_input('drag')
        self.add_output('C_L')
        self.add_output('C_D')

        self.declare_partials('*', '*')
        self.declare_partials('C_L', 'drag', dependent=False)
        self.declare_partials('C_D', 'lift', dependent=False)

    def compute(self, inputs, outputs):
        lift = inputs['lift']
        drag = inputs['drag']
        rho_kg_m3 = inputs['rho_kg_m3']
        v_m_s = inputs['v_m_s']
        wing_area_m2 = inputs['wing_area_m2']

        outputs['C_L'] = lift / (0.5 * rho_kg_m3 * v_m_s ** 2 * wing_area_m2)
        outputs['C_D'] = drag / (0.5 * rho_kg_m3 * v_m_s ** 2 * wing_area_m2)

        print(outputs['C_L'], outputs['C_D'])

    def compute_partials(self, inputs, partials):
        lift = inputs['lift']
        drag = inputs['drag']
        rho_kg_m3 = inputs['rho_kg_m3']
        v_m_s = inputs['v_m_s']
        wing_area_m2 = inputs['wing_area_m2']

        partials['C_L', 'lift'] = 1. / (0.5 * rho_kg_m3 * v_m_s ** 2 * wing_area_m2)
        partials['C_L', 'rho_kg_m3'] = -lift / (0.5 * rho_kg_m3 ** 2 * v_m_s ** 2 * wing_area_m2)
        partials['C_L', 'v_m_s'] = -2 * lift / (0.5 * rho_kg_m3 * v_m_s ** 3 * wing_area_m2)
        partials['C_L', 'wing_area_m2'] = -lift / (0.5 * rho_kg_m3 * v_m_s ** 2 * wing_area_m2 ** 2)

        partials['C_D', 'drag'] = 1. / (0.5 * rho_kg_m3 * v_m_s ** 2 * wing_area_m2)
        partials['C_D', 'rho_kg_m3'] = -drag / (0.5 * rho_kg_m3 ** 2 * v_m_s ** 2 * wing_area_m2)
        partials['C_D', 'v_m_s'] = -2 * drag / (0.5 * rho_kg_m3 * v_m_s ** 3 * wing_area_m2)
        partials['C_D', 'wing_area_m2'] = -drag / (0.5 * rho_kg_m3 * v_m_s ** 2 * wing_area_m2 ** 2)
