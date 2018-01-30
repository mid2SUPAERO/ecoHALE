from __future__ import print_function
import numpy as np

from openmdao.api import Group

from openaerostruct_v2.aerodynamics.components.forces.vlm_panel_coeffs_comp import VLMPanelCoeffsComp
from openaerostruct_v2.aerodynamics.components.forces.vlm_panel_coeffs_capped_comp import VLMPanelCoeffsCappedComp
from openaerostruct_v2.aerodynamics.components.forces.vlm_panel_coeffs_factor_comp import VLMPanelCoeffsFactorComp
from openaerostruct_v2.aerodynamics.components.forces.vlm_rotate_panel_forces_comp import VLMRotatePanelForcesComp
from openaerostruct_v2.aerodynamics.components.forces.vlm_total_forces_comp import VLMTotalForcesComp
from openaerostruct_v2.aerodynamics.components.forces.vlm_total_coeffs_comp import VLMTotalCoeffsComp
from openaerostruct_v2.common.product_comp import ProductComp


class VLMPostprocessGroup(Group):

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)
        self.metadata.declare('lifting_surfaces', types=list)

    def setup(self):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']

        system_size = 0

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            system_size += (num_points_x - 1) * (num_points_z - 1)

        # panel_forces_rotated
        comp = VLMRotatePanelForcesComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces)
        self.add_subsystem('vlm_rotate_panel_forces_comp', comp, promotes=['*'])

        # {}_sec_C_L, {}_sec_C_D
        comp = VLMPanelCoeffsComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces)
        self.add_subsystem('vlm_panel_coeffs_comp', comp, promotes=['*'])

        # {}_sec_C_L_capped
        comp = VLMPanelCoeffsCappedComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces)
        self.add_subsystem('vlm_panel_coeffs_capped_comp', comp, promotes=['*'])

        # sec_C_L_factor
        comp = VLMPanelCoeffsFactorComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces)
        self.add_subsystem('vlm_panel_coeffs_factor_comp', comp, promotes=['*'])

        # panel_forces_rotated_capped
        comp = ProductComp(shape=(num_nodes, system_size, 3),
            in_name1='panel_forces_rotated', in_name2='sec_C_L_factor', out_name='panel_forces_rotated_capped')
        self.add_subsystem('vlm_panel_forces_rotated_capped_comp', comp, promotes=['*'])

        # lift, drag (depends on panel_forces_rotated or panel_forces_rotated_capped)
        comp = VLMTotalForcesComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces)
        self.add_subsystem('vlm_total_forces_comp', comp, promotes=['*'])

        # C_L, C_D
        comp = VLMTotalCoeffsComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces)
        self.add_subsystem('vlm_total_coeffs_comp', comp, promotes=['*'])

        comp = VLMModifyCoeffsComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces)
        self.add_subsystem('vlm_modify_coeffs_comp', comp, promotes=['*'])
