from __future__ import print_function
import numpy as np

from openmdao.api import Group

from openaerostruct_v2.aerodynamics.components.forces.vlm_panel_coeffs_comp import VLMPanelCoeffsComp
from openaerostruct_v2.aerodynamics.components.forces.vlm_rotate_panel_forces_comp import VLMRotatePanelForcesComp
from openaerostruct_v2.aerodynamics.components.forces.vlm_total_forces_comp import VLMTotalForcesComp
from openaerostruct_v2.aerodynamics.components.forces.vlm_total_coeffs_comp import VLMTotalCoeffsComp


class VLMPostprocessGroup(Group):

    def initialize(self):
        self.metadata.declare('num_nodes', type_=int)
        self.metadata.declare('lifting_surfaces', type_=list)

    def setup(self):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']

        comp = VLMRotatePanelForcesComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces)
        self.add_subsystem('vlm_rotate_panel_forces_comp', comp, promotes=['*'])

        comp = VLMPanelCoeffsComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces)
        self.add_subsystem('vlm_panel_coeffs_comp', comp, promotes=['*'])

        comp = VLMTotalForcesComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces)
        self.add_subsystem('vlm_total_forces_comp', comp, promotes=['*'])

        comp = VLMTotalCoeffsComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces)
        self.add_subsystem('vlm_total_coeffs_comp', comp, promotes=['*'])
