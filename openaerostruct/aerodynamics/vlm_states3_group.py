from __future__ import print_function
import numpy as np

from openmdao.api import Group

from openaerostruct.common.linear_comb_comp import LinearCombComp

from openaerostruct.aerodynamics.components.forces.vlm_panel_forces_comp import VLMPanelForcesComp
from openaerostruct.aerodynamics.components.forces.vlm_panel_forces_surf_comp import VLMPanelForcesSurfComp


class VLMStates3Group(Group):

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)
        self.metadata.declare('lifting_surfaces', types=list)

    def setup(self):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']

        size = 0
        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data.num_points_x
            num_points_z = 2 * lifting_surface_data.num_points_z_half - 1
            size += (num_points_x - 1) * (num_points_z - 1)

        in_names = ['force_pts_eval_velocities', 'freestream_vel', 'vlm_ext_velocities']
        out_name = 'force_pts_velocities'
        comp = LinearCombComp(shape=(num_nodes, size, 3), in_names=in_names, out_name=out_name)
        self.add_subsystem('vlm_force_pt_vel_comp', comp, promotes=['*'])

        comp = VLMPanelForcesComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces)
        self.add_subsystem('vlm_panel_forces_comp', comp, promotes=['*'])

        comp = VLMPanelForcesSurfComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces)
        self.add_subsystem('vlm_panel_forces_surf_comp', comp, promotes=['*'])
