from __future__ import print_function
import numpy as np

from openmdao.api import Group

from openaerostruct_v2.aerodynamics.components.mesh.vlm_ref_axis_comp import VLMRefAxisComp
from openaerostruct_v2.aerodynamics.components.mesh.vlm_mesh_comp import VLMMeshComp

from openaerostruct_v2.aerodynamics.components.velocities.vlm_freestream_vel_comp import VLMFreestreamVelComp


class VLMPreprocessGroup(Group):

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)
        self.metadata.declare('lifting_surfaces', types=list)

    def setup(self):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']

        size = 0
        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1
            size += (num_points_x - 1) * (num_points_z - 1)

        comp = VLMRefAxisComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces)
        self.add_subsystem('vlm_ref_axis_comp', comp, promotes=['*'])

        comp = VLMMeshComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces, vortex_mesh=False)
        self.add_subsystem('vlm_mesh_comp', comp, promotes=['*'])

        comp = VLMMeshComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces, vortex_mesh=True)
        self.add_subsystem('vlm_vortex_mesh_comp', comp, promotes=['*'])

        comp = VLMFreestreamVelComp(num_nodes=num_nodes, size=size)
        self.add_subsystem('vlm_freestream_vel_comp', comp, promotes=['*'])
