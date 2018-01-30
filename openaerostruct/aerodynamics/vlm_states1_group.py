from __future__ import print_function
import numpy as np

from openmdao.api import Group

from openaerostruct.aerodynamics.components.mesh.vlm_displace_meshes_comp import VLMDisplaceMeshesComp
from openaerostruct.aerodynamics.components.mesh.vlm_normals_comp import VLMNormalsComp
from openaerostruct.aerodynamics.components.mesh.vlm_mesh_points_comp import VLMMeshPointsComp
from openaerostruct.aerodynamics.components.mesh.vlm_mesh_cp_comp import VLMMeshCPComp


class VLMStates1Group(Group):

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)
        self.metadata.declare('lifting_surfaces', types=list)

    def setup(self):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']

        num_collocation_points = 0

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            num_collocation_points += (num_points_x - 1) * (num_points_z - 1)

        num_force_points = num_collocation_points

        comp = VLMDisplaceMeshesComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces)
        self.add_subsystem('vlm_displace_meshes_comp', comp, promotes=['*'])

        comp = VLMNormalsComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces)
        self.add_subsystem('vlm_normals_comp', comp, promotes=['*'])

        comp = VLMMeshPointsComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces)
        self.add_subsystem('vlm_mesh_points_comp', comp, promotes=['*'])

        comp = VLMMeshCPComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces)
        self.add_subsystem('vlm_mesh_cp_comp', comp, promotes=['*'])
