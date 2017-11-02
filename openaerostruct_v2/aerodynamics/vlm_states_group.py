from __future__ import print_function
import numpy as np

from openmdao.api import Group

from openaerostruct_v2.aerodynamics.components.mesh.vlm_displace_meshes_comp import VLMDisplaceMeshesComp
from openaerostruct_v2.aerodynamics.components.mesh.vlm_normals_comp import VLMNormalsComp
from openaerostruct_v2.aerodynamics.components.mesh.vlm_mesh_points_comp import VLMMeshPointsComp
from openaerostruct_v2.aerodynamics.components.mesh.vlm_mesh_cp_comp import VLMMeshCPComp

from openaerostruct_v2.aerodynamics.components.velocities.vlm_eval_vectors_comp import VLMEvalVectorsComp
from openaerostruct_v2.aerodynamics.components.velocities.vlm_eval_vel_mtx_comp import VLMEvalVelMtxComp
from openaerostruct_v2.aerodynamics.components.velocities.vlm_eval_velocities_comp import VLMEvalVelocitiesComp

from openaerostruct_v2.aerodynamics.components.circulations.vlm_mtx_rhs_comp import VLMMtxRHSComp
from openaerostruct_v2.aerodynamics.components.circulations.vlm_circulations_comp import VLMCirculationsComp
from openaerostruct_v2.aerodynamics.components.circulations.vlm_horseshoe_circulations_comp import VLMHorseshoeCirculationsComp

from openaerostruct_v2.aerodynamics.components.forces.vlm_panel_forces_comp import VLMPanelForcesComp
from openaerostruct_v2.aerodynamics.components.forces.vlm_panel_forces_surf_comp import VLMPanelForcesSurfComp


class VLMStatesGroup(Group):

    def initialize(self):
        self.metadata.declare('lifting_surfaces', type_=list)

    def setup(self):
        lifting_surfaces = self.metadata['lifting_surfaces']

        num_collocation_points = 0

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            num_collocation_points += (num_points_x - 1) * (num_points_z - 1)

        num_force_points = num_collocation_points

        comp = VLMDisplaceMeshesComp(lifting_surfaces=lifting_surfaces)
        self.add_subsystem('vlm_displace_meshes_comp', comp, promotes=['*'])

        comp = VLMNormalsComp(lifting_surfaces=lifting_surfaces)
        self.add_subsystem('vlm_normals_comp', comp, promotes=['*'])

        comp = VLMMeshPointsComp(lifting_surfaces=lifting_surfaces)
        self.add_subsystem('vlm_mesh_points_comp', comp, promotes=['*'])

        comp = VLMMeshCPComp(lifting_surfaces=lifting_surfaces)
        self.add_subsystem('vlm_mesh_cp_comp', comp, promotes=['*'])

        comp = VLMEvalVectorsComp(lifting_surfaces=lifting_surfaces, eval_name='coll_pts',
            num_eval_points=num_collocation_points)
        self.add_subsystem('vlm_collocation_eval_vectors_comp', comp, promotes=['*'])

        comp = VLMEvalVelMtxComp(lifting_surfaces=lifting_surfaces, eval_name='coll_pts',
            num_eval_points=num_collocation_points)
        self.add_subsystem('vlm_collocation_eval_vel_mtx_comp', comp, promotes=['*'])

        comp = VLMMtxRHSComp(lifting_surfaces=lifting_surfaces)
        self.add_subsystem('vlm_mtx_rhs_comp', comp, promotes=['*'])

        comp = VLMCirculationsComp(lifting_surfaces=lifting_surfaces)
        self.add_subsystem('vlm_circulations_comp', comp, promotes=['*'])

        comp = VLMHorseshoeCirculationsComp(lifting_surfaces=lifting_surfaces)
        self.add_subsystem('vlm_horseshoe_circulations_comp', comp, promotes=['*'])

        comp = VLMEvalVectorsComp(lifting_surfaces=lifting_surfaces, eval_name='force_pts',
            num_eval_points=num_force_points)
        self.add_subsystem('vlm_force_eval_vectors_comp', comp, promotes=['*'])

        comp = VLMEvalVelMtxComp(lifting_surfaces=lifting_surfaces, eval_name='force_pts',
            num_eval_points=num_force_points)
        self.add_subsystem('vlm_force_eval_vel_mtx_comp', comp, promotes=['*'])

        comp = VLMEvalVelocitiesComp(lifting_surfaces=lifting_surfaces, eval_name='force_pts',
            num_eval_points=num_force_points)
        self.add_subsystem('vlm_force_eval_velocities_comp', comp, promotes=['*'])

        comp = VLMPanelForcesComp(lifting_surfaces=lifting_surfaces)
        self.add_subsystem('vlm_panel_forces_comp', comp, promotes=['*'])

        comp = VLMPanelForcesSurfComp(lifting_surfaces=lifting_surfaces)
        self.add_subsystem('vlm_panel_forces_surf_comp', comp, promotes=['*'])
