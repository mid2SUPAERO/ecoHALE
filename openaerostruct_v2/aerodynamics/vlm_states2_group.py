from __future__ import print_function
import numpy as np

from openmdao.api import Group

from openaerostruct_v2.aerodynamics.components.velocities.vlm_inflow_velocities_comp import VLMInflowVelocitiesComp
from openaerostruct_v2.aerodynamics.components.velocities.vlm_eval_vectors_comp import VLMEvalVectorsComp
from openaerostruct_v2.aerodynamics.components.velocities.vlm_eval_vel_mtx_comp import VLMEvalVelMtxComp
from openaerostruct_v2.aerodynamics.components.velocities.vlm_eval_velocities_comp import VLMEvalVelocitiesComp

from openaerostruct_v2.aerodynamics.components.circulations.vlm_mtx_rhs_comp import VLMMtxRHSComp
from openaerostruct_v2.aerodynamics.components.circulations.vlm_circulations_comp import VLMCirculationsComp
from openaerostruct_v2.aerodynamics.components.circulations.vlm_horseshoe_circulations_comp import VLMHorseshoeCirculationsComp

from openaerostruct_v2.aerodynamics.components.forces.vlm_panel_forces_comp import VLMPanelForcesComp
from openaerostruct_v2.aerodynamics.components.forces.vlm_panel_forces_surf_comp import VLMPanelForcesSurfComp


class VLMStates2Group(Group):

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

        comp = VLMInflowVelocitiesComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces, suffix='t')
        self.add_subsystem('vlm_inflow_velocities_t_comp', comp, promotes=['*'])

        comp = VLMInflowVelocitiesComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces, suffix='f')
        self.add_subsystem('vlm_inflow_velocities_f_comp', comp, promotes=['*'])

        comp = VLMEvalVectorsComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces,
            eval_name='coll_pts', num_eval_points=num_collocation_points)
        self.add_subsystem('vlm_collocation_eval_vectors_comp', comp, promotes=['*'])

        comp = VLMEvalVelMtxComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces,
            eval_name='coll_pts', num_eval_points=num_collocation_points)
        self.add_subsystem('vlm_collocation_eval_vel_mtx_comp', comp, promotes=['*'])

        comp = VLMMtxRHSComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces)
        self.add_subsystem('vlm_mtx_rhs_comp', comp, promotes=['*'])

        comp = VLMCirculationsComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces)
        self.add_subsystem('vlm_circulations_comp', comp, promotes=['*'])

        comp = VLMHorseshoeCirculationsComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces)
        self.add_subsystem('vlm_horseshoe_circulations_comp', comp, promotes=['*'])

        comp = VLMEvalVectorsComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces,
            eval_name='force_pts', num_eval_points=num_force_points)
        self.add_subsystem('vlm_force_eval_vectors_comp', comp, promotes=['*'])

        comp = VLMEvalVelMtxComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces,
            eval_name='force_pts', num_eval_points=num_force_points)
        self.add_subsystem('vlm_force_eval_vel_mtx_comp', comp, promotes=['*'])

        comp = VLMEvalVelocitiesComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces,
            eval_name='force_pts', num_eval_points=num_force_points)
        self.add_subsystem('vlm_force_eval_velocities_comp', comp, promotes=['*'])

        comp = VLMPanelForcesComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces)
        self.add_subsystem('vlm_panel_forces_comp', comp, promotes=['*'])

        comp = VLMPanelForcesSurfComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces)
        self.add_subsystem('vlm_panel_forces_surf_comp', comp, promotes=['*'])
