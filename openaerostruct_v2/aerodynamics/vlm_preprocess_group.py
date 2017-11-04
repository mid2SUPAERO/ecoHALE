from __future__ import print_function
import numpy as np

from openmdao.api import Group

from openaerostruct_v2.aerodynamics.components.mesh.vlm_ref_axis_comp import VLMRefAxisComp
from openaerostruct_v2.aerodynamics.components.mesh.vlm_mesh_comp import VLMMeshComp

from openaerostruct_v2.aerodynamics.components.velocities.vlm_inflow_velocities_comp import VLMInflowVelocitiesComp


class VLMPreprocessGroup(Group):

    def initialize(self):
        self.metadata.declare('num_nodes', type_=int)
        self.metadata.declare('lifting_surfaces', type_=list)
        self.metadata.declare('section_origin', type_=(int, float))

    def setup(self):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']
        section_origin = self.metadata['section_origin']

        comp = VLMRefAxisComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces)
        self.add_subsystem('vlm_ref_axis_comp', comp, promotes=['*'])

        comp = VLMMeshComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces,
            section_origin=section_origin, vortex_mesh=False)
        self.add_subsystem('vlm_mesh_comp', comp, promotes=['*'])

        comp = VLMMeshComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces,
            section_origin=section_origin, vortex_mesh=True)
        self.add_subsystem('vlm_vortex_mesh_comp', comp, promotes=['*'])

        comp = VLMInflowVelocitiesComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces)
        self.add_subsystem('vlm_inflow_velocities_comp', comp, promotes=['*'])
