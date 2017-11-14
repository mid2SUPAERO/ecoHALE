from __future__ import print_function
import numpy as np

from openmdao.api import Group

from openaerostruct_v2.aerodynamics.components.mesh.vlm_ref_axis_comp import VLMRefAxisComp
from openaerostruct_v2.aerodynamics.components.mesh.vlm_mesh_comp import VLMMeshComp


class VLMPreprocessGroup(Group):

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)
        self.metadata.declare('lifting_surfaces', types=list)

    def setup(self):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']

        comp = VLMRefAxisComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces)
        self.add_subsystem('vlm_ref_axis_comp', comp, promotes=['*'])

        comp = VLMMeshComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces, vortex_mesh=False)
        self.add_subsystem('vlm_mesh_comp', comp, promotes=['*'])

        comp = VLMMeshComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces, vortex_mesh=True)
        self.add_subsystem('vlm_vortex_mesh_comp', comp, promotes=['*'])
