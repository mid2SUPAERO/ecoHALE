from __future__ import print_function
import numpy as np

from openmdao.api import Group

from openaerostruct_v2.structures.components.fea_forces_comp import FEAForcesComp
from openaerostruct_v2.structures.components.fea_states_comp import FEAStatesComp
from openaerostruct_v2.structures.components.fea_disp_comp import FEADispComp


class FEAStatesGroup(Group):

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)
        self.metadata.declare('wing_data', types=dict)

    def setup(self):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['wing_data']['lifting_surfaces']

        comp = FEAForcesComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces)
        self.add_subsystem('fea_forces_comp', comp, promotes=['*'])

        comp = FEAStatesComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces)
        self.add_subsystem('fea_states_comp', comp, promotes=['*'])

        comp = FEADispComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces)
        self.add_subsystem('fea_disp_comp', comp, promotes=['*'])
