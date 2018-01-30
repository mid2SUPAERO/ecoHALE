from __future__ import print_function
import numpy as np

from openmdao.api import Group

from openaerostruct.structures.components.fea_forces_comp import FEAForcesComp
from openaerostruct.structures.components.fea_states_comp import FEAStatesComp
from openaerostruct.structures.components.fea_disp_comp import FEADispComp


class FEAStatesGroup(Group):

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)
        self.metadata.declare('lifting_surfaces', types=list)
        self.metadata.declare('fea_scaler', types=float, default=1e6)

    def setup(self):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']
        fea_scaler = self.metadata['fea_scaler']

        comp = FEAForcesComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces, fea_scaler=fea_scaler)
        self.add_subsystem('fea_forces_comp', comp, promotes=['*'])

        comp = FEAStatesComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces)
        self.add_subsystem('fea_states_comp', comp, promotes=['*'])

        comp = FEADispComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces)
        self.add_subsystem('fea_disp_comp', comp, promotes=['*'])
