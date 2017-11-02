from __future__ import print_function
import numpy as np

from openmdao.api import Group

from openaerostruct_v2.structures.components.fea_forces_comp import FEAForcesComp
from openaerostruct_v2.structures.components.fea_states_comp import FEAStatesComp
from openaerostruct_v2.structures.components.fea_disp_comp import FEADispComp


class FEAStatesGroup(Group):

    def initialize(self):
        self.metadata.declare('lifting_surfaces', type_=list)

    def setup(self):
        lifting_surfaces = self.metadata['lifting_surfaces']

        comp = FEAForcesComp(lifting_surfaces=lifting_surfaces)
        self.add_subsystem('fea_forces_comp', comp, promotes=['*'])

        comp = FEAStatesComp(lifting_surfaces=lifting_surfaces)
        self.add_subsystem('fea_states_comp', comp, promotes=['*'])

        comp = FEADispComp(lifting_surfaces=lifting_surfaces)
        self.add_subsystem('fea_disp_comp', comp, promotes=['*'])
