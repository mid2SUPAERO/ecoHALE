from __future__ import print_function
import numpy as np

from openmdao.api import Group

from openaerostruct_v2.structures.components.fea_volume_comp import FEAVolumeComp
from openaerostruct_v2.structures.components.fea_compliance_comp import FEAComplianceComp


class FEAPostprocessGroup(Group):

    def initialize(self):
        self.metadata.declare('lifting_surfaces', type_=list)

    def setup(self):
        lifting_surfaces = self.metadata['lifting_surfaces']

        comp = FEAVolumeComp(lifting_surfaces=lifting_surfaces)
        self.add_subsystem('fea_volume_comp', comp, promotes=['*'])

        comp = FEAComplianceComp(lifting_surfaces=lifting_surfaces)
        self.add_subsystem('fea_compliance_comp', comp, promotes=['*'])
