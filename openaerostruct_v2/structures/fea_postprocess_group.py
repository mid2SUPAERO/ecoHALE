from __future__ import print_function
import numpy as np

from openmdao.api import Group

from openaerostruct_v2.structures.components.fea_volume_comp import FEAVolumeComp
from openaerostruct_v2.structures.components.fea_compliance_comp import FEAComplianceComp


class FEAPostprocessGroup(Group):

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)
        self.metadata.declare('wing_data', types=dict)

    def setup(self):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['wing_data']['lifting_surfaces']

        comp = FEAVolumeComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces)
        self.add_subsystem('fea_volume_comp', comp, promotes=['*'])

        comp = FEAComplianceComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces)
        self.add_subsystem('fea_compliance_comp', comp, promotes=['*'])
