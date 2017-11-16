from __future__ import print_function
import numpy as np

from openmdao.api import Group

from openaerostruct_v2.structures.components.fea_volume_comp import FEAVolumeComp
from openaerostruct_v2.structures.components.fea_weight_comp import FEAWeightComp
from openaerostruct_v2.structures.components.fea_compliance_comp import FEAComplianceComp
from openaerostruct_v2.structures.components.fea_local_disp_comp import FEALocalDispComp
from openaerostruct_v2.structures.components.fea_vonmises_comp import FEAVonmisesComp
from openaerostruct_v2.structures.components.fea_ks_comp import FEAKSComp


class FEAPostprocessGroup(Group):

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)
        self.metadata.declare('lifting_surfaces', types=list)

    def setup(self):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']

        comp = FEAVolumeComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces)
        self.add_subsystem('fea_volume_comp', comp, promotes=['*'])

        comp = FEAWeightComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces)
        self.add_subsystem('fea_weight_comp', comp, promotes=['*'])

        comp = FEAComplianceComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces)
        self.add_subsystem('fea_compliance_comp', comp, promotes=['*'])

        comp = FEALocalDispComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces)
        self.add_subsystem('fea_local_disp_comp', comp, promotes=['*'])

        comp = FEAVonmisesComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces)
        self.add_subsystem('fea_vonmises_comp', comp, promotes=['*'])

        comp = FEAKSComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces)
        self.add_subsystem('fea_ks_comp', comp, promotes=['*'])
