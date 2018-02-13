from __future__ import print_function
import numpy as np

from openmdao.api import Group

from openaerostruct.aerostruct.components.as_fuelburn_comp import ASFuelburnComp
from openaerostruct.aerostruct.components.as_lift_equals_weight_comp import ASLiftEqualsWeightComp


class AerostructPostprocessGroup(Group):

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)
        self.metadata.declare('lifting_surfaces', types=list)

    def setup(self):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']

        comp = ASFuelburnComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces)
        self.add_subsystem('as_fuelburn_comp', comp, promotes=['*'])

        comp = ASLiftEqualsWeightComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces)
        self.add_subsystem('as_lift_equals_weight_comp', comp, promotes=['*'])
