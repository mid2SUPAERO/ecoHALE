from __future__ import print_function
import numpy as np

from openmdao.api import Group

from openaerostruct_v2.aerostruct.components.as_load_transfer_comp import ASLoadTransferComp


class LoadTransferGroup(Group):

    def initialize(self):
        self.metadata.declare('lifting_surfaces', type_=list)

    def setup(self):
        lifting_surfaces = self.metadata['lifting_surfaces']

        self.add_subsystem('as_load_transfer_comp',
            ASLoadTransferComp(lifting_surfaces=lifting_surfaces),
            promotes=['*'],
        )
