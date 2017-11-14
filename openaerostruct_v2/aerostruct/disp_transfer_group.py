from __future__ import print_function
import numpy as np

from openmdao.api import Group

from openaerostruct_v2.aerostruct.components.as_disp_transform_comp import ASDispTransformComp
from openaerostruct_v2.aerostruct.components.as_disp_transfer_comp import ASDispTransferComp


class DispTransferGroup(Group):

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)
        self.metadata.declare('lifting_surfaces', types=list)

    def setup(self):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']

        self.add_subsystem('as_disp_transform_comp',
            ASDispTransformComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces),
            promotes=['*'],
        )
        self.add_subsystem('as_disp_transfer_comp',
            ASDispTransferComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces, vortex_mesh=False),
            promotes=['*'],
        )
        self.add_subsystem('as_disp_transfer_vortex_comp',
            ASDispTransferComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces, vortex_mesh=True),
            promotes=['*'],
        )
