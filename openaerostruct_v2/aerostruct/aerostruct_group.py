from __future__ import print_function
import numpy as np

from openmdao.api import Group, NonlinearBlockGS, LinearBlockGS

from openaerostruct_v2.aerodynamics.vlm_states1_group import VLMStates1Group
from openaerostruct_v2.aerodynamics.vlm_states2_group import VLMStates2Group

from openaerostruct_v2.structures.fea_states_group import FEAStatesGroup

from openaerostruct_v2.aerostruct.load_transfer_group import LoadTransferGroup
from openaerostruct_v2.aerostruct.disp_transfer_group import DispTransferGroup


class AerostructGroup(Group):

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)
        self.metadata.declare('wing_data', types=dict)

    def setup(self):
        num_nodes = self.metadata['num_nodes']
        wing_data = self.metadata['wing_data']

        self.add_subsystem('vlm_states1_group',
            VLMStates1Group(num_nodes=num_nodes, wing_data=wing_data),
            promotes=['*'],
        )
        self.add_subsystem('vlm_states2_group',
            VLMStates2Group(num_nodes=num_nodes, wing_data=wing_data),
            promotes=['*'],
        )
        self.add_subsystem('load_transfer_group',
            LoadTransferGroup(num_nodes=num_nodes, wing_data=wing_data),
            promotes=['*'],
        )
        self.add_subsystem('fea_states_group',
            FEAStatesGroup(num_nodes=num_nodes, wing_data=wing_data),
            promotes=['*'],
        )
        self.add_subsystem('disp_transfer_group',
            DispTransferGroup(num_nodes=num_nodes, wing_data=wing_data),
            promotes=['*'],
        )

        self.nonlinear_solver = NonlinearBlockGS(iprint=2, maxiter=100)
        self.linear_solver = LinearBlockGS(iprint=2, maxiter=100)
