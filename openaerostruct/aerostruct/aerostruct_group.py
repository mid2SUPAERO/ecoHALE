from __future__ import print_function
import numpy as np

from openmdao.api import Group, NonlinearBlockGS, LinearBlockGS, ScipyKrylov, DenseJacobian, DirectSolver, NewtonSolver

from openaerostruct.aerodynamics.vlm_states_group import VLMStatesGroup

from openaerostruct.structures.fea_states_group import FEAStatesGroup

from openaerostruct.aerostruct.load_transfer_group import LoadTransferGroup
from openaerostruct.aerostruct.disp_transfer_group import DispTransferGroup


class AerostructGroup(Group):

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)
        self.metadata.declare('lifting_surfaces', types=list)
        self.metadata.declare('vlm_scaler', types=float, default=1e2)
        self.metadata.declare('fea_scaler', types=float, default=1e6)

    def setup(self):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']
        vlm_scaler = self.metadata['vlm_scaler']
        fea_scaler = self.metadata['fea_scaler']

        self.add_subsystem('vlm_states_group',
            VLMStatesGroup(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces, vlm_scaler=vlm_scaler),
            promotes=['*'],
        )
        self.add_subsystem('load_transfer_group',
            LoadTransferGroup(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces),
            promotes=['*'],
        )
        self.add_subsystem('fea_states_group',
            FEAStatesGroup(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces, fea_scaler=fea_scaler),
            promotes=['*'],
        )
        self.add_subsystem('disp_transfer_group',
            DispTransferGroup(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces),
            promotes=['*'],
        )

        self.nonlinear_solver = NonlinearBlockGS(iprint=2, maxiter=20, atol=1e-10, rtol=1e-10, use_aitken=True)
        self.linear_solver = ScipyKrylov(iprint=-1, maxiter=50, atol=1e-12, rtol=1e-12)
        self.linear_solver.precon = LinearBlockGS(iprint=-1, maxiter=1)
