from __future__ import print_function
import numpy as np

from openmdao.api import Group, NonlinearBlockGS, LinearBlockGS, ScipyKrylov, DenseJacobian, DirectSolver, NewtonSolver

from openaerostruct_v2.aerodynamics.vlm_states1_group import VLMStates1Group
from openaerostruct_v2.aerodynamics.vlm_states2_group import VLMStates2Group
from openaerostruct_v2.aerodynamics.vlm_states3_group import VLMStates3Group

from openaerostruct_v2.structures.fea_states_group import FEAStatesGroup

from openaerostruct_v2.aerostruct.load_transfer_group import LoadTransferGroup
from openaerostruct_v2.aerostruct.disp_transfer_group import DispTransferGroup


class AerostructGroup(Group):

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)
        self.metadata.declare('lifting_surfaces', types=list)
        self.metadata.declare('vlm_scaler', types=float)
        self.metadata.declare('fea_scaler', types=float)

    def setup(self):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']
        vlm_scaler = self.metadata['vlm_scaler']
        fea_scaler = self.metadata['fea_scaler']

        self.add_subsystem('vlm_states1_group',
            VLMStates1Group(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces),
            promotes=['*'],
        )
        self.add_subsystem('vlm_states2_group',
            VLMStates2Group(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces, vlm_scaler=vlm_scaler),
            promotes=['*'],
        )
        self.add_subsystem('vlm_states3_group',
            VLMStates3Group(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces),
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

        self.nonlinear_solver = NonlinearBlockGS(iprint=2, maxiter=20, atol=1e-10, rtol=1e-10)
        self.linear_solver = ScipyKrylov(iprint=2, maxiter=20, atol=1e-10, rtol=1e-10)
        self.linear_solver.precon = LinearBlockGS(iprint=-1, maxiter=1)

    # def _solve_linear(self, vec_names, mode, rel_systems):
    #     if mode == 'fwd':
    #         for vec_name in vec_names:
    #             self._vectors['output'][vec_name].set_const(0.)
    #     elif mode == 'rev':
    #         for vec_name in vec_names:
    #             self._vectors['residual'][vec_name].set_const(0.)
    #
    #     return super(AerostructGroup, self)._solve_linear(vec_names, mode, rel_systems)
