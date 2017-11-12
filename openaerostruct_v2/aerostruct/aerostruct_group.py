from __future__ import print_function
import numpy as np

from openmdao.api import Group, NonlinearBlockGS, LinearBlockGS

from openaerostruct_v2.aerodynamics.vlm_preprocess_group import VLMPreprocessGroup
from openaerostruct_v2.aerodynamics.vlm_states1_group import VLMStates1Group
from openaerostruct_v2.aerodynamics.vlm_states2_group import VLMStates2Group
from openaerostruct_v2.aerodynamics.vlm_postprocess_group import VLMPostprocessGroup

from openaerostruct_v2.structures.fea_preprocess_group import FEAPreprocessGroup
from openaerostruct_v2.structures.fea_states_group import FEAStatesGroup
from openaerostruct_v2.structures.fea_postprocess_group import FEAPostprocessGroup

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


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    from openmdao.api import Problem, IndepVarComp, pyOptSparseDriver, view_model

    from openaerostruct_v2.geometry.inputs_group import InputsGroup
    from openaerostruct_v2.structures.fea_bspline_group import FEABsplineGroup

    num_nodes = 2

    num_points_x = 2
    num_points_z_half = 3
    num_points_z = 2 * num_points_z_half - 1
    lifting_surfaces = [
        ('wing', {
            'num_points_x': num_points_x, 'num_points_z_half': num_points_z_half,
            'airfoil': np.zeros(num_points_x),
            'chord': 1., 'twist': 0. * np.pi / 180., 'sweep_x': 0., 'dihedral_y': 0., 'span': 5,
            'twist_bspline': (2, 2),
            'sec_z_bspline': (num_points_z_half, 2),
            'chord_bspline': (2, 2),
            'thickness_bspline': (10, 3),
            'thickness' : .1,
            'radius' : 1.,
        })
    ]
    wing_data = {
        'section_origin': 0.25,
        'spar_location': 0.35,
        'E': 70.e9,
        'G': 29.e9,
        'lifting_surfaces': lifting_surfaces,
        'airfoil': np.zeros(num_points_x),
    }

    prob = Problem()
    prob.model = Group()

    indep_var_comp = IndepVarComp()
    indep_var_comp.add_output('v_m_s', shape=num_nodes, val=200.)
    indep_var_comp.add_output('alpha_rad', shape=num_nodes, val=3. * np.pi / 180.)
    indep_var_comp.add_output('rho_kg_m3', shape=num_nodes, val=1.225)
    prob.model.add_subsystem('indep_var_comp', indep_var_comp, promotes=['*'])

    inputs_group = InputsGroup(num_nodes=num_nodes, wing_data=wing_data)
    prob.model.add_subsystem('inputs_group', inputs_group, promotes=['*'])

    group = FEABsplineGroup(num_nodes=num_nodes, wing_data=wing_data)
    prob.model.add_subsystem('tube_bspline_group', group, promotes=['*'])

    prob.model.add_subsystem('vlm_preprocess_group',
        VLMPreprocessGroup(num_nodes=num_nodes, wing_data=wing_data),
        promotes=['*'],
    )
    prob.model.add_subsystem('fea_preprocess_group',
        FEAPreprocessGroup(num_nodes=num_nodes, wing_data=wing_data),
        promotes=['*'],
    )

    prob.model.add_subsystem('aerostruct_group',
        AerostructGroup(num_nodes=num_nodes, wing_data=wing_data),
        promotes=['*'],
    )

    prob.model.add_subsystem('vlm_postprocess_group',
        VLMPostprocessGroup(num_nodes=num_nodes, wing_data=wing_data),
        promotes=['*'],
    )
    prob.model.add_subsystem('fea_postprocess_group',
        FEAPostprocessGroup(num_nodes=num_nodes, wing_data=wing_data),
        promotes=['*'],
    )

    prob.setup()

    prob.run_model()
    prob.check_partials(compact_print=True)

    # print(prob['circulations'])
    # print(prob['wing_disp'])

    # view_model(prob)
