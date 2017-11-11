from __future__ import print_function
import numpy as np

from openmdao.api import Group, NonlinearBlockGS, LinearBlockGS

from openaerostruct_v2.aerodynamics.vlm_preprocess_group import VLMPreprocessGroup
from openaerostruct_v2.aerodynamics.vlm_states_group import VLMStatesGroup
from openaerostruct_v2.aerodynamics.vlm_postprocess_group import VLMPostprocessGroup

from openaerostruct_v2.structures.fea_preprocess_group import FEAPreprocessGroup
from openaerostruct_v2.structures.fea_states_group import FEAStatesGroup
from openaerostruct_v2.structures.fea_postprocess_group import FEAPostprocessGroup

from openaerostruct_v2.aerostruct.load_transfer_group import LoadTransferGroup
from openaerostruct_v2.aerostruct.disp_transfer_group import DispTransferGroup


class AerostructGroup(Group):

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)
        self.metadata.declare('lifting_surfaces', types=list)
        self.metadata.declare('section_origin', types=(int, float))
        self.metadata.declare('spar_location', types=(int, float))
        self.metadata.declare('E')
        self.metadata.declare('G')

    def setup(self):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']
        section_origin = self.metadata['section_origin']
        spar_location = self.metadata['spar_location']
        E = self.metadata['E']
        G = self.metadata['G']

        self.add_subsystem('vlm_states_group',
            VLMStatesGroup(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces),
            promotes=['*'],
        )
        self.add_subsystem('load_transfer_group',
            LoadTransferGroup(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces),
            promotes=['*'],
        )
        self.add_subsystem('fea_states_group',
            FEAStatesGroup(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces),
            promotes=['*'],
        )
        self.add_subsystem('disp_transfer_group',
            DispTransferGroup(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces),
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

    E = 70.e9
    G = 25.e9

    num_points_x = 2
    num_points_z_half = 3

    num_points_z = 2 * num_points_z_half - 1

    airfoil = np.zeros(num_points_x)
    # airfoil[1:-1] = 0.2

    section_origin = 0.25
    spar_location = 0.35
    lifting_surfaces = [
        ('wing', {
            'num_points_x': num_points_x, 'num_points_z_half': num_points_z_half,
            'airfoil': airfoil,
            'chord': 1., 'twist': 0. * np.pi / 180., 'sweep_x': 0., 'dihedral_y': 0., 'span': 5,
            'twist_bspline': (2, 2),
            'sec_z_bspline': (num_points_z_half, 2),
            'chord_bspline': (2, 2),
            'thickness_bspline': (10, 3),
            'thickness' : .005,
            'radius' : 0.1,
        })
    ]

    prob = Problem()
    prob.model = Group()

    indep_var_comp = IndepVarComp()
    indep_var_comp.add_output('v_m_s', shape=num_nodes, val=200.)
    indep_var_comp.add_output('alpha_rad', shape=num_nodes, val=3. * np.pi / 180.)
    indep_var_comp.add_output('rho_kg_m3', shape=num_nodes, val=1.225)
    prob.model.add_subsystem('indep_var_comp', indep_var_comp, promotes=['*'])

    inputs_group = InputsGroup(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces)
    prob.model.add_subsystem('inputs_group', inputs_group, promotes=['*'])

    group = FEABsplineGroup(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces)
    prob.model.add_subsystem('tube_bspline_group', group, promotes=['*'])

    prob.model.add_subsystem('vlm_preprocess_group',
        VLMPreprocessGroup(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces,
            section_origin=section_origin),
        promotes=['*'],
    )
    prob.model.add_subsystem('fea_preprocess_group',
        FEAPreprocessGroup(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces,
            section_origin=section_origin, spar_location=spar_location, E=E, G=G),
        promotes=['*'],
    )

    prob.model.add_subsystem('aerostruct_group',
        AerostructGroup(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces,
            section_origin=section_origin, spar_location=spar_location, E=E, G=G),
        promotes=['*'],
    )

    prob.model.add_subsystem('vlm_postprocess_group',
        VLMPostprocessGroup(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces),
        promotes=['*'],
    )
    prob.model.add_subsystem('fea_postprocess_group',
        FEAPostprocessGroup(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces),
        promotes=['*'],
    )

    prob.setup()

    prob.run_model()
    prob.check_partials(compact_print=True)

    # print(prob['circulations'])
    # print(prob['wing_disp'])

    # view_model(prob)
