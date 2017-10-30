from __future__ import print_function
import numpy as np

from openmdao.api import Group

from openaerostruct_v2.aerodynamics.vlm_group import VLMGroup
from openaerostruct_v2.structures.fea_group import FEAGroup
from openaerostruct_v2.aerostruct.components.as_load_transfer_comp import ASLoadTransferComp


class AerostructGroup(Group):

    def initialize(self):
        self.metadata.declare('lifting_surfaces', type_=list)
        self.metadata.declare('section_origin', type_=(int, float))
        self.metadata.declare('spar_location', type_=(int, float))
        self.metadata.declare('E')
        self.metadata.declare('G')

    def setup(self):
        lifting_surfaces = self.metadata['lifting_surfaces']
        section_origin = self.metadata['section_origin']
        spar_location = self.metadata['spar_location']
        E = self.metadata['E']
        G = self.metadata['G']

        self.add_subsystem('vlm_group',
            VLMGroup(section_origin=section_origin, lifting_surfaces=lifting_surfaces),
            promotes=['*'],
        )

        self.add_subsystem('fea_group',
            FEAGroup(section_origin=section_origin, lifting_surfaces=lifting_surfaces,
                spar_location=spar_location, E=E, G=G),
            promotes=['*'],
        )

        self.add_subsystem('as_load_transfer_comp',
            ASLoadTransferComp(lifting_surfaces=lifting_surfaces),
        )


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    from openmdao.api import Problem, IndepVarComp, pyOptSparseDriver, view_model

    from openaerostruct_v2.geometry.inputs_group import InputsGroup

    E = 1.e1
    G = 1.e1

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
            'thickness' : .1,
            'radius' : 1.,
        })
    ]

    prob = Problem()
    prob.model = Group()

    indep_var_comp = IndepVarComp()
    indep_var_comp.add_output('v_m_s', 200.)
    indep_var_comp.add_output('alpha_rad', 3. * np.pi / 180.)
    indep_var_comp.add_output('rho_kg_m3', 1.225)
    prob.model.add_subsystem('indep_var_comp', indep_var_comp, promotes=['*'])

    inputs_group = InputsGroup(lifting_surfaces=lifting_surfaces)
    prob.model.add_subsystem('inputs_group', inputs_group, promotes=['*'])

    prob.model.add_subsystem('aerostruct_group',
        AerostructGroup(section_origin=section_origin, lifting_surfaces=lifting_surfaces,
            spar_location=spar_location, E=E, G=G),
        promotes=['*'],
    )

    prob.setup()

    prob.run_model()
    prob.check_partials(compact_print=True)
