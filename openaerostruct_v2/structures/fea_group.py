from __future__ import print_function
import numpy as np

from openmdao.api import Group

from openaerostruct_v2.structures.components.fea_mesh_comp import FEAMeshComp
from openaerostruct_v2.structures.components.tube_properties_comp import TubePropertiesComp
from openaerostruct_v2.structures.components.fea_transform_comp import FEATransformComp
from openaerostruct_v2.structures.components.fea_length_comp import FEALengthComp
from openaerostruct_v2.structures.components.fea_local_stiff_comp import FEALocalStiffComp
from openaerostruct_v2.structures.components.fea_local_stiff_permuted_comp import FEALocalStiffPermutedComp
from openaerostruct_v2.structures.components.fea_local_stiff_transformed_comp import FEALocalStiffTransformedComp
from openaerostruct_v2.structures.components.fea_global_stiff_comp import FEAGlobalStiffComp


class FEAGroup(Group):

    def initialize(self):
        self.metadata.declare('num', type_=int)
        self.metadata.declare('section_origin', type_=(int, float))
        self.metadata.declare('spar_location', type_=(int, float))
        self.metadata.declare('lifting_surfaces', type_=list)
        self.metadata.declare('E')
        self.metadata.declare('G')

    def setup(self):
        num = self.metadata['num']
        section_origin = self.metadata['section_origin']
        spar_location = self.metadata['spar_location']
        lifting_surfaces = self.metadata['lifting_surfaces']
        E = self.metadata['E']
        G = self.metadata['G']

        comp = FEAMeshComp(lifting_surfaces=lifting_surfaces, section_origin=section_origin,
            spar_location=spar_location)
        self.add_subsystem('fea_mesh_comp', comp, promotes=['*'])

        comp = TubePropertiesComp(lifting_surfaces=lifting_surfaces)
        self.add_subsystem('tube_properties_comp', comp, promotes=['*'])

        comp = FEATransformComp(lifting_surfaces=lifting_surfaces)
        self.add_subsystem('fea_transform_comp', comp, promotes=['*'])

        comp = FEALengthComp(lifting_surfaces=lifting_surfaces)
        self.add_subsystem('fea_length_comp', comp, promotes=['*'])

        comp = FEALocalStiffComp(lifting_surfaces=lifting_surfaces, E=E, G=G)
        self.add_subsystem('fea_local_stiff_comp', comp, promotes=['*'])

        comp = FEALocalStiffPermutedComp(lifting_surfaces=lifting_surfaces)
        self.add_subsystem('fea_local_stiff_permuted_comp', comp, promotes=['*'])

        comp = FEALocalStiffTransformedComp(lifting_surfaces=lifting_surfaces)
        self.add_subsystem('fea_local_stiff_transformed_comp', comp, promotes=['*'])

        comp = FEAGlobalStiffComp(lifting_surfaces=lifting_surfaces)
        self.add_subsystem('fea_global_stiff_comp', comp, promotes=['*'])


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    from openmdao.api import Problem, IndepVarComp, pyOptSparseDriver, view_model

    from openaerostruct_v2.geometry.inputs_group import InputsGroup

    E = 1.
    G = 1.

    num_points_x = 2
    num_points_z_half = 3

    num_points_z = 2 * num_points_z_half - 1

    airfoil = np.zeros(num_points_x)
    # airfoil[1:-1] = 0.2

    num = 1
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
        })
    ]

    prob = Problem()
    prob.model = Group()

    indep_var_comp = IndepVarComp()
    indep_var_comp.add_output('v_m_s', 200.)
    indep_var_comp.add_output('alpha_rad', 3. * np.pi / 180.)
    indep_var_comp.add_output('rho_kg_m3', 1.225)
    indep_var_comp.add_output('wing_tube_radius', shape=num_points_z - 1)
    indep_var_comp.add_output('wing_tube_thickness', shape=num_points_z - 1)
    prob.model.add_subsystem('indep_var_comp', indep_var_comp, promotes=['*'])

    inputs_group = InputsGroup(lifting_surfaces=lifting_surfaces)
    prob.model.add_subsystem('inputs_group', inputs_group, promotes=['*'])

    prob.model.add_subsystem('fea_group',
        FEAGroup(num=num, section_origin=section_origin, lifting_surfaces=lifting_surfaces,
            spar_location=spar_location, E=E, G=G),
        promotes=['*'],
    )

    prob.setup()

    prob['wing_chord_cp'] = [0.5, 1.0, 0.5]
    prob['wing_tube_radius'] = 1.0
    prob['wing_tube_thickness'] = 0.1

    prob.run_model()
    prob.check_partials(compact_print=True)
    # exit()
