from __future__ import print_function
import numpy as np

from openmdao.api import Group

from openaerostruct_v2.structures.components.tube_properties_comp import TubePropertiesComp
from openaerostruct_v2.structures.components.fea_mesh_comp import FEAMeshComp
from openaerostruct_v2.structures.components.fea_transform_comp import FEATransformComp
from openaerostruct_v2.structures.components.fea_length_comp import FEALengthComp
from openaerostruct_v2.structures.components.fea_local_stiff_comp import FEALocalStiffComp
from openaerostruct_v2.structures.components.fea_local_stiff_permuted_comp import FEALocalStiffPermutedComp
from openaerostruct_v2.structures.components.fea_local_stiff_transformed_comp import FEALocalStiffTransformedComp
from openaerostruct_v2.structures.components.fea_global_stiff_comp import FEAGlobalStiffComp
from openaerostruct_v2.structures.components.fea_forces_comp import FEAForcesComp
from openaerostruct_v2.structures.components.fea_states_comp import FEAStatesComp
from openaerostruct_v2.structures.components.fea_disp_comp import FEADispComp
from openaerostruct_v2.structures.components.fea_volume_comp import FEAVolumeComp
from openaerostruct_v2.structures.components.fea_compliance_comp import FEAComplianceComp


class FEAGroup(Group):

    def initialize(self):
        self.metadata.declare('section_origin', type_=(int, float))
        self.metadata.declare('spar_location', type_=(int, float))
        self.metadata.declare('lifting_surfaces', type_=list)
        self.metadata.declare('E')
        self.metadata.declare('G')

    def setup(self):
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

        comp = FEAForcesComp(lifting_surfaces=lifting_surfaces)
        self.add_subsystem('fea_forces_comp', comp, promotes=['*'])

        comp = FEAStatesComp(lifting_surfaces=lifting_surfaces)
        self.add_subsystem('fea_states_comp', comp, promotes=['*'])

        comp = FEADispComp(lifting_surfaces=lifting_surfaces)
        self.add_subsystem('fea_disp_comp', comp, promotes=['*'])

        comp = FEAVolumeComp(lifting_surfaces=lifting_surfaces)
        self.add_subsystem('fea_volume_comp', comp, promotes=['*'])

        comp = FEAComplianceComp(lifting_surfaces=lifting_surfaces)
        self.add_subsystem('fea_compliance_comp', comp, promotes=['*'])


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    from openmdao.api import Problem, IndepVarComp, pyOptSparseDriver, view_model

    from openaerostruct_v2.geometry.inputs_group import InputsGroup
    from openaerostruct_v2.structures.fea_bspline_group import FEABsplineGroup

    E = 1.e11
    G = 1.e11

    num_points_x = 2
    num_points_z_half = 30

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
    indep_var_comp.add_output('wing_loads', np.outer(np.ones(num_points_z), np.array([0., 1., 0., 0., 0., 0.])))
    prob.model.add_subsystem('indep_var_comp', indep_var_comp, promotes=['*'])

    inputs_group = InputsGroup(lifting_surfaces=lifting_surfaces)
    prob.model.add_subsystem('inputs_group', inputs_group, promotes=['*'])

    group = FEABsplineGroup(lifting_surfaces=lifting_surfaces)
    prob.model.add_subsystem('tube_bspline_group', group, promotes=['*'])

    prob.model.add_subsystem('fea_group',
        FEAGroup(section_origin=section_origin, lifting_surfaces=lifting_surfaces,
            spar_location=spar_location, E=E, G=G),
        promotes=['*'],
    )

    prob.model.add_design_var('wing_tube_thickness_cp', lower=0.01)
    prob.model.add_objective('compliance')
    prob.model.add_constraint('structural_volume', equals=10)

    prob.driver = pyOptSparseDriver()
    prob.driver.options['optimizer'] = 'SNOPT'
    prob.driver.opt_settings['Major optimality tolerance'] = 2e-7
    prob.driver.opt_settings['Major feasibility tolerance'] = 2e-7
    # prob.driver.opt_settings['Verify level'] = 3

    prob.setup()

    prob['wing_chord_cp'] = [0.5, 1.0, 0.5]

    prob.run_model()
    # prob.check_partials(compact_print=True)
    # exit()

    print(prob['structural_volume'])

    prob.run_driver()

    print(prob['wing_tube_thickness'])

    print(prob['wing_disp'])

    import matplotlib.pyplot as plt
    x = prob['wing_fea_mesh']
    plt.plot(0.5 * x[:-1] + 0.5 * x[1:], prob['wing_tube_thickness'])
    plt.show()
