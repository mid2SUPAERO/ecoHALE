from __future__ import print_function
import unittest
import numpy as np
from numpy.testing import assert_almost_equal

from openmdao.api import Problem, IndepVarComp, ScipyOptimizeDriver, view_model, Group, ExecComp, SqliteRecorder

from openaerostruct.geometry.inputs_group import InputsGroup
from openaerostruct.aerodynamics.vlm_preprocess_group import VLMPreprocessGroup
from openaerostruct.aerodynamics.vlm_states_group import VLMStatesGroup
from openaerostruct.aerodynamics.vlm_postprocess_group import VLMPostprocessGroup

from openaerostruct.tests.utils import get_default_lifting_surfaces
from openaerostruct.common.lifting_surface import LiftingSurface


class TestAero(unittest.TestCase):

    def setup_aero(self):

        mode = 1

        check_derivs = mode == 0

        num_nodes = 1 if not check_derivs else 2

        num_points_x = 2
        num_points_z_half = 2
        num_points_z = 2 * num_points_z_half - 1
        g = 9.81

        lifting_surface = LiftingSurface('wing')

        lifting_surface.initialize_mesh(num_points_x, num_points_z_half, airfoil_x=np.linspace(0., 1., num_points_x), airfoil_y=np.zeros(num_points_x))
        lifting_surface.set_mesh_parameters(distribution='sine', section_origin=.25)
        lifting_surface.set_structural_properties(E=70.e9, G=29.e9, spar_location=0.35, sigma_y=200e6, rho=2700)
        lifting_surface.set_aero_properties(factor2=.119, factor4=-0.064, cl_factor=1.05)

        lifting_surface.set_chord(1.)
        lifting_surface.set_twist(0.)
        lifting_surface.set_sweep(0.)
        lifting_surface.set_dihedral(0.)
        lifting_surface.set_span(5.)
        lifting_surface.set_thickness(0.05)
        lifting_surface.set_radius(0.1)


        lifting_surfaces = [('wing', lifting_surface)]

        vlm_scaler = 1e0

        prob = Problem()
        prob.model = Group()

        indep_var_comp = IndepVarComp()
        indep_var_comp.add_output('v_m_s', shape=num_nodes, val=200.)
        indep_var_comp.add_output('alpha_rad', shape=num_nodes, val=3. * np.pi / 180.)
        indep_var_comp.add_output('rho_kg_m3', shape=num_nodes, val=1.225)
        indep_var_comp.add_output('Re_1e6', shape=num_nodes, val=2.)
        indep_var_comp.add_output('C_l_max', shape=num_nodes, val=1.5)
        indep_var_comp.add_output('reference_area', shape=num_nodes, val=7.5)
        prob.model.add_subsystem('indep_var_comp', indep_var_comp, promotes=['*'])

        inputs_group = InputsGroup(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces)
        prob.model.add_subsystem('inputs_group', inputs_group, promotes=['*'])

        prob.model.add_subsystem('vlm_preprocess_group',
            VLMPreprocessGroup(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces),
            promotes=['*'],
        )
        prob.model.add_subsystem('vlm_states_group',
            VLMStatesGroup(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces, vlm_scaler=vlm_scaler),
            promotes=['*'],
        )
        prob.model.add_subsystem('vlm_postprocess_group',
            VLMPostprocessGroup(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces),
            promotes=['*'],
        )
        prob.model.add_subsystem('objective',
            ExecComp('obj=sum(C_D)', C_D=np.zeros(num_nodes)),
            promotes=['*'],
        )

        prob.model.add_design_var('alpha_rad', lower=-3.*np.pi/180., upper=8.*np.pi/180.)
        prob.model.add_design_var('wing_twist_dv', lower=-3.*np.pi/180., upper=8.*np.pi/180.)
        prob.model.add_objective('obj')
        prob.model.add_constraint('C_L', equals=np.linspace(0.4, 0.6, num_nodes))

        prob.driver = ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.opt_settings['Major optimality tolerance'] = 1e-12
        prob.driver.opt_settings['Major feasibility tolerance'] = 1e-12

        if mode == 2:
            prob.driver.add_recorder(SqliteRecorder('aero.hst'))
            prob.driver.recording_options['includes'] = ['*']

        prob.setup()

        prob['wing_chord_dv'] = [0.5, 1.0, 0.5]

        return prob

    def test_aero_analysis(self):
        prob = self.setup_aero()
        prob.run_model()
        assert_almost_equal(prob['obj'], 0.0020292)

    def test_aero_optimization(self):
        prob = self.setup_aero()
        prob.run_driver()
        assert_almost_equal(prob['obj'],  0.0048556)

if __name__ == "__main__":
    unittest.main()
