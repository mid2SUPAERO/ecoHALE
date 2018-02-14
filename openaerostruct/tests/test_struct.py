from __future__ import print_function
import numpy as np
import unittest
from numpy.testing import assert_almost_equal

from openmdao.api import Problem, IndepVarComp, ScipyOptimizeDriver, view_model, Group, ExecComp, SqliteRecorder

from openaerostruct.geometry.inputs_group import InputsGroup
from openaerostruct.structures.fea_bspline_group import FEABsplineGroup
from openaerostruct.structures.fea_preprocess_group import FEAPreprocessGroup
from openaerostruct.structures.fea_states_group import FEAStatesGroup
from openaerostruct.structures.fea_postprocess_group import FEAPostprocessGroup

from openaerostruct.tests.utils import get_default_lifting_surfaces
from openaerostruct.common.lifting_surface import LiftingSurface


class TestStruct(unittest.TestCase):

    def setup_struct(self):

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

        wing_loads = np.zeros((num_nodes, num_points_z, 6))
        wing_loads[0, :, 1] = 1e3
        if num_nodes > 1:
            wing_loads[1, :, 0] = 1e3

        fea_scaler = 1e0

        prob = Problem()
        prob.model = Group()

        indep_var_comp = IndepVarComp()
        indep_var_comp.add_output('v_m_s', shape=num_nodes, val=200.)
        indep_var_comp.add_output('alpha_rad', shape=num_nodes, val=3. * np.pi / 180.)
        indep_var_comp.add_output('rho_kg_m3', shape=num_nodes, val=1.225)
        indep_var_comp.add_output('wing_loads', val=wing_loads)
        prob.model.add_subsystem('indep_var_comp', indep_var_comp, promotes=['*'])

        inputs_group = InputsGroup(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces)
        prob.model.add_subsystem('inputs_group', inputs_group, promotes=['*'])

        prob.model.add_subsystem('tube_bspline_group',
            FEABsplineGroup(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces),
            promotes=['*'])

        prob.model.add_subsystem('fea_preprocess_group',
            FEAPreprocessGroup(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces, fea_scaler=fea_scaler),
            promotes=['*'],
        )
        prob.model.add_subsystem('fea_states_group',
            FEAStatesGroup(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces, fea_scaler=fea_scaler),
            promotes=['*'],
        )
        prob.model.add_subsystem('fea_postprocess_group',
            FEAPostprocessGroup(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces),
            promotes=['*'],
        )
        prob.model.add_subsystem('objective',
            ExecComp('obj=1000*sum(compliance)', compliance=np.zeros(num_nodes)),
            promotes=['*'],
        )

        prob.model.add_design_var('wing_tube_thickness_dv', lower=0.001, upper=0.1, scaler=1e3)
        prob.model.add_objective('structural_weight', scaler=1e0)
        prob.model.add_constraint('wing_ks', upper=0.)

        prob.driver = ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.opt_settings['Major optimality tolerance'] = 2e-12
        prob.driver.opt_settings['Major feasibility tolerance'] = 2e-12

        prob.setup()

        prob['wing_chord_dv'] = [0.5, 1.0, 0.5]

        return prob

    def test_struct_analysis(self):
        prob = self.setup_struct()
        prob.run_model()
        assert_almost_equal(prob['obj'], 16170.5464818)

    def test_struct_optimization(self):
        prob = self.setup_struct()
        prob.run_driver()
        assert_almost_equal(prob['obj'], 384729.7510236)

if __name__ == "__main__":
    unittest.main()
