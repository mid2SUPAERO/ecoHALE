from __future__ import division, print_function
import sys
from time import time
import unittest
import numpy as np

from openaerostruct.geometry.utils import generate_mesh
from openaerostruct.geometry.geometry_group import Geometry
from openaerostruct.transfer.displacement_transfer import DisplacementTransfer
from openaerostruct.structures.struct_groups import SpatialBeamAlone

from openaerostruct.aerodynamics.aero_groups import AeroPoint

from openaerostruct.integration.aerostruct_groups import Aerostruct, AerostructPoint
from openaerostruct.aerodynamics.states import VLMStates
from openaerostruct.geometry.utils import generate_mesh
from openaerostruct.geometry.geometry_group import Geometry

from openmdao.api import IndepVarComp, Problem, Group, NewtonSolver, ScipyIterativeSolver, LinearBlockGS, NonlinearBlockGS, DirectSolver, DenseJacobian, LinearRunOnce, PetscKSP, ScipyOptimizer# TODO, SqliteRecorder, CaseReader, profile
from openmdao.api import view_model
from six import iteritems

# Append the parent directory to the system path so we can call those Python
# files. If you have OpenAeroStruct in your PYTHONPATH, this is not necessary.
from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

try:
    from openaerostruct.fortran import OAS_API
    fortran_flag = True
except:
    fortran_flag = False

class TestAero(unittest.TestCase):

    currentResult = []

    def run(self, result=None):
        self.currentResult.append(result) # remember result for use in tearDown
        unittest.TestCase.run(self, result) # call superclass run method

    # def test_aero_analysis_flat(self):
    #     OAS_prob = OASProblem({'type' : 'aero',
    #                            'optimize' : False,
    #                            'record_db' : False})
    #     OAS_prob.add_surface({'span_cos_spacing' : 0})
    #     OAS_prob.setup()
    #     OAS_prob.run()
    #     prob = OAS_prob.prob
    #
    #     self.assertAlmostEqual(prob['wing_perf.CL'][0], .46173591841167, places=5)
    #     self.assertAlmostEqual(prob['wing_perf.CD'][0], .005524603647, places=5)
    #
    # def test_aero_analysis_flat_multiple(self):
    #     OAS_prob = OASProblem({'type' : 'aero',
    #                            'optimize' : False,
    #                            'record_db' : False})
    #     OAS_prob.add_surface({'span_cos_spacing' : 0.})
    #     OAS_prob.add_surface({'name' : 'tail',
    #                           'span_cos_spacing' : 0.,
    #                           'offset' : np.array([0., 0., 1000000.])})
    #     OAS_prob.setup()
    #     OAS_prob.run()
    #     prob = OAS_prob.prob
    #     self.assertAlmostEqual(prob['wing_perf.CL'][0], .46173591841167, places=5)
    #     self.assertAlmostEqual(prob['tail_perf.CL'][0], .46173591841167, places=5)
    #
    # def test_aero_analysis_flat_side_by_side(self):
    #     OAS_prob = OASProblem({'type' : 'aero',
    #                            'optimize' : False,
    #                            'record_db' : False})
    #     OAS_prob.add_surface({'name' : 'wing',
    #                           'span' : 5.,
    #                           'num_y' : 3,
    #                           'span_cos_spacing' : 0.,
    #                           'symmetry' : False,
    #                           'offset' : np.array([0., -2.5, 0.])})
    #     OAS_prob.add_surface({'name' : 'tail',
    #                           'span' : 5.,
    #                           'num_y' : 3,
    #                           'span_cos_spacing' : 0.,
    #                           'symmetry' : False,
    #                           'offset' : np.array([0., 2.5, 0.])})
    #     OAS_prob.setup()
    #     OAS_prob.run()
    #     prob = OAS_prob.prob
    #     self.assertAlmostEqual(prob['wing_perf.CL'][0], 0.46173591841167183, places=5)
    #     self.assertAlmostEqual(prob['tail_perf.CL'][0], 0.46173591841167183, places=5)
    #     self.assertAlmostEqual(prob['wing_perf.CD'][0], .005524603647, places=5)
    #     self.assertAlmostEqual(prob['tail_perf.CD'][0], .005524603647, places=5)
    #
    # def test_aero_analysis_flat_full(self):
    #     OAS_prob = OASProblem({'type' : 'aero',
    #                            'optimize' : False,
    #                            'record_db' : False})
    #     surf_dict = {'symmetry' : False}
    #     OAS_prob.add_surface(surf_dict)
    #     OAS_prob.setup()
    #     OAS_prob.run()
    #     prob = OAS_prob.prob
    #     self.assertAlmostEqual(prob['wing_perf.CL'][0], .45655138, places=5)
    #     self.assertAlmostEqual(prob['wing_perf.CD'][0], 0.0055402121081108589, places=5)
    #
    # def test_aero_analysis_flat_viscous_full(self):
    #     OAS_prob = OASProblem({'type' : 'aero',
    #                            'optimize' : False,
    #                            'record_db' : False,
    #                            'with_viscous' : True})
    #     surf_dict = {'symmetry' : False}
    #     OAS_prob.add_surface(surf_dict)
    #     OAS_prob.setup()
    #     OAS_prob.run()
    #     prob = OAS_prob.prob
    #     self.assertAlmostEqual(prob['wing_perf.CL'][0], .45655138, places=5)
    #     self.assertAlmostEqual(prob['wing_perf.CD'][0], 0.01989351, places=5)

    def test_aero_optimization(self):
        # Create a dictionary to store options about the surface
        mesh_dict = {'num_y' : 7,
                     'num_x' : 2,
                     'wing_type' : 'CRM',
                     'symmetry' : True,
                     'num_twist_cp' : 5}

        mesh, twist_cp = generate_mesh(mesh_dict)

        surf_dict = {
                    # Wing definition
                    'name' : 'wing',        # name of the surface
                    'type' : 'aero',
                    'symmetry' : True,     # if true, model one half of wing
                                            # reflected across the plane y = 0
                    'S_ref_type' : 'wetted', # how we compute the wing area,
                                             # can be 'wetted' or 'projected'

                    'num_twist_cp' : 5,

                    # Aerodynamic performance of the lifting surface at
                    # an angle of attack of 0 (alpha=0).
                    # These CL0 and CD0 values are added to the CL and CD
                    # obtained from aerodynamic analysis of the surface to get
                    # the total CL and CD.
                    # These CL0 and CD0 values do not vary wrt alpha.
                    'CL0' : 0.0,            # CL of the surface at alpha=0
                    'CD0' : 0.015,            # CD of the surface at alpha=0

                    'fem_origin' : 0.35,

                    # Airfoil properties for viscous drag calculation
                    'k_lam' : 0.05,         # percentage of chord with laminar
                                            # flow, used for viscous drag
                    't_over_c' : 0.15,      # thickness over chord ratio (NACA0015)
                    'c_max_t' : .303,       # chordwise location of maximum (NACA0015)
                                            # thickness
                    'with_viscous' : True,  # if true, compute viscous drag
                    }

        surf_dict.update({'twist_cp' : twist_cp,
                          'mesh' : mesh})

        surf_dict['num_x'], surf_dict['num_y'] = surf_dict['mesh'].shape[:2]

        surfaces = [surf_dict]

        # Create the problem and the model group
        prob = Problem()

        indep_var_comp = IndepVarComp()
        indep_var_comp.add_output('v', val=248.136)
        indep_var_comp.add_output('alpha', val=5.)
        indep_var_comp.add_output('M', val=0.84)
        indep_var_comp.add_output('re', val=1.e6)
        indep_var_comp.add_output('rho', val=0.38)
        indep_var_comp.add_output('S_ref_total', val=0.)
        indep_var_comp.add_output('cg', val=np.zeros((3)))

        prob.model.add_subsystem('prob_vars',
            indep_var_comp,
            promotes=['*'])

        # Loop over each surface in the surfaces list
        for surface in surfaces:

            geom_group = Geometry(surface=surface)

            # Add tmp_group to the problem as the name of the surface.
            # Note that is a group and performance group for each
            # individual surface.
            prob.model.add_subsystem(surface['name'], geom_group)

        # Loop through and add a certain number of aero points
        for i in range(1):

            # Create the aero point group and add it to the model
            aero_group = AeroPoint(surfaces=surfaces)
            point_name = 'aero_point_{}'.format(i)
            prob.model.add_subsystem(point_name, aero_group)

            # Connect flow properties to the analysis point
            prob.model.connect('v', point_name + '.v')
            prob.model.connect('alpha', point_name + '.alpha')
            prob.model.connect('M', point_name + '.M')
            prob.model.connect('re', point_name + '.re')
            prob.model.connect('rho', point_name + '.rho')
            prob.model.connect('S_ref_total', point_name + '.S_ref_total')
            prob.model.connect('cg', point_name + '.cg')

            # Connect the parameters within the model for each aero point
            for surface in surfaces:

                name = surface['name']

                # Connect the mesh from the geometry component to the analysis point
                prob.model.connect(name + '.def_mesh', point_name + '.' + name + '.def_mesh')

                # Perform the connections with the modified names within the
                # 'aero_states' group.
                prob.model.connect(name + '.def_mesh', point_name + '.aero_states.' + name + '_def_mesh')

        from openmdao.api import pyOptSparseDriver
        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = "SNOPT"
        prob.driver.opt_settings = {'Major optimality tolerance': 1.0e-8,
                                    'Major feasibility tolerance': 1.0e-8}

        # # Setup problem and add design variables, constraint, and objective
        prob.model.add_design_var('wing.twist_cp', lower=-10., upper=15.)
        prob.model.add_constraint(point_name + '.wing_perf.CL', equals=0.5)
        prob.model.add_objective(point_name + '.wing_perf.CD', scaler=1e4)

        # Set up the problem
        prob.setup()

        # prob.run_model()
        prob.run_driver()

        self.assertAlmostEqual(prob['aero_point_0.wing_perf.CD'][0], 0.03227945)


    # TODO: implement this when it's in Blue
    # if fortran_flag:
    #     def test_aero_optimization_fd(self):
    #         # Need to use SLSQP here because SNOPT finds a different optimum
    #         OAS_prob = OASProblem({'type' : 'aero',
    #                                'optimize' : True,
    #                                'record_db' : False,
    #                                'optimizer' : 'SLSQP',
    #                                'force_fd' : True})
    #         OAS_prob.add_surface()
    #
    #         OAS_prob.add_design_var('wing.twist_cp', lower=-10., upper=15.)
    #         OAS_prob.add_design_var('wing.sweep', lower=10., upper=30.)
    #         OAS_prob.add_design_var('wing.dihedral', lower=-10., upper=20.)
    #         OAS_prob.add_design_var('wing.taper', lower=.5, upper=2.)
    #         OAS_prob.add_constraint('wing_perf.CL', equals=0.5)
    #         OAS_prob.add_objective('wing_perf.CD', scaler=1e4)
    #
    #         OAS_prob.setup()
    #
    #         OAS_prob.run()
    #         prob = OAS_prob.prob
    #         self.assertAlmostEqual(prob['wing_perf.CD'][0], 0.0038513637269919619, places=5)

    # if fortran_flag:
    #     def test_aero_optimization_chord_monotonic(self):
    #         OAS_prob = OASProblem({'type' : 'aero',
    #                                'optimize' : True,
    #                                'record_db' : False,
    #                                'with_viscous' : False})
    #         OAS_prob.add_surface({
    #         'chord_cp' : np.random.random(5),
    #         'num_y' : 11,
    #         'monotonic_con' : ['chord'],
    #         'span_cos_spacing' : 0.,
    #         })
    #
    #         OAS_prob.add_design_var('wing.chord_cp', lower=0.1, upper=5.)
    #         OAS_prob.add_design_var('alpha', lower=-10., upper=10.)
    #         OAS_prob.add_constraint('wing_perf.CL', equals=0.1)
    #         OAS_prob.add_constraint('wing.S_ref', equals=20)
    #         OAS_prob.add_constraint('wing.monotonic_chord', upper=0.)
    #         OAS_prob.add_objective('wing_perf.CD', scaler=1e4)
    #
    #         OAS_prob.setup()
    #
    #         OAS_prob.run()
    #         prob = OAS_prob.prob
    #         self.assertAlmostEqual(prob['wing_perf.CD'][0], 0.00057432581266351113, places=5)
    #         self.assertAlmostEqual(prob['wing.monotonic_chord'][0], -1.710374671671999, places=4)
    #
    # if fortran_flag:
    #     def test_aero_optimization_chord_monotonic_no_sym(self):
    #         OAS_prob = OASProblem({'type' : 'aero',
    #                                'optimize' : True,
    #                                'record_db' : False,
    #                                'symmetry' : False,
    #                                'with_viscous' : False})
    #         OAS_prob.add_surface({
    #         'chord_cp' : np.random.random(5),
    #         'num_y' : 11,
    #         'monotonic_con' : ['chord'],
    #         'span_cos_spacing' : 0.,
    #         })
    #
    #         OAS_prob.add_design_var('wing.chord_cp', lower=0.1, upper=5.)
    #         OAS_prob.add_design_var('alpha', lower=-10., upper=10.)
    #         OAS_prob.add_constraint('wing_perf.CL', equals=0.1)
    #         OAS_prob.add_constraint('wing.S_ref', equals=20)
    #         OAS_prob.add_constraint('wing.monotonic_chord', upper=0.)
    #         OAS_prob.add_objective('wing_perf.CD', scaler=1e4)
    #
    #         OAS_prob.setup()
    #
    #         OAS_prob.run()
    #         prob = OAS_prob.prob
    #         self.assertAlmostEqual(prob['wing_perf.CD'][0], 0.00057432581266351113, places=5)
    #         self.assertAlmostEqual(prob['wing.monotonic_chord'][0], -1.710374671671999, places=2)
    #
    # if fortran_flag:
    #     def test_aero_viscous_optimization(self):
    #         OAS_prob = OASProblem({'type' : 'aero',
    #                                'optimize' : True,
    #                                'record_db' : False,
    #                                'with_viscous' : True})
    #         OAS_prob.add_surface()
    #
    #         OAS_prob.add_design_var('wing.twist_cp', lower=-10., upper=15.)
    #         OAS_prob.add_design_var('wing.sweep', lower=10., upper=30.)
    #         OAS_prob.add_design_var('wing.dihedral', lower=-10., upper=20.)
    #         OAS_prob.add_constraint('wing_perf.CL', equals=0.5)
    #         OAS_prob.add_objective('wing_perf.CD', scaler=1e4)
    #
    #         OAS_prob.setup()
    #
    #         OAS_prob.run()
    #         prob = OAS_prob.prob
    #         self.assertAlmostEqual(prob['wing_perf.CD'][0], 0.0202008, places=5)
    #
    # if fortran_flag:
    #     def test_aero_viscous_chord_optimization(self):
    #         # Need to use SLSQP here because SNOPT finds a different optimum
    #         OAS_prob = OASProblem({'type' : 'aero',
    #                                'optimize' : True,
    #                                'record_db' : False,
    #                                'optimizer' : 'SLSQP',
    #                                'with_viscous' : True})
    #         OAS_prob.add_surface()
    #
    #         OAS_prob.add_design_var('wing.chord_cp', lower=0.1, upper=3.)
    #         OAS_prob.add_design_var('alpha', lower=-10., upper=10.)
    #         OAS_prob.add_constraint('wing_perf.CL', equals=0.5)
    #         OAS_prob.add_constraint('wing.S_ref', equals=20)
    #         OAS_prob.add_objective('wing_perf.CD', scaler=1e4)
    #
    #         OAS_prob.setup()
    #
    #         OAS_prob.run()
    #         prob = OAS_prob.prob
    #         self.assertAlmostEqual(prob['wing_perf.CD'][0], 0.02439342, places=5)
    #
    # if fortran_flag:
    #     def test_aero_multiple_opt(self):
    #         OAS_prob = OASProblem({'type' : 'aero',
    #                                'optimize' : True,
    #                                'record_db' : False})
    #         surf_dict = { 'name' : 'wing',
    #                       'span' : 5.,
    #                       'num_y' : 3,
    #                       'span_cos_spacing' : 0.}
    #         OAS_prob.add_surface(surf_dict)
    #         surf_dict.update({'name' : 'tail',
    #                        'offset' : np.array([0., 0., 10.])})
    #         OAS_prob.add_surface(surf_dict)
    #
    #         OAS_prob.add_design_var('tail.twist_cp', lower=-10., upper=15.)
    #         OAS_prob.add_design_var('tail.sweep', lower=10., upper=30.)
    #         OAS_prob.add_design_var('tail.dihedral', lower=-10., upper=20.)
    #         OAS_prob.add_design_var('tail.taper', lower=.5, upper=2.)
    #         OAS_prob.add_constraint('tail_perf.CL', equals=0.5)
    #         OAS_prob.add_objective('tail_perf.CD', scaler=1e4)
    #
    #         OAS_prob.setup()
    #
    #         OAS_prob.run()
    #         prob = OAS_prob.prob
    #         self.assertAlmostEqual(prob['wing_perf.CL'][0], 0.41532382375677429, places=4)
    #         self.assertAlmostEqual(prob['tail_perf.CL'][0], .5, places=5)
    #         self.assertAlmostEqual(prob['wing_perf.CD'][0], .0075400306289957033, places=5)
    #         self.assertAlmostEqual(prob['tail_perf.CD'][0], 0.008087914662238814, places=5)


class TestStruct(unittest.TestCase):

    currentResult = []

    def run(self, result=None):
        self.currentResult.append(result) # remember result for use in tearDown
        unittest.TestCase.run(self, result) # call superclass run method

    # def test_struct_analysis(self):
    #     OAS_prob = OASProblem({'type' : 'struct',
    #                            'optimize' : False,
    #                            'record_db' : False})
    #     surf_dict = {'symmetry' : False}
    #     OAS_prob.add_surface(surf_dict)
    #     OAS_prob.setup()
    #     OAS_prob.run()
    #     prob = OAS_prob.prob
    #     self.assertAlmostEqual(prob['wing.structural_weight'][0], 988.13495481064024, places=3)
    #
    # def test_struct_analysis_symmetry(self):
    #     OAS_prob = OASProblem({'type' : 'struct',
    #                            'optimize' : False,
    #                            'record_db' : False})
    #     surf_dict = {'symmetry' : True}
    #     OAS_prob.add_surface(surf_dict)
    #     OAS_prob.setup()
    #     OAS_prob.run()
    #     prob = OAS_prob.prob
    #     self.assertAlmostEqual(prob['wing.structural_weight'][0], 988.13495481063956, places=3)

    # if fortran_flag:
    #     def test_struct_optimization(self):
    #         OAS_prob = OASProblem({'type' : 'struct',
    #                                'optimize' : True,
    #                                'record_db' : False})
    #         OAS_prob.add_surface({'symmetry' : False})
    #
    #         OAS_prob.add_design_var('wing.thickness_cp', lower=0.001, upper=0.25, scaler=1e2)
    #         OAS_prob.add_constraint('wing.failure', upper=0.)
    #         OAS_prob.add_constraint('wing.thickness_intersects', upper=0.)
    #         OAS_prob.add_objective('wing.structural_weight', scaler=1e-3)
    #
    #         OAS_prob.setup()
    #
    #         OAS_prob.run()
    #         prob = OAS_prob.prob
    #
    #         self.assertAlmostEqual(prob['wing.structural_weight'][0], 1154.4491377169238, places=2)

    def test_struct_optimization_symmetry(self):
        # Create a dictionary to store options about the surface
        mesh_dict = {'num_y' : 7,
                     'wing_type' : 'CRM',
                     'symmetry' : True,
                     'num_twist_cp' : 5}

        mesh, twist_cp = generate_mesh(mesh_dict)

        surf_dict = {
                    # Wing definition
                    'name' : 'wing',        # name of the surface
                    'type' : 'structural',
                    'symmetry' : True,     # if true, model one half of wing
                                            # reflected across the plane y = 0

                    # Structural values are based on aluminum 7075
                    'E' : 70.e9,            # [Pa] Young's modulus of the spar
                    'G' : 30.e9,            # [Pa] shear modulus of the spar
                    'yield' : 500.e6 / 2.5, # [Pa] yield stress divided by 2.5 for limiting case
                    'mrho' : 3.e3,          # [kg/m^3] material density
                    'fem_origin' : 0.35,    # normalized chordwise location of the spar
                    't_over_c' : 0.15,      # maximum airfoil thickness
                    'thickness_cp' : np.ones((3)) * .1,

                    'exact_failure_constraint' : False,
                    }

        surf_dict.update({'mesh' : mesh})

        surf_dict['num_x'], surf_dict['num_y'] = surf_dict['mesh'].shape[:2]

        surfaces = [surf_dict]

        # Create the problem and assign the model group
        prob = Problem()

        # Loop over each surface in the surfaces list
        for surface in surfaces:
            ny = surface['num_y']

            indep_var_comp = IndepVarComp()
            indep_var_comp.add_output('loads', val=np.ones(ny) * 2e5)

            struct_group = SpatialBeamAlone(surface=surface)

            # Add indep_vars to the structural group
            struct_group.add_subsystem('indep_vars',
                 indep_var_comp,
                 promotes=['*'])

            prob.model.add_subsystem(surface['name'], struct_group)



            # TODO: add this to the metadata
            # prob.model.add_metadata(surface['name'] + '_yield_stress', surface['yield'])
            # prob.model.add_metadata(surface['name'] + '_fem_origin', surface['fem_origin'])

        from openmdao.api import pyOptSparseDriver
        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = "SNOPT"
        prob.driver.opt_settings = {'Major optimality tolerance': 1.0e-8,
                                    'Major feasibility tolerance': 1.0e-8}

        # Setup problem and add design variables, constraint, and objective
        prob.model.add_design_var('wing.thickness_cp', lower=0.01, upper=0.5, scaler=1e2)
        prob.model.add_constraint('wing.failure', upper=0.)
        prob.model.add_constraint('wing.thickness_intersects', upper=0.)

        # Add design variables, constraisnt, and objective on the problem
        prob.model.add_objective('wing.structural_weight', scaler=1e-4)

        # Set up the problem
        prob.setup()

        view_model(prob, outfile='struct.html', show_browser=False)

        # prob.run_model()
        prob.run_driver()

        self.assertAlmostEqual(prob['wing.structural_weight'][0], 469492.85006454)



    # ## TODO: figure out why this isn't working
    # if fortran_flag:
    #     def test_struct_optimization_symmetry_exact(self):
    #         OAS_prob = OASProblem({'type' : 'struct',
    #                                'optimize' : True,
    #                                'record_db' : False})
    #         OAS_prob.add_surface({'exact_failure_constraint' : True})
    #
    #         OAS_prob.add_design_var('wing.thickness_cp', lower=0.001, upper=0.25, scaler=1e2)
    #         OAS_prob.add_constraint('wing.failure', upper=0.)
    #         OAS_prob.add_constraint('wing.thickness_intersects', upper=0.)
    #         OAS_prob.add_objective('wing.structural_weight', scaler=1e-3)
    #
    #         OAS_prob.setup()
    #
    #         OAS_prob.run()
    #         prob = OAS_prob.prob
    #         self.assertAlmostEqual(prob['wing.structural_weight'][0], 1132.0650209475402, places=2)


class TestAeroStruct(unittest.TestCase):

    currentResult = []

    def run(self, result=None):
        self.currentResult.append(result) # remember result for use in tearDown
        unittest.TestCase.run(self, result) # call superclass run method

#     def test_aerostruct_analysis(self):
#         OAS_prob = OASProblem({'type' : 'aerostruct',
#                                'optimize' : False,
#                                'record_db' : False})
#         surf_dict = {'num_y' : 13,
#                   'num_x' : 2,
#                   'wing_type' : 'CRM',
#                   'CD0' : 0.015,
#                   'symmetry' : False}
#         OAS_prob.add_surface(surf_dict)
#         OAS_prob.setup()
#         OAS_prob.run()
#         prob = OAS_prob.prob
#
#         self.assertAlmostEqual(prob['wing_perf.CL'][0], 0.57198417, places=5)
#         self.assertAlmostEqual(prob['wing_perf.failure'][0], -0.5431833, places=5)
#         self.assertAlmostEqual(prob['fuelburn'][0], 64705.74304495, places=2)
#         self.assertAlmostEqual(prob['CM'][1], -0.14059402501934362, places=2)
#
#     def test_aerostruct_analysis_symmetry(self):
#         OAS_prob = OASProblem({'type' : 'aerostruct',
#                                'optimize' : False,
#                                'record_db' : False})
#         surf_dict = {'symmetry' : True,
#                   'num_y' : 13,
#                   'num_x' : 2,
#                   'wing_type' : 'CRM',
#                   'CD0' : 0.015}
#         OAS_prob.add_surface(surf_dict)
#         OAS_prob.setup()
#         OAS_prob.run()
#         prob = OAS_prob.prob
#
#         self.assertAlmostEqual(prob['wing_perf.CL'][0], 0.60630038, places=5)
#         self.assertAlmostEqual(prob['wing_perf.failure'][0], -0.57587391, places=5)
#         self.assertAlmostEqual(prob['fuelburn'][0], 68028.68895158, places=1)
#         self.assertAlmostEqual(prob['CM'][1], -0.14572267574012124, places=2)
#
#     def test_aerostruct_analysis_symmetry_deriv(self):
#         OAS_prob = OASProblem({'type' : 'aerostruct',
#                                'optimize' : False,
#                                'record_db' : True})
#         surf_dict = {'symmetry' : True,
#                   'num_y' : 7,
#                   'num_x' : 2,
#                   'wing_type' : 'CRM',
#                   'CD0' : 0.015}
#         OAS_prob.add_surface(surf_dict)
#         OAS_prob.setup()
#         OAS_prob.run()
#         prob = OAS_prob.prob
#
#         data = prob.check_partials(out_stream=None)
#
#         new_dict = {}
#         for key1 in data.keys():
#             for key2 in data[key1].keys():
#                 for key3 in data[key1][key2].keys():
#                     if 'rel' in key3:
#                         error = np.linalg.norm(data[key1][key2][key3])
#                         new_key = key1+'_'+key2[0]+'_'+key2[1]+'_'+key3
#                         new_dict.update({new_key : error})
#
#         for key in new_dict.keys():
#             error = new_dict[key]
#             if not np.isnan(error):
#
#                 # The FD check is not valid for these cases
#                 if 'assembly_forces_Iy' in key or 'assembly_forces_J' in key or \
#                 'assembly_forces_A' in key or 'assembly_K_loads' in key or \
#                 'assembly_forces_loads' in key or 'assembly_forces_Iz' in key or \
#                 'assembly_forces_nodes' in key or 'CM_wing_S_ref' in key or \
#                 'CM_rho' or 'CD_wing_S_ref' or 'CL_wing_S_ref' in key:
#                     pass
#                 elif 'K' in key or 'vonmises' in key:
#                     self.assertAlmostEqual(0., error, places=0)
#                 else:
#                     self.assertAlmostEqual(0., error, places=2)

    def test_aerostruct_optimization(self):
        # Create a dictionary to store options about the surface
        mesh_dict = {'num_y' : 5,
                     'num_x' : 2,
                     'wing_type' : 'CRM',
                     'symmetry' : True,
                     'num_twist_cp' : 5}

        mesh, twist_cp = generate_mesh(mesh_dict)

        surf_dict = {
                    # Wing definition
                    'name' : 'wing',        # name of the surface
                    'type' : 'aerostruct',
                    'symmetry' : True,     # if true, model one half of wing
                                            # reflected across the plane y = 0
                    'S_ref_type' : 'wetted', # how we compute the wing area,
                                             # can be 'wetted' or 'projected'

                    'thickness_cp' : np.array([.1, .2, .3]),

                    # Aerodynamic performance of the lifting surface at
                    # an angle of attack of 0 (alpha=0).
                    # These CL0 and CD0 values are added to the CL and CD
                    # obtained from aerodynamic analysis of the surface to get
                    # the total CL and CD.
                    # These CL0 and CD0 values do not vary wrt alpha.
                    'CL0' : 0.0,            # CL of the surface at alpha=0
                    'CD0' : 0.015,            # CD of the surface at alpha=0

                    # Airfoil properties for viscous drag calculation
                    'k_lam' : 0.05,         # percentage of chord with laminar
                                            # flow, used for viscous drag
                    't_over_c' : 0.15,      # thickness over chord ratio (NACA0015)
                    'c_max_t' : .303,       # chordwise location of maximum (NACA0015)
                                            # thickness
                    'with_viscous' : True,

                    # Structural values are based on aluminum 7075
                    'E' : 70.e9,            # [Pa] Young's modulus of the spar
                    'G' : 30.e9,            # [Pa] shear modulus of the spar
                    'yield' : 500.e6 / 2.5, # [Pa] yield stress divided by 2.5 for limiting case
                    'mrho' : 3.e3,          # [kg/m^3] material density
                    'fem_origin' : 0.35,    # normalized chordwise location of the spar

                    # Constraints
                    'exact_failure_constraint' : False, # if false, use KS function
                    }

        surf_dict.update({'twist_cp' : twist_cp,
                          'mesh' : mesh})

        surf_dict['num_x'], surf_dict['num_y'] = surf_dict['mesh'].shape[:2]

        surfaces = [surf_dict]

        # Create the problem and assign the model group
        prob = Problem()

        # Add problem information as an independent variables component
        indep_var_comp = IndepVarComp()
        indep_var_comp.add_output('v', val=248.136)
        indep_var_comp.add_output('alpha', val=5.)
        indep_var_comp.add_output('M', val=0.84)
        indep_var_comp.add_output('re', val=1.e6)
        indep_var_comp.add_output('rho', val=0.38)
        indep_var_comp.add_output('CT', val=9.80665 * 17.e-6)
        indep_var_comp.add_output('R', val=11.165e6)
        indep_var_comp.add_output('W0', val=0.4 * 3e5)
        indep_var_comp.add_output('a', val=295.4)
        indep_var_comp.add_output('load_factor', val=1.)
        indep_var_comp.add_output('empty_cg', val=np.zeros((3)))

        prob.model.add_subsystem('prob_vars',
             indep_var_comp,
             promotes=['*'])

        # Loop over each surface in the surfaces list
        for surface in surfaces:

            # Get the surface name and create a group to contain components
            # only for this surface
            name = surface['name']

            aerostruct_group = Aerostruct(surface=surface)

            # Add tmp_group to the problem with the name of the surface.
            prob.model.add_subsystem(name, aerostruct_group)

        # Loop through and add a certain number of aero points
        for i in range(1):

            point_name = 'AS_point_{}'.format(i)
            # Connect the parameters within the model for each aero point

            # Create the aero point group and add it to the model
            AS_point = AerostructPoint(surfaces=surfaces)

            coupled = AS_point.get_subsystem('coupled')
            prob.model.add_subsystem(point_name, AS_point)

            # Connect flow properties to the analysis point
            prob.model.connect('v', point_name + '.v')
            prob.model.connect('alpha', point_name + '.alpha')
            prob.model.connect('M', point_name + '.M')
            prob.model.connect('re', point_name + '.re')
            prob.model.connect('rho', point_name + '.rho')
            prob.model.connect('CT', point_name + '.CT')
            prob.model.connect('R', point_name + '.R')
            prob.model.connect('W0', point_name + '.W0')
            prob.model.connect('a', point_name + '.a')
            prob.model.connect('empty_cg', point_name + '.empty_cg')
            prob.model.connect('load_factor', point_name + '.load_factor')

            for surface in surfaces:

                com_name = point_name + '.' + name + '_perf'
                prob.model.connect(name + '.K', point_name + '.coupled.' + name + '.K')

                # Connect aerodyamic mesh to coupled group mesh
                prob.model.connect(name + '.mesh', point_name + '.coupled.' + name + '.mesh')

                # Connect performance calculation variables
                prob.model.connect(name + '.radius', com_name + '.radius')
                prob.model.connect(name + '.A', com_name + '.A')
                prob.model.connect(name + '.thickness', com_name + '.thickness')
                prob.model.connect(name + '.nodes', com_name + '.nodes')

        from openmdao.api import pyOptSparseDriver
        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = "SNOPT"
        prob.driver.opt_settings = {'Major optimality tolerance': 1.0e-8,
                                    'Major feasibility tolerance': 1.0e-8}

        # Setup problem and add design variables, constraint, and objective
        prob.model.add_design_var('wing.twist_cp', lower=-10., upper=15.)
        prob.model.add_design_var('wing.thickness_cp', lower=0.01, upper=0.5, scaler=1e2)
        prob.model.add_constraint('AS_point_0.wing_perf.failure', upper=0.)
        prob.model.add_constraint('AS_point_0.wing_perf.thickness_intersects', upper=0.)

        # Add design variables, constraisnt, and objective on the problem
        prob.model.add_design_var('alpha', lower=-10., upper=10.)
        prob.model.add_constraint('AS_point_0.L_equals_W', equals=0.)
        prob.model.add_objective('AS_point_0.fuelburn', scaler=1e-5)

        # Set up the problem
        prob.setup()

        # Save an N2 diagram for the problem
        view_model(prob, outfile='aerostruct.html', show_browser=False)

        # prob.run_model()
        prob.run_driver()

        # prob.check_partials(compact_print=True)

        self.assertAlmostEqual(prob['AS_point_0.fuelburn'][0], 97252.32207548726)

#
    # if fortran_flag:
    #     def test_aerostruct_optimization_symmetry(self):
    #         OAS_prob = OASProblem({'type' : 'aerostruct',
    #                                'optimize' : True,
    #                                'with_viscous' : True,
    #                                'record_db' : False})
    #         surf_dict = {'symmetry' : True,
    #                   'num_y' : 7,
    #                   'num_x' : 3,
    #                   'wing_type' : 'CRM',
    #                   'CD0' : 0.015,
    #                   'num_twist_cp' : 2,
    #                   'num_thickness_cp' : 2}
    #         OAS_prob.add_surface(surf_dict)
    #
    #         OAS_prob.add_design_var('wing.twist_cp', lower=-15., upper=15.)
    #         OAS_prob.add_design_var('wing.thickness_cp', lower=0.01, upper=0.5, scaler=1e2)
    #         OAS_prob.add_constraint('wing_perf.failure', upper=0.)
    #         OAS_prob.add_constraint('wing_perf.thickness_intersects', upper=0.)
    #         OAS_prob.add_design_var('alpha', lower=-10., upper=10.)
    #         OAS_prob.add_constraint('L_equals_W', equals=0.)
    #         OAS_prob.add_objective('fuelburn', scaler=1e-4)
    #
    #         OAS_prob.setup()
    #
    #         OAS_prob.run()
    #         prob = OAS_prob.prob
    #
    #         self.assertAlmostEqual(prob['fuelburn'][0], 96264.09776801, places=0)
    #         self.assertAlmostEqual(prob['wing_perf.failure'][0], 0, places=5)
    #
    # if fortran_flag:
    #     def test_aerostruct_optimization_symmetry_multiple(self):
    #         OAS_prob = OASProblem({'type' : 'aerostruct',
    #                                'optimize' : False,
    #                                'with_viscous' : True,
    #                                'record_db' : True,
    #                                'optimizer' : 'SNOPT'})
    #         surf_dict = {'name' : 'wing',
    #                      'symmetry' : True,
    #                      'num_y' : 5,
    #                      'num_x' : 2,
    #                      'wing_type' : 'CRM',
    #                      'CD0' : 0.015,
    #                      'num_twist_cp' : 2,
    #                      'num_thickness_cp' : 2}
    #         OAS_prob.add_surface(surf_dict)
    #         surf_dict.update({'name' : 'tail',
    #                           'offset':np.array([10., 0., 10.])})
    #         OAS_prob.add_surface(surf_dict)
    #
    #         # Add design variables and constraints for both the wing and tail
    #         OAS_prob.add_design_var('wing.twist_cp', lower=-15., upper=15.)
    #         OAS_prob.add_design_var('wing.thickness_cp', lower=0.01, upper=0.5, scaler=1e2)
    #         OAS_prob.add_constraint('wing_perf.failure', upper=0.)
    #         OAS_prob.add_constraint('wing_perf.thickness_intersects', upper=0.)
    #         OAS_prob.add_design_var('tail.twist_cp', lower=-15., upper=15.)
    #         OAS_prob.add_design_var('tail.thickness_cp', lower=0.01, upper=0.5, scaler=1e2)
    #         OAS_prob.add_constraint('tail_perf.failure', upper=0.)
    #         OAS_prob.add_constraint('tail_perf.thickness_intersects', upper=0.)
    #
    #         OAS_prob.add_design_var('alpha', lower=-10., upper=10.)
    #         OAS_prob.add_constraint('L_equals_W', equals=0.)
    #         OAS_prob.add_objective('fuelburn', scaler=1e-5)
    #
    #         OAS_prob.setup()
    #
    #         OAS_prob.run()
    #         prob = OAS_prob.prob
    #
    #         self.assertAlmostEqual(prob['fuelburn'][0], 156050.56515008, places=1)
    #         self.assertAlmostEqual(prob['wing_perf.failure'][0], -0.08722559, places=5)


if __name__ == "__main__":

    # Get user-supplied argument if provided
    try:
        arg = sys.argv[1]
        arg_provided = True
    except:
        arg_provided = False

    # Based on user input, run one subgroup of tests
    if arg_provided:
        if 'aero' == arg:
            test_classes = [TestAero]
        elif 'struct' == arg:
            test_classes = [TestStruct]
        elif 'aerostruct' == arg:
            test_classes = [TestAeroStruct]
    else:
        arg = 'full'
        test_classes = [TestAero, TestStruct, TestAeroStruct]

    print()
    print('+==================================================+')
    print('             Running ' + arg + ' test suite')
    print('+==================================================+')
    print()

    failures = []
    errors = []
    num_tests = 0

    # Loop through each requested discipline test
    for test_class in test_classes:

        # Set up the test suite and run the tests corresponding to this subgroup
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        unittest.TextTestRunner().run(suite)

        failures.extend(test_class.currentResult[-1].failures)
        errors.extend(test_class.currentResult[-1].errors)
        num_tests += len(test_class.currentResult)

    # Print results and force an exit if an error or failure occurred
    print()
    if len(failures) or len(errors):
        print("There have been errors or failures! Please check the log to " +
              "see which tests failed.\n")
        sys.exit(1)

    else:
        print("Successfully ran {} tests with no errors!\n".format(num_tests))
