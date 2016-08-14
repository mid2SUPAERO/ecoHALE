from __future__ import division
import sys
from time import time
import numpy
import unittest

from openmdao.api import IndepVarComp, Problem, Group, ScipyOptimizer, Newton, ScipyGMRES, LinearGaussSeidel, NLGaussSeidel, SqliteRecorder, profile
from openmdao.devtools.partition_tree_n2 import view_tree
from geometry import GeometryMesh, Bspline, gen_crm_mesh, gen_mesh, get_inds
from transfer import TransferDisplacements, TransferLoads
from vlm import VLMStates, VLMFunctionals
from spatialbeam import SpatialBeamStates, SpatialBeamFunctionals, radii
from materials import MaterialsTube
from functionals import FunctionalBreguetRange, FunctionalEquilibrium

from gs_newton import HybridGSNewton
from b_spline import get_bspline_mtx

class TestVLM(unittest.TestCase):

    # Solve some simple VLM problems

    def get_default_dict(self):
        defaults = {'num_x' : 3,
                    'num_y' : 5,
                    'span' : 10.,
                    'chord' : 1.,
                    'cosine_spacing' : 1,
                    'dihedral' : 0.,
                    'sweep' : 0.,
                    'taper' : 1.,
                    'Re' : 0.,
                    'alpha' : 5.,
                    'optimize' : False,
                    'W0' : 0.5 * 2.5e6, # [N] (MTOW of B777 is 3e5 kg with fuel)
                    'CT' : 9.81 * 17.e-6, # [1/s] (9.81 N/kg * 17e-6 kg/N/s)
                    'R' : 14.3e6, # [m] maximum range
                    'M' : 0.84, # at cruise
                    'rho' : 0.38, # [kg/m^3] at 35,000 ft
                    'a' : 295.4, # [m/s] at 35,000 ft
                    'CL0' : 0.2,
                    'CD0' : 0.015,
                    'symmetry' : False
                    }
        return defaults

    def run_aero_case(self, input_dict={}):

        v_dict = self.get_default_dict()
        v_dict.update(input_dict)

        for name in v_dict.keys():
            exec(name + ' = v_dict[ "' + name + '"]')

        v = a * M

        mesh = gen_mesh(num_x, num_y, span, chord, cosine_spacing)

        if symmetry:
            num_y = int((num_y+1)/2)
            mesh = mesh[:, :num_y, :]

        num_twist = numpy.max([int((num_y - 1) / 5), 5])

        mesh = mesh.reshape(-1, mesh.shape[-1])
        aero_ind = numpy.atleast_2d(numpy.array([num_x, num_y]))
        fem_ind = [num_y]

        # Compute the aero and fem indices
        aero_ind, fem_ind = get_inds(aero_ind, fem_ind)

        # Create the top-level system
        root = Group()

        # Define Jacobians for b-spline controls
        tot_n_fem = numpy.sum(fem_ind[:, 0])
        jac = get_bspline_mtx(num_twist, num_y)

        # Define the independent variables
        des_vars = [
            ('twist_cp', numpy.zeros(num_twist)),
            ('dihedral', dihedral),
            ('sweep', sweep),
            ('span', span),
            ('taper', taper),
            ('v', v),
            ('alpha', alpha),
            ('rho', rho),
            ('disp', numpy.zeros((tot_n_fem, 6))),
            ('aero_ind', aero_ind),
            ('fem_ind', fem_ind),
            ('Re', Re)
        ]

        # Add VLM components to the top-level system
        root.add('des_vars',
                 IndepVarComp(des_vars),
                 promotes=['*'])
        root.add('twist_bsp',
                 Bspline('twist_cp', 'twist', jac),
                 promotes=['*'])
        root.add('mesh',
                 GeometryMesh(mesh, aero_ind),
                 promotes=['*'])
        root.add('def_mesh',
                 TransferDisplacements(aero_ind, fem_ind),
                 promotes=['*'])
        root.add('vlmstates',
                 VLMStates(aero_ind, symmetry),
                 promotes=['*'])
        root.add('vlmfuncs',
                 VLMFunctionals(aero_ind, CL0, CD0),
                 promotes=['*'])

        # Set the optimization problem settings
        prob = Problem()
        prob.root = root

        try:  # Use SNOPT optimizer if installed
            from openmdao.api import pyOptSparseDriver
            prob.driver = pyOptSparseDriver()
            prob.driver.options['optimizer'] = "SNOPT"
            prob.driver.opt_settings = {'Major optimality tolerance': 1.0e-8,
                                        'Major feasibility tolerance': 1.0e-8}
        except:  # Use SLSQP optimizer if SNOPT not installed
            prob.driver = ScipyOptimizer()
            prob.driver.options['optimizer'] = 'SLSQP'
            prob.driver.options['disp'] = True
            prob.driver.options['tol'] = 1.0e-8

        # Add design variables for the optimizer to control
        # Note that the scaling is very important to get correct convergence
        prob.driver.add_desvar('twist_cp', lower=-10., upper=15., scaler=1e0)
        # prob.driver.add_desvar('alpha', lower=-10., upper=10.)
        prob.driver.add_desvar('sweep', lower=-10., upper=30.)
        prob.driver.add_desvar('dihedral', lower=-10., upper=20.)
        prob.driver.add_desvar('taper', lower=.5, upper=2.)

        # Set the objective (minimize CD on the main wing)
        prob.driver.add_objective('CD_wing', scaler=1e4)

        # Set the constraint (CL = 0.5 for the main wing)
        prob.driver.add_constraint('CL_wing', equals=0.5)

        # Record optimization history to a database
        # Data saved here can be examined using `plot_all.py`
        prob.driver.add_recorder(SqliteRecorder('test.db'))

        # Can finite difference over the entire model
        # Generally faster than using component derivatives
        prob.root.deriv_options['type'] = 'fd'

        # Setup the problem
        prob.setup()

        prob.run_once()
        if not optimize:  # run analysis once
            pass
        else:  # perform optimization
            prob.run()
        return prob

    def run_aerostruct_case(self, input_dict={}):

        v_dict = self.get_default_dict()
        v_dict.update(input_dict)

        for name in v_dict.keys():
            exec(name + ' = v_dict[ "' + name + '"]')

        v = a * M

        npi = int(((num_y - 1) / 2) * .6)
        npo = int(npi * 5 / 3)
        mesh = gen_crm_mesh(n_points_inboard=npi, n_points_outboard=npo, num_x=num_x)
        num_x, num_y = mesh.shape[:2]

        if symmetry:
            num_y = int((num_y+1)/2)
            mesh = mesh[:, :num_y, :]

        num_twist = numpy.max([int((num_y - 1) / 5), 5])

        r = radii(mesh)
        mesh = mesh.reshape(-1, mesh.shape[-1])
        aero_ind = numpy.atleast_2d(numpy.array([num_x, num_y]))
        fem_ind = [num_y]
        aero_ind, fem_ind = get_inds(aero_ind, fem_ind)

        # Set the number of thickness control points and the initial thicknesses
        num_thickness = num_twist
        t = r / 10

        if symmetry:
            W0 /= 2.

        # Define the material properties
        execfile('aluminum.py')

        # Create the top-level system
        root = Group()

        # Define Jacobians for b-spline controls
        tot_n_fem = numpy.sum(fem_ind[:, 0])
        num_surf = fem_ind.shape[0]
        jac_twist = get_bspline_mtx(num_twist, num_y)
        jac_thickness = get_bspline_mtx(num_thickness, tot_n_fem-num_surf)

        # Define the independent variables
        indep_vars = [
            ('span', span),
            ('twist_cp', numpy.zeros(num_twist)),
            ('thickness_cp', numpy.ones(num_thickness)*numpy.max(t)),
            ('v', v),
            ('alpha', alpha),
            ('rho', rho),
            ('r', r),
            ('Re', 0.),  # set Re=0 if you don't want skin friction drag added
            ('M', M),
            ('aero_ind', aero_ind),
            ('fem_ind', fem_ind)
        ]

        # Add material components to the top-level system
        root.add('indep_vars',
                 IndepVarComp(indep_vars),
                 promotes=['*'])
        root.add('twist_bsp',
                 Bspline('twist_cp', 'twist', jac_twist),
                 promotes=['*'])
        root.add('thickness_bsp',
                 Bspline('thickness_cp', 'thickness', jac_thickness),
                 promotes=['*'])
        root.add('tube',
                 MaterialsTube(fem_ind),
                 promotes=['*'])

        # Create a coupled group to contain the aero, sruct, and transfer components
        coupled = Group()
        coupled.add('mesh',
                    GeometryMesh(mesh, aero_ind),
                    promotes=['*'])
        coupled.add('def_mesh',
                    TransferDisplacements(aero_ind, fem_ind),
                    promotes=['*'])
        coupled.add('vlmstates',
                    VLMStates(aero_ind, symmetry),
                    promotes=['*'])
        coupled.add('loads',
                    TransferLoads(aero_ind, fem_ind),
                    promotes=['*'])
        coupled.add('spatialbeamstates',
                    SpatialBeamStates(aero_ind, fem_ind, E, G),
                    promotes=['*'])

        # Set solver properties
        coupled.ln_solver = ScipyGMRES()
        coupled.ln_solver.preconditioner = LinearGaussSeidel()
        coupled.vlmstates.ln_solver = LinearGaussSeidel()
        coupled.spatialbeamstates.ln_solver = LinearGaussSeidel()

        coupled.nl_solver = HybridGSNewton()   ### Uncomment this out to use Hybrid GS Newton
        coupled.nl_solver.nlgs.options['maxiter'] = 10
        coupled.nl_solver.nlgs.options['atol'] = 1e-8
        coupled.nl_solver.nlgs.options['rtol'] = 1e-12
        coupled.nl_solver.newton.options['atol'] = 1e-7
        coupled.nl_solver.newton.options['rtol'] = 1e-7
        coupled.nl_solver.newton.options['maxiter'] = 5

        # Add the coupled group and functional groups to compute performance
        root.add('coupled',
                 coupled,
                 promotes=['*'])
        root.add('vlmfuncs',
                 VLMFunctionals(aero_ind, CL0, CD0),
                 promotes=['*'])
        root.add('spatialbeamfuncs',
                 SpatialBeamFunctionals(aero_ind, fem_ind, E, G, stress, mrho),
                 promotes=['*'])
        root.add('fuelburn',
                 FunctionalBreguetRange(W0, CT, a, R, M, aero_ind),
                 promotes=['*'])
        root.add('eq_con',
                 FunctionalEquilibrium(W0, aero_ind),
                 promotes=['*'])

        # Set the optimization problem settings
        prob = Problem()
        prob.root = root
        # prob.print_all_convergence()

        try:  # Use SNOPT optimizer if installed
            from openmdao.api import pyOptSparseDriver
            prob.driver = pyOptSparseDriver()
            prob.driver.options['optimizer'] = "SNOPT"
            prob.driver.opt_settings = {'Major optimality tolerance': 1.0e-7,
                                        'Major feasibility tolerance': 1.0e-7}
        except:  # Use SLSQP optimizer if SNOPT not installed
            prob.driver = ScipyOptimizer()
            prob.driver.options['optimizer'] = 'SLSQP'
            prob.driver.options['disp'] = True
            prob.driver.options['tol'] = 1.0e-8

        # Add design variables for the optimizer to control
        # Note that the scaling is very important to get correct convergence
        prob.driver.add_desvar('twist_cp',lower= -15.,
                               upper=15., scaler=1e0)
        prob.driver.add_desvar('alpha', lower=-10., upper=10., scaler=1e0)
        prob.driver.add_desvar('thickness_cp',
                               lower= 0.01,
                               upper= 0.25, scaler=1e2)

        # Set the objective (minimize fuelburn)
        prob.driver.add_objective('fuelburn', scaler=1e-5)

        # Set the constraints (no structural failure and lift = weight)
        prob.driver.add_constraint('failure', upper=0.0)
        prob.driver.add_constraint('eq_con', equals=0.0)

        # Record optimization history to a database
        # Data saved here can be examined using `plot_all.py`
        prob.driver.add_recorder(SqliteRecorder('test_as.db'))

        # Set up the problem
        prob.setup()

        prob.run_once()

        if not optimize:  # run analysis once
            pass
        else:  # perform optimization
            prob.run()
        return prob

    def test_aero_analysis_flat(self):
        prob = self.run_aero_case()  # Match the CL
        self.assertAlmostEqual(prob['CL'], .65655138)

    def test_aero_analysis_flat_symmetry(self):
        v_dict = {'symmetry' : True}
        prob = self.run_aero_case(v_dict)  # Match the CL
        self.assertAlmostEqual(prob['CL'], .65655138)

    def test_aero_optimization_flat(self):
        v_dict = {'optimize' : True}
        prob = self.run_aero_case(v_dict)  # Match the objective value
        self.assertAlmostEqual(prob['CD_wing'], .0314570988)

    def test_aerostruct_analysis(self):
        v_dict = {'num_y' : 13,
                  'num_x' : 2}
        prob = self.run_aerostruct_case(v_dict)  # Match the CL and failure values
        self.assertAlmostEqual(prob['CL'], .58245256)
        self.assertAlmostEqual(prob['failure'], -.431801158)

    def test_aerostruct_analysis_symmetry(self):
        v_dict = {'symmetry' : True,
                  'num_y' : 13,
                  'num_x' : 2}
        prob = self.run_aerostruct_case(v_dict)  # Match the CL and failure values
        self.assertAlmostEqual(prob['CL'], .58245256)
        self.assertAlmostEqual(prob['failure'], -.5011158763)

    def test_aerostruct_optimization(self):
        v_dict = {'optimize' : True,
                  'num_y' : 13,
                  'num_x' : 2}
        prob = self.run_aerostruct_case(v_dict)  # Match the objective and failure values
        self.assertAlmostEqual(prob['fuelburn'], 922881.031535157)
        self.assertAlmostEqual(prob['failure'], 1e-9)

    def test_aerostruct_optimization_symmetry(self):
        v_dict = {'optimize' : True,
                  'symmetry' : True,
                  'num_y' : 13,
                  'num_x' : 2}
        prob = self.run_aerostruct_case(v_dict)  # Match the objective and failure values
        self.assertAlmostEqual(prob['fuelburn'], 427046.16000037867)
        self.assertAlmostEqual(prob['failure'], 1e-9)

if __name__ == "__main__":
    print
    print '===================================================='
    print '|                                                  |'
    print '|             RUNNING FULL TEST SUITE              |'
    print '|                                                  |'
    print '===================================================='
    print
    unittest.main()
