from __future__ import division
import numpy
import sys
import time

from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from openmdao.api import IndepVarComp, Problem, Group, ScipyOptimizer, Newton, ScipyGMRES, LinearGaussSeidel, NLGaussSeidel, SqliteRecorder, profile
from geometry import GeometryMesh, Bspline, gen_crm_mesh, gen_rect_mesh
from transfer import TransferDisplacements, TransferLoads
from vlm import VLMStates, VLMFunctionals, VLMGeometry
from spatialbeam import SpatialBeamStates, SpatialBeamFunctionals, radii
from materials import MaterialsTube
from functionals import FunctionalBreguetRange, FunctionalEquilibrium

from openmdao.api import view_model
from run_classes import OASProblem
from gs_newton import HybridGSNewton
from b_spline import get_bspline_mtx

try:
    input_arg = sys.argv[1]
except IndexError:
    print '\n +--------------------------------------------------------+'
    print   ' | ERROR: Please supply an input argument to this script. |'
    print   ' | Example: Run `python problems.py prob1`                |'
    print   ' +--------------------------------------------------------+\n'
    raise

if input_arg == 'prob1':

    # Set problem type
    prob_dict = {'type' : 'struct'}

    # Instantiate problem and add default surface
    OAS_prob = OASProblem(prob_dict)
    OAS_prob.add_surface({'name' : 'wing',
                          'num_y' : 13,
                          'span_cos_spacing' : 0,
                          'symmetry' : True})

    # Get the created surface
    surface = OAS_prob.surfaces[0]

    num_y = surface['num_y']

    r = radii(surface['mesh'])
    thickness = r / 10

    # Define the loads
    loads = numpy.zeros((num_y, 6))
    loads[0, 2] = loads[-1, 2] = 1e3 # tip load of 1 kN
    # loads[:, 2] = 1e3 # load of 1 kN at each node

    root = Group()

    des_vars = [
        ('twist', numpy.zeros(num_y)),
        ('span', surface['span']),
        ('r', r),
        ('thickness', thickness),
        ('loads', loads)
    ]

    root.add('des_vars',
             IndepVarComp(des_vars),
             promotes=['*'])
    root.add('mesh',
             GeometryMesh(surface),
             promotes=['*'])
    root.add('tube',
             MaterialsTube(surface),
             promotes=['*'])
    root.add('spatialbeamstates',
             SpatialBeamStates(surface),
             promotes=['*'])
    root.add('spatialbeamfuncs',
             SpatialBeamFunctionals(surface),
             promotes=['*'])

    prob = Problem()
    prob.root = root


    prob.driver = ScipyOptimizer()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['disp'] = True
    # prob.driver.options['tol'] = 1.0e-12

    prob.driver.add_desvar('thickness', lower=0.001, upper=0.75, scaler=1e2)
    prob.driver.add_objective('weight', scaler=1e-3)
    prob.driver.add_constraint('failure', upper=0., scaler=1e-4)

    prob.root.deriv_options['type'] = 'fd'
    #
    prob.root.deriv_options['form'] = 'central'
    prob.root.deriv_options['step_size'] = 1e-10

    prob.driver.add_recorder(SqliteRecorder('spatialbeam.db'))

    prob.setup()

    view_model(prob, outfile="prob1.html", show_browser=False)

    # prob.run_once()
    # prob.check_partial_derivatives(compact_print=True)

    st = time.time()
    prob.run()
    print "run time: {} secs".format(time.time() - st)
    print 'thickness distribution:', prob['thickness']


elif 'prob2' in input_arg or 'prob3' in input_arg:

    # Set problem type
    prob_dict = {'type' : 'aerostruct'}

    # Instantiate problem and add default surface
    OAS_prob = OASProblem(prob_dict)
    OAS_prob.add_surface({'name' : '',
                          'wing_type' : 'CRM',
                          'num_y' : 9,
                          'num_x' : 2,
                          'span_cos_spacing' : 0,
                          'CL0' : 0.2,
                          'CD0' : 0.015,
                          'symmetry' : True})

    # Get the created surface
    surface = OAS_prob.surfaces[0]
    prob_dict = OAS_prob.prob_dict
    num_y = surface['num_y']
    r = radii(surface['mesh'])
    thickness = r / 10
    thickness[:] = numpy.max((thickness))
    num_twist = num_thickness = num_y

    span = surface['span']
    v = prob_dict['v']
    alpha = prob_dict['alpha']
    rho = prob_dict['rho']
    M = prob_dict['M']
    Re = prob_dict['Re']

    # Create the top-level system
    root = Group()

    # Define the independent variables
    indep_vars = [
        ('span', span),
        ('twist', numpy.zeros(num_twist)),
        ('thickness', thickness),
        ('v', v),
        ('alpha', alpha),
        ('rho', rho),
        ('r', r),
        ('M', M),
        ('Re', Re),
    ]

    ############################################################
    # Problem 2a:
    # These are your components, put them in the correct groups.
    # indep_vars_comp, tube_comp, and weiss_func_comp have been
    # done for you as examples
    ############################################################

    indep_vars_comp = IndepVarComp(indep_vars)
    tube_comp = MaterialsTube(surface)

    mesh_comp = GeometryMesh(surface)
    geom_comp = VLMGeometry(surface)
    spatialbeamstates_comp = SpatialBeamStates(surface)
    def_mesh_comp = TransferDisplacements(surface)
    vlmstates_comp = VLMStates(OAS_prob.surfaces, OAS_prob.prob_dict)
    loads_comp = TransferLoads(surface)

    vlmfuncs_comp = VLMFunctionals(surface)
    spatialbeamfuncs_comp = SpatialBeamFunctionals(surface)
    fuelburn_comp = FunctionalBreguetRange(OAS_prob.surfaces, OAS_prob.prob_dict)
    eq_con_comp = FunctionalEquilibrium(OAS_prob.surfaces, OAS_prob.prob_dict)

    ############################################################
    ############################################################

    root.add('indep_vars',
             indep_vars_comp,
             promotes=['*'])
    root.add('tube',
             tube_comp,
             promotes=['*'])

    # Add components to the MDA here
    coupled = Group()
    coupled.add('mesh',
        mesh_comp,
        promotes=["*"])
    coupled.add('spatialbeamstates',
        spatialbeamstates_comp,
        promotes=["*"])
    coupled.add('def_mesh',
        def_mesh_comp,
        promotes=["*"])
    coupled.add('geom',
        geom_comp,
        promotes=["*"])
    coupled.add('vlmstates',
        vlmstates_comp,
        promotes=["*"])
    coupled.add('loads',
        loads_comp,
        promotes=["*"])

    ############################################################
    # Problem 2b:
    # Comment/uncomment these solver blocks to try different
    # nonlinear solver methods
    ############################################################

    ## Nonlinear Gauss Seidel on the coupled group
    coupled.nl_solver = NLGaussSeidel()
    coupled.nl_solver.options['iprint'] = 1
    coupled.nl_solver.options['atol'] = 1e-5
    coupled.nl_solver.options['rtol'] = 1e-12

    ## Newton Solver on the coupled group
    # coupled.nl_solver = Newton()
    # coupled.nl_solver.options['iprint'] = 1


    ## Hybrid NLGS-Newton on the coupled group
    # coupled.nl_solver = HybridGSNewton()
    # coupled.nl_solver.nlgs.options['iprint'] = 1
    # coupled.nl_solver.nlgs.options['maxiter'] = 5
    # coupled.nl_solver.newton.options['atol'] = 1e-7
    # coupled.nl_solver.newton.options['rtol'] = 1e-7
    # coupled.nl_solver.newton.options['iprint'] = 1

    # Newton solver on the root group
    # root.nl_solver = Newton()
    # root.nl_solver.options['iprint'] = 1

    ############################################################
    # Problem 2c:
    # Comment/uncomment these solver blocks to try different
    # linear solvers
    ############################################################

    ## Linear Gauss Seidel Solver
    # coupled.ln_solver = LinearGaussSeidel()
    # coupled.ln_solver.options['maxiter'] = 100

    ## Krylov Solver - No preconditioning
    # coupled.ln_solver = ScipyGMRES()
    # coupled.ln_solver.options['iprint'] = 1

    ## Krylov Solver - LNGS preconditioning
    coupled.ln_solver = ScipyGMRES()
    coupled.ln_solver.options['iprint'] = 1
    coupled.ln_solver.preconditioner = LinearGaussSeidel()
    coupled.vlmstates.ln_solver = LinearGaussSeidel()
    coupled.spatialbeamstates.ln_solver = LinearGaussSeidel()

    # adds the MDA to root (do not remove!)
    root.add('coupled',
             coupled,
             promotes=['*'])

    # Add functional components here
    root.add('vlmfuncs',
            vlmfuncs_comp,
            promotes=['*'])
    root.add('spatialbeamfuncs',
            spatialbeamfuncs_comp,
            promotes=['*'])
    root.add('fuelburn',
            fuelburn_comp,
            promotes=['*'])
    root.add('eq_con',
            eq_con_comp,
            promotes=['*'])

    prob = Problem()
    prob.root = root

    # change file name to save data from each experiment separately
    prob.driver.add_recorder(SqliteRecorder('aerostruct.db'))

    #############################################################
    # Problem 3b:
    # Comment out the following code to run analytic derivatives
    ##############################################################
    # prob.root.deriv_options['type'] = 'fd'
    # prob.root.deriv_options['step_type'] = 'relative'
    # prob.root.deriv_options['form'] = 'forward'
    # prob.root.deriv_options['step_size'] = 1e-6
    #####################################################

    prob.setup()
    prob.print_all_convergence() # makes OpenMDAO print out solver convergence data

    # uncomment this to see an n2 diagram of your problem
    view_model(prob, outfile="aerostruct_n2.html", show_browser=False)

    st = time.time()
    prob.run_once()

    if 'prob3a' in input_arg:

        print "------------------------------------------------"
        print "Solving for Derivatives"
        print "------------------------------------------------"
        st = time.time()
        profile.setup(prob)
        profile.start()
        jac = prob.calc_gradient(['twist','alpha','thickness'], ['fuelburn', 'vonmises'], return_format="dict")
        run_time = time.time() - st
        profile.stop()

        print "d_fuelburn/d_alpha", jac['fuelburn']['alpha']
        print "norm(d_fuelburn/d_twist)", numpy.linalg.norm(jac['fuelburn']['twist'])
        print "norm(d_fuelburn/d_thickness)", numpy.linalg.norm(jac['fuelburn']['thickness'])

    # Uncomment this to print partial derivatives accuracy information
    prob.check_partial_derivatives(compact_print=True)

    if 'prob3c' in input_arg:

        prob.driver = ScipyOptimizer()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['disp'] = True
        prob.driver.options['tol'] = 1.0e-5
        prob.driver.options['maxiter'] = 40

        prob.driver.add_recorder(SqliteRecorder('aerostruct.db'))

        ###############################################################
        # Add design vars
        ###############################################################
        prob.driver.add_desvar('twist',lower= -10.,
                               upper=10., scaler=1e0)
        prob.driver.add_desvar('alpha', lower=-10., upper=10., scaler=1e0)
        prob.driver.add_desvar('thickness',
                               lower= 0.003,
                               upper= 0.25, scaler=1000)

        ###############################################################
        # Add constraints, and objectives
        ###############################################################
        prob.driver.add_objective('fuelburn')
        prob.driver.add_constraint('failure', upper=0.0)
        prob.driver.add_constraint('eq_con', equals=0.0)


    prob.setup()
    # view_model(prob, outfile="my_aerostruct_n2.html", show_browser=True) # generate the n2 diagram diagram

    st = time.time()

    # Actually run the optimization
    prob.run()

    print "run time: {} secs".format(time.time() - st)
    print "fuelburn:", prob['fuelburn']
