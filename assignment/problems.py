"""
This is the main script you need to change for the assignment.

Call it as `python problems.py <-prob_name->` where <-prob_name-> is one of:

prob1
prob2
prob3ab
prob3c

"""

# Base imports
from __future__ import division, print_function
import numpy
import sys
import time

# Append the parent directory to the system path so we can call those
# Python files
from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

# Import OpenMDAO methods
from openmdao.api import IndepVarComp, Problem, Group, ScipyOptimizer, Newton, ScipyGMRES, LinearGaussSeidel, NLGaussSeidel, SqliteRecorder, profile, view_model

# Import OpenAeroStruct methods
from geometry import GeometryMesh, Bspline, gen_crm_mesh, gen_rect_mesh
from transfer import TransferDisplacements, TransferLoads
from vlm import VLMStates, VLMFunctionals, VLMGeometry
from spatialbeam import SpatialBeamStates, SpatialBeamFunctionals, radii
from materials import MaterialsTube
from functionals import FunctionalBreguetRange, FunctionalEquilibrium
from run_classes import OASProblem
from gs_newton import HybridGSNewton
from b_spline import get_bspline_mtx

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Parse the user-supplied command-line input and store it as input_arg
try:
    input_arg = sys.argv[1]
except IndexError:
    print('\n +--------------------------------------------------------+')
    print(' | ERROR: Please supply an input argument to this script. |')
    print(' | Example: Run `python problems.py prob1`                |')
    print(' +--------------------------------------------------------+\n')
    raise

# Make sure that the user-supplied input is one of the valid options
input_options = ['prob1', 'prob2', 'prob3ab', 'prob3c']
print_str = ''.join(str(e) + ', ' for e in input_options)
if input_arg not in input_options:
    print('\n +---------------------------------------------------------------+')
    print(' | ERROR: Please supply a correct input argument to this script. |')
    print(' | Possible options are ' + print_str[:-2] + '            |')
    print(' +---------------------------------------------------------------+\n')
    raise

# Check input_arg and run prob1 if selected
if input_arg == 'prob1':

    # Set problem type. Any option not set here will be set in the method
    # `get_default_prob_dict` within `run_classes.py`
    prob_dict = {'type' : 'struct'}

    # Instantiate OpenAeroStruct (OAS) problem
    OAS_prob = OASProblem(prob_dict)

    # We specify the necessary parameters here.
    # Defaults are set in `get_default_surf_dict` within `run_classes.py` and
    # then are overwritten with the settings here.
    surface = {'name' : 'wing',        # name of the surface
               'num_x' : 2,            # number of chordwise points
               'num_y' : 13,            # number of spanwise points
               'span' : 10.,           # full wingspan
               'chord' : 1.,           # root chord
               'span_cos_spacing' : 0,   # 0 for uniform spanwise panels
                                        # 1 for cosine-spaced panels
                                        # any value between 0 and 1 for
                                        # a mixed spacing

                # Structural values are based on aluminum
                'E' : 70.e9,            # [Pa] Young's modulus of the spar
                'G' : 30.e9,            # [Pa] shear modulus of the spar
                'stress' : 20.e6,       # [Pa] yield stress
                'mrho' : 3.e3,          # [kg/m^3] material density
                'symmetry' : True,      # if true, model one half of wing
                                        # reflected across the plane y = 0
                'W0' : 0.5 * 2.5e6,     # [N] MTOW of B777 is 3e5 kg with fuel
                }
    # Add our defined surface to the OAS_prob object.
    OAS_prob.add_surface(surface)

    # Get the finalized surface, which includes the created mesh object.
    # Here, `surface` is a dictionary that contains information relevant to
    # one surface within the analysis or optimization.
    surface = OAS_prob.surfaces[0]

    # If you want to view the information contained within `surface`,
    # uncomment the following two lines of code.
    # for key in surface.keys():
    #     print(key, surface[key])

    # Obtain the number of spanwise node points from the defined surface.
    num_y = surface['num_y']

    # Create an array of radii for the spar elements.
    r = radii(surface['mesh'])

    # Obtain the starting thickness for each of the spar elements based
    # on the radii.
    thickness = r / 10

    # Define the loads here. Choose either a tip load or distributed load
    # by commenting the lines as necessary.
    loads = numpy.zeros((num_y, 6))
    P = 1e4  # load of 10 kN
    # loads[0, 2] = P  # tip load
    loads[1:, 2] = P / (num_y - 1)  # load distributed across all nodes

    # Instantiate the OpenMDAO group for the root problem.
    root = Group()

    # Create a list of tuples that contains the design variables.
    # These will be used in the analysis and optimization and will connect
    # to variables within the components.
    # For example, here we set the loads, and SpatialBeamStates computes
    # the displacements based off of these loads.
    des_vars = [
        ('twist', numpy.zeros(surface['num_y'])),
        ('span', surface['span']),
        ('r', r),
        ('thickness', thickness),
        ('loads', loads)
    ]

    # Add components to the root problem. Note that each of the components
    # is defined with `promotes=['*']`, which means that all parameters
    # within that component are promoted to the root level so that all
    # other components can access them. For example, GeometryMesh creates
    # the mesh, and SpatialBeamStates uses this mesh information.
    # The data passing happens behind-the-scenes in OpenMDAO.
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

    # Instantiate an OpenMDAO problem and all the root group that we just
    # created to the problem.
    prob = Problem()
    prob.root = root

    # Set a driver for the optimization problem. Without a driver, OpenMDAO
    # doesn't know how to change the parameters to achieve an optimal solution.
    # There are a few options, but ScipyOptimizer and SLSQP are generally
    # the best options to use without installing additional packages.
    prob.driver = ScipyOptimizer()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['disp'] = True

    # Add design variables, objective, and a constraint.
    # We set the design variable that the optimizer can control.
    # Note the lower and upper bounds. We also scale this design variable
    # so that it's on the same order of magnitude as the objective
    # and constraints.
    prob.driver.add_desvar('thickness', lower=0.001, upper=0.25, scaler=1e2)
    # Again, note the scaler on the weight measure.
    prob.driver.add_objective('weight', scaler=1e-3)
    # Add a failure constraint that none of the beam elements can go past
    # yield stress. Note that we use a Kreisselmeier-Steinhauser (KS)
    # function to aggregate the constraints into a single constraint.
    # Because this is a conservative aggregation, some of the structural
    # elements may be well below the failure limit instead of right at it.
    prob.driver.add_constraint('failure', upper=0.)

    # Simply use finite-differencing over the entire model to get the
    # derivatives used for optimization.
    prob.root.deriv_options['type'] = 'fd'

    # Record the optimization history in `spatialbeam.db`. You can view
    # this by running `python plot_all.py s` or `python OptView.py s`.
    recorder = SqliteRecorder('spatialbeam.db')
    recorder.options['record_params'] = True
    recorder.options['record_derivs'] = True
    prob.driver.add_recorder(recorder)

    # Have OpenMDAO set up the problem that we have constructed.
    prob.setup()

    # Create an html output file showing the problem formulation and data
    # passing with an interactive chart. Open this in any web browser.
    view_model(prob, outfile="prob1.html", show_browser=False)

    # Start timing and perform the optimization.
    st = time.time()
    prob.run()
    print("\nrun time: {} secs".format(time.time() - st))
    print('thickness distribution:', prob['thickness'], "\n")

    # Uncomment the following line to check the partial derivatives of each
    # component and view their accuracy.
    # prob.check_partial_derivatives(compact_print=True)

elif 'prob2' in input_arg or 'prob3' in input_arg:

    # Set problem type. Any option not set here will be set in the method
    # `get_default_prob_dict` within `run_classes.py`
    prob_dict = {'type' : 'aerostruct'}

    # Instantiate OpenAeroStruct (OAS) problem
    OAS_prob = OASProblem(prob_dict)

    # We specify the necessary parameters here.
    # Defaults are set in `get_default_surf_dict` within `run_classes.py` and
    # then are overwritten with the settings here.
    surface  = {'name' : '',        # name of the surface
                'num_x' : 2,            # number of chordwise points
                'num_y' : 9,            # number of spanwise points; must be odd
                                        # for the CRM case, this is an approximation
                                        # of the number of spanwise points;
                                        # it may not produce a mesh with the
                                        # exact requested value
                'span_cos_spacing' : 0,   # 0 for uniform spanwise panels
                                        # 1 for cosine-spaced panels
                                        # any value between 0 and 1 for
                                        # a mixed spacing
                'CL0' : 0.2,            # CL value at AoA (alpha) = 0
                'CD0' : 0.015,            # CD value at AoA (alpha) = 0

                # Structural values are based on aluminum
                'E' : 70.e9,            # [Pa] Young's modulus of the spar
                'G' : 30.e9,            # [Pa] shear modulus of the spar
                'stress' : 20.e6,       # [Pa] yield stress
                'mrho' : 3.e3,          # [kg/m^3] material density
                'fem_origin' : 0.35,    # chordwise location of the spar
                'symmetry' : True,     # if true, model one half of wing
                                        # reflected across the plane y = 0
                'W0' : 0.5 * 2.5e6,     # [N] MTOW of B777 is 3e5 kg with fuel
                'wing_type' : 'CRM',   # initial shape of the wing
                                        # either 'CRM' or 'rect'
                }

    # Add our defined surface to the OAS_prob object.
    OAS_prob.add_surface(surface)

    # Get the finalized surface, which includes the created mesh object.
    # Here, `surface` is a dictionary that contains information relevant to
    # one surface within the analysis or optimization.
    surface = OAS_prob.surfaces[0]
    print(surface['mesh'])

    # If you want to view the information contained within `surface`,
    # uncomment the following two lines of code.
    # for key in surface.keys():
    #     print(key, surface[key])

    # Also get the created prob_dict, which contains information about the
    # flow over the wing.
    prob_dict = OAS_prob.prob_dict

    # Obtain the number of spanwise node points from the defined surface.
    num_y = surface['num_y']

    # Create an array of radii for the spar elements.
    r = radii(surface['mesh'])

    # Obtain the starting thickness for each of the spar elements based
    # on the radii.
    thickness = r / 10

    # Instantiate the OpenMDAO group for the root problem.
    root = Group()

    # Create a list of tuples that contains the design variables.
    # These will be used in the analysis and optimization and will connect
    # to variables within the components.
    # For example, here we set the twist, and VLMGeometry computes
    # the new mesh based off of these twist values.
    indep_vars = [
        ('span', surface['span']),
        ('twist', numpy.zeros(num_y)),
        ('thickness', thickness),
        ('v', prob_dict['v']),
        ('alpha', prob_dict['alpha']),
        ('rho', prob_dict['rho']),
        ('r', r),
        ('M', prob_dict['M']),
        ('Re', prob_dict['Re']),
    ]

    ###############################################################
    # Problem 2a:
    # These are your components. Here we simply create the objects.
    ###############################################################

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

    #################################################################
    # Problem 2a:
    # Now add the components you created above to the correct groups.
    # indep_vars_comp, tube_comp, and vlm_funcs have been
    # done for you as examples.
    #################################################################

    # Add functional components here
    root.add('indep_vars',
             indep_vars_comp,
             promotes=['*'])
    root.add('tube',
             tube_comp,
             promotes=['*'])
    root.add('vlmfuncs',
            vlmfuncs_comp,
            promotes=['*'])
    "<-- add more components here -->"

    # Add components to the coupled MDA here
    coupled = Group()
    coupled.add('mesh',
        "<-- insert mesh_comp here -->",
        promotes=["*"])
    "<-- add more components here -->"

    ############################################################
    # Problem 2b:
    # Try different nonlinear solvers on the coupled and root groups.
    # Nonlinear Gauss Seidel is included as an example.
    # Examine http://openmdao.readthedocs.io/en/latest/srcdocs/packages/openmdao.solvers.html
    # to see syntax for other options.
    ############################################################

    ## Nonlinear Gauss Seidel on the coupled group
    coupled.nl_solver = NLGaussSeidel()
    coupled.nl_solver.options['iprint'] = 1
    coupled.nl_solver.options['atol'] = 1e-5
    coupled.nl_solver.options['rtol'] = 1e-12

    ############################################################
    # Problem 2c:
    # Try different linear solvers for the coupled group.
    # Again examine http://openmdao.readthedocs.io/en/latest/srcdocs/packages/openmdao.solvers.html
    # for linear solver options.
    ############################################################

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

    # Instantiate an OpenMDAO problem and all the root group that we just
    # created to the problem.
    prob = Problem()
    prob.root = root



    #############################################################
    # Problem 3b:
    # Look at
    # http://openmdao.readthedocs.io/en/latest/usr-guide/examples/example_fd.html
    # to see how to force use of finite differencing or complex step.
    # Try using finite differencing on the root problem here.
    ##############################################################

    # Multidisciplinary analysis
    if 'prob2' in input_arg or 'prob3ab' in input_arg:

        # Record the optimization history in `aerostruct.db`. You can view
        # this by running `python plot_all.py as` or `python OptView.py as`.
        recorder = SqliteRecorder('aerostruct.db')
        recorder.options['record_params'] = True
        recorder.options['record_derivs'] = True
        recorder.options['record_metadata'] = True
        prob.driver.add_recorder(recorder)

        # Have OpenMDAO set up the problem that we have constructed.
        prob.setup()

        # Print OpenMDAO solver convergence data
        prob.print_all_convergence()

        # Start timing to see how long the analysis and derivative
        # computation takes.
        st = time.time()

        # Run analysis.
        prob.run_once()

        # Multidisciplinary analysis derivatives
        if 'prob3ab' in input_arg:

            print("------------------------------------------------")
            print("Solving for Derivatives")
            print("------------------------------------------------")

            # Calculate the gradients of fuelburn and vonmises (stress)
            # wrt twist, alpha, and thickness
            jac = prob.calc_gradient(['twist','alpha','thickness'], ['fuelburn', 'vonmises'], return_format="dict")

            # Print the derivative results.
            # Note that we convert fuelburn from Newtosn to kg.
            print("d_fuelburn/d_alpha", jac['fuelburn']['alpha'] / 9.8)
            print("norm(d_fuelburn/d_twist)", numpy.linalg.norm(jac['fuelburn']['twist'] / 9.8))
            print("norm(d_fuelburn/d_thickness)", numpy.linalg.norm(jac['fuelburn']['thickness'] / 9.8))

    # Multidisciplinary optimization
    if 'prob3c' in input_arg:

        # Set a driver for the optimization problem. Without a driver, OpenMDAO
        # doesn't know how to change the parameters to achieve an optimal solution.
        # There are a few options, but ScipyOptimizer and SLSQP are generally
        # the best options to use without installing additional packages.
        prob.driver = ScipyOptimizer()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['disp'] = True
        prob.driver.options['tol'] = 1.0e-5
        prob.driver.options['maxiter'] = 80

        ###############################################################
        # Problem 3c:
        # Add design variables here
        ###############################################################
        prob.driver.add_desvar('<--insert_var_name-->',
                               lower='<-- lower_bound -->',
                               upper='<-- upper_bound -->',
                               scaler='<-- scaler_val -->')

        ###############################################################
        # Problem 3c:
        # Add constraints and objectives
        ###############################################################
        prob.driver.add_objective('<-- insert_var_name -->')
        prob.driver.add_constraint('<-- insert_var_name -->', upper='<-- upper_bound -->')
        prob.driver.add_constraint('<-- insert_var_name -->', equals='<-- eq_val -->')

        # Record the optimization history in `aerostruct.db`. You can view
        # this by running `python plot_all.py as` or `python OptView.py as`.
        recorder = SqliteRecorder('aerostruct.db')
        recorder.options['record_params'] = True
        recorder.options['record_derivs'] = True
        recorder.options['record_metadata'] = True
        prob.driver.add_recorder(recorder)

        # Have OpenMDAO set up the problem that we have constructed.
        prob.setup()

        # Print OpenMDAO solver convergence data
        prob.print_all_convergence()

        # Start timing for the optimization
        st = time.time()

        # Actually run the optimization
        prob.run()

    # Create an html output file showing the problem formulation and data
    # passing with an interactive chart. Open this in any web browser.
    view_model(prob, outfile="my_aerostruct_n2.html", show_browser=False)

    # Uncomment this to print partial derivatives accuracy information
    # prob.check_partial_derivatives(compact_print=True)

    # Print the run time and current fuelburn
    print("\nrun time: {} secs".format(time.time() - st))
    print("fuelburn:", prob['fuelburn'] / 9.8, "kg\n")
