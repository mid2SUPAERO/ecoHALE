from __future__ import division, print_function
import sys
import numpy as np

from openaerostruct.integration.groups import Aerostruct, AerostructPoint, CoupledAS, CoupledPerformance
from openaerostruct.functionals.total_performance import TotalPerformance
from openaerostruct.integration.utils import connect_aerostruct, connect_aerostruct_old
from openaerostruct.transfer.load_transfer import LoadTransfer
from openaerostruct.aerodynamics.states import VLMStates

from openaerostruct.geometry.utils import generate_mesh


from openmdao.api import IndepVarComp, Problem, Group, NewtonSolver, ScipyIterativeSolver, LinearBlockGS, NonlinearBlockGS, DirectSolver, DenseJacobian, LinearRunOnce, PetscKSP, ScipyOptimizer# TODO, SqliteRecorder, CaseReader, profile
from openmdao.api import view_model


prob_dict = {
            # Problem and solver options
            'with_viscous' : True,  # if true, compute viscous drag

            # Flow/environment properties
            'Re' : 1e6,              # Reynolds number
            'reynolds_length' : 1.0, # characteristic Reynolds length
            'alpha' : 5.,            # [degrees] angle of attack
            'M' : 0.84,              # Mach number at cruise
            'rho' : 0.38,            # [kg/m^3] air density at 35,000 ft
            'a' : 295.4,             # [m/s] speed of sound at 35,000 ft
            'g' : 9.80665,           # [m/s^2] acceleration due to gravity

            # Aircraft properties
            'CT' : 9.80665 * 17.e-6, # [1/s] (9.80665 N/kg * 17e-6 kg/N/s)
                                     # specific fuel consumption
            'R' : 11.165e6,          # [m] maximum range (B777-300)
            'cg' : np.zeros((3)),    # Center of gravity for the
                                     # entire aircraft. Used in trim
                                     # and stability calculations.
            'W0' : 0.4 * 3e5,        # [kg] weight of the airplane without
                                     # the wing structure and fuel.
                                     # The default is 40% of the MTOW of
                                     # B777-300 is 3e5 kg.
            'beta' : 1.,             # weighting factor for mixed objective
            'S_ref_total' : None,    # [m^2] total reference area for the aircraft
            }

prob_dict['v'] = prob_dict['M'] * prob_dict['a']

# Create a dictionary to store options about the surface
mesh_dict = {'num_y' : 7,
             'num_x' : 2,
             'wing_type' : 'CRM',
             'symmetry' : True,
             'num_twist_cp' : 5}

mesh, twist_cp = generate_mesh(mesh_dict)

surf_dict = {
            # Wing definition
            'name' : 'wing_',        # name of the surface
            'symmetry' : True,     # if true, model one half of wing
                                    # reflected across the plane y = 0
            'S_ref_type' : 'wetted', # how we compute the wing area,
                                     # can be 'wetted' or 'projected'

            'num_twist_cp' : 5,
            'num_thickness_cp' : 2,
            'num_y' : 7,
            'num_x' : 2,

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

            # Structural values are based on aluminum 7075
            'E' : 70.e9,            # [Pa] Young's modulus of the spar
            'G' : 30.e9,            # [Pa] shear modulus of the spar
            'yield' : 500.e6 / 2.5, # [Pa] yield stress divided by 2.5 for limiting case
            'mrho' : 3.e3,          # [kg/m^3] material density
            'fem_origin' : 0.35,    # normalized chordwise location of the spar
            'loads' : None,         # [N] allow the user to input loads
            'thickness_cp' : np.array([.1, .2, .3]),

            # Constraints
            'exact_failure_constraint' : False, # if false, use KS function
            'monotonic_con' : None, # add monotonic constraint to the given
                                    # distributed variable. Ex. 'chord_cp'
            }

surf_dict.update({'twist_cp' : twist_cp,
                  'mesh' : mesh})

surf_dict['num_x'], surf_dict['num_y'] = surf_dict['mesh'].shape[:2]

surfaces = [surf_dict]

# Create the problem and assign the model group
prob = Problem()

# Add problem information as an independent variables component
indep_var_comp = IndepVarComp()
indep_var_comp.add_output('v', val=prob_dict['v'])
indep_var_comp.add_output('alpha', val=prob_dict['alpha'])
indep_var_comp.add_output('M', val=prob_dict['M'])
indep_var_comp.add_output('re', val=prob_dict['Re']/prob_dict['reynolds_length'])
indep_var_comp.add_output('rho', val=prob_dict['rho'])

prob.model.add_subsystem('prob_vars',
     indep_var_comp,
     promotes=['*'])

# Loop over each surface in the surfaces list
for surface in surfaces:

    # Get the surface name and create a group to contain components
    # only for this surface
    name = surface['name']
    ny = surface['num_y']

    # Add independent variables that do not belong to a specific component
    indep_var_comp = IndepVarComp()

    indep_var_comp.add_output('twist_cp', val=surface['twist_cp'])
    indep_var_comp.add_output('thickness_cp', val=surface['thickness_cp'])
    # indep_var_comp.add_output('radius', val=np.ones((ny)) * .5)

    aerostruct_group = Aerostruct(surface=surface, indep_var_comp=indep_var_comp)

    # Add tmp_group to the problem with the name of the surface.
    prob.model.add_subsystem(name[:-1], aerostruct_group)

# Loop through and add a certain number of aero points
for i in range(1):

    if 1:

        # Create the aero point group and add it to the model
        AS_point = AerostructPoint(surfaces=surfaces, prob_dict=prob_dict)
        point_name = 'AS_point_{}'.format(i)
        prob.model.add_subsystem(point_name, AS_point)

        # Connect flow properties to the analysis point
        prob.model.connect('v', point_name + '.v')
        prob.model.connect('alpha', point_name + '.alpha')
        prob.model.connect('M', point_name + '.M')
        prob.model.connect('re', point_name + '.re')
        prob.model.connect('rho', point_name + '.rho')

        # Connect the parameters within the model for each aero point
        for surface in surfaces:
            connect_aerostruct(prob.model, point_name, surface['name'])

    else:

        coupled = Group()

        for surface in surfaces:

            name = surface['name']

            # Add components to the 'coupled' group for each surface.
            # The 'coupled' group must contain all components and parameters
            # needed to converge the aerostructural system.
            coupled_AS_group = CoupledAS(surface=surface)

            coupled.add_subsystem(name[:-1], coupled_AS_group)

            # TODO: add this info to the metadata
            # prob.model.add_metadata(surface['name'] + 'yield_stress', surface['yield'])
            # prob.model.add_metadata(surface['name'] + 'fem_origin', surface['fem_origin'])

        # Add a single 'aero_states' component for the whole system within the
        # coupled group.
        coupled.add_subsystem('aero_states',
            VLMStates(surfaces=surfaces),
            promotes=['v', 'alpha', 'rho'])

        # Explicitly connect parameters from each surface's group and the common
        # 'aero_states' group.
        for surface in surfaces:
            name = surface['name']

            # Add a loads component to the coupled group
            coupled.add_subsystem(name + 'loads', LoadTransfer(surface=surface))

        # Set solver properties for the coupled group
        # coupled.linear_solver = ScipyIterativeSolver()
        # coupled.linear_solver.precon = LinearRunOnce()
        #
        # coupled.nonlinear_solver = NonlinearBlockGS()
        # coupled.nonlinear_solver.options['maxiter'] = 50

        coupled.jacobian = DenseJacobian()
        coupled.linear_solver = DirectSolver()
        coupled.nonlinear_solver = NewtonSolver(solve_subsystems=True)

        coupled.linear_solver.options['iprint'] = 2
        coupled.nonlinear_solver.options['iprint'] = 2

        # Add the coupled group to the model problem
        prob.model.add_subsystem('coupled', coupled, promotes=['v', 'alpha', 'rho'])

        for surface in surfaces:
            name = surface['name']

            # Add a performance group which evaluates the data after solving
            # the coupled system
            perf_group = CoupledPerformance(surface=surface, prob_dict=prob_dict)

            prob.model.add_subsystem(name + 'perf', perf_group, promotes=["rho", "v", "alpha", "re", "M"])
            connect_aerostruct_old(prob.model, name)

        # Add functionals to evaluate performance of the system.
        # Note that only the interesting results are promoted here; not all
        # of the parameters.
        prob.model.add_subsystem('total_perf',
                 TotalPerformance(surfaces=surfaces, prob_dict=prob_dict),
                 promotes=['L_equals_W', 'fuelburn', 'CM', 'CL', 'CD', 'v', 'rho', 'cg', 'weighted_obj', 'total_weight'])


from openmdao.api import pyOptSparseDriver
prob.driver = pyOptSparseDriver()
prob.driver.options['optimizer'] = "SNOPT"
prob.driver.opt_settings = {'Major optimality tolerance': 1.0e-8,
                                 'Major feasibility tolerance': 1.0e-8}

# # Setup problem and add design variables, constraint, and objective
# prob.model.add_design_var('wing.twist_cp', lower=-10., upper=15.)
# prob.model.add_design_var('wing.thickness_cp', lower=0.01, upper=0.5, scaler=1e2)
# prob.model.add_constraint('AS_point_0.wing_perf.failure', upper=0.)
# prob.model.add_constraint('AS_point_0.wing_perf.thickness_intersects', upper=0.)
#
# # Add design variables, constraisnt, and objective on the problem
# prob.model.add_design_var('alpha', lower=-10., upper=10.)
# prob.model.add_constraint('AS_point_0.L_equals_W', equals=0.)
# prob.model.add_objective('AS_point_0.fuelburn', scaler=1e-5)

# Set up the problem
prob.setup()

# prob.print_all_convergence()

# Save an N2 diagram for the problem
view_model(prob, outfile='aerostruct.html', show_browser=False)

prob.run_model()
# prob.run_driver()

# prob.check_partials(compact_print=True)

# for name in prob.model._outputs:
#     print(name)
#     print(prob.model._outputs[name])
#     print()

print("\nFuelburn", prob['AS_point_0.fuelburn'])
