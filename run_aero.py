from __future__ import division, print_function
import sys
import numpy as np
from openaerostruct.integration.integration import OASProblem
from openaerostruct.geometry.utils import generate_mesh
from openaerostruct.geometry.bsplines import Bsplines
from openaerostruct.geometry.new_geometry_mesh import GeometryMesh
from openaerostruct.transfer.displacement_transfer import DisplacementTransfer
from openaerostruct.aerodynamics.utils import connect_aero

from openaerostruct.aerodynamics.groups import AeroPoint

from openmdao.api import IndepVarComp, Problem, Group, NewtonSolver, ScipyIterativeSolver, LinearBlockGS, NonlinearBlockGS, DirectSolver, DenseJacobian, LinearRunOnce, PetscKSP, ScipyOptimizer# TODO, SqliteRecorder, CaseReader, profile
from openmdao.api import view_model
from six import iteritems


# Set problem type
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

            'S_ref_total' : None,    # [m^2] total reference area for the aircraft
            'cg' : np.zeros((3))
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

            'fem_origin' : 0.35,

            # Airfoil properties for viscous drag calculation
            'k_lam' : 0.05,         # percentage of chord with laminar
                                    # flow, used for viscous drag
            't_over_c' : 0.15,      # thickness over chord ratio (NACA0015)
            'c_max_t' : .303,       # chordwise location of maximum (NACA0015)
                                    # thickness
            }

surf_dict.update({'twist_cp' : twist_cp,
                  'mesh' : mesh})
# TODO: move offset back to in the loop

surf_dict['num_x'], surf_dict['num_y'] = surf_dict['mesh'].shape[:2]

surfaces = [surf_dict]

# Create the problem and the model group
prob = Problem()

indep_var_comp = IndepVarComp()
indep_var_comp.add_output('v', val=prob_dict['v'])
indep_var_comp.add_output('alpha', val=prob_dict['alpha'])
indep_var_comp.add_output('M', val=prob_dict['M'])
indep_var_comp.add_output('re', val=prob_dict['Re']/prob_dict['reynolds_length'])
indep_var_comp.add_output('rho', val=prob_dict['rho'])
indep_var_comp.add_output('cg', val=prob_dict['cg'])

prob.model.add_subsystem('prob_vars',
    indep_var_comp,
    promotes=['*'])

# Loop over each surface in the surfaces list
for surface in surfaces:

    # Get the surface name and create a group to contain components
    # only for this surface
    ny = surface['mesh'].shape[1]

    # Add independent variables that do not belong to a specific component
    indep_var_comp = IndepVarComp()
    indep_var_comp.add_output('disp', val=np.zeros((ny, 6)))
    indep_var_comp.add_output('twist_cp', val=surface['twist_cp'])

    tmp_group = Group()

    # Add structural components to the surface-specific group
    tmp_group.add_subsystem('indep_vars',
             indep_var_comp,
             promotes=['*'])

    # Add bspline components for active bspline geometric variables.
    tmp_group.add_subsystem('twist_bsp', Bsplines(
        in_name='twist_cp', out_name='twist',
        num_cp=int(surface['num_twist_cp']), num_pt=int(ny)),
        promotes=['*'])

    tmp_group.add_subsystem('mesh',
        GeometryMesh(surface=surface),
        promotes=['*'])

    tmp_group.add_subsystem('def_mesh',
        DisplacementTransfer(surface=surface),
        promotes=['*'])

    # Add tmp_group to the problem as the name of the surface.
    # Note that is a group and performance group for each
    # individual surface.
    prob.model.add_subsystem(surface['name'][:-1], tmp_group, promotes=[])

# Loop through and add a certain number of aero points
for i in range(1):

    # Create the aero point group and add it to the model
    aero_group = AeroPoint(surfaces=surfaces, prob_dict=prob_dict)
    point_name = 'aero_point_{}'.format(i)
    prob.model.add_subsystem(point_name, aero_group, promotes=[])

    # Connect flow properties to the analysis point
    prob.model.connect('v', point_name + '.v')
    prob.model.connect('alpha', point_name + '.alpha')
    prob.model.connect('M', point_name + '.M')
    prob.model.connect('re', point_name + '.re')
    prob.model.connect('rho', point_name + '.rho')

    # Connect the parameters within the model for each aero point
    for surface in surfaces:
        connect_aero(prob.model, point_name, surface['name'])

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

view_model(prob, outfile='aero.html', show_browser=False)

# prob.run_model()
prob.run_driver()

prob.check_partials(compact_print=True)

print("\nWing CL:", prob['aero_point_0.wing_perf.CL'])
print("Wing CD:", prob['aero_point_0.wing_perf.CD'])
