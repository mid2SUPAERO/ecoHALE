from __future__ import division, print_function
import numpy as np

from openaerostruct.geometry.utils import generate_mesh
from openaerostruct.geometry.bsplines import Bsplines
from openaerostruct.geometry.new_geometry_mesh import GeometryMesh
from openaerostruct.transfer.displacement_transfer import DisplacementTransfer

from openaerostruct.aerodynamics.groups import AeroPoint

from openmdao.api import IndepVarComp, Problem, Group, NewtonSolver, ScipyIterativeSolver, LinearBlockGS, NonlinearBlockGS, DirectSolver, DenseJacobian, LinearRunOnce, PetscKSP, ScipyOptimizer# TODO, SqliteRecorder, CaseReader, profile
from openmdao.api import view_model
from six import iteritems


# Set problem type
prob_dict = {
            # Problem and solver options
            'with_viscous' : True,  # if true, compute viscous drag
            'g' : 9.80665,           # [m/s^2] acceleration due to gravity
            'S_ref_total' : None,    # [m^2] total reference area for the aircraft
            }


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
indep_var_comp.add_output('cg', val=np.zeros((3)))

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
        promotes_inputs=['twist_cp'], promotes_outputs=['twist'])

    tmp_group.add_subsystem('mesh',
        GeometryMesh(surface=surface),
        promotes_inputs=['twist'],
        promotes_outputs=['mesh'])

    tmp_group.add_subsystem('def_mesh',
        DisplacementTransfer(surface=surface),
        promotes_inputs=['disp', 'mesh'],
        promotes_outputs=['def_mesh'])

    # Add tmp_group to the problem as the name of the surface.
    # Note that is a group and performance group for each
    # individual surface.
    prob.model.add_subsystem(surface['name'][:-1], tmp_group)

# Loop through and add a certain number of aero points
for i in range(1):

    # Create the aero point group and add it to the model
    aero_group = AeroPoint(surfaces=surfaces, prob_dict=prob_dict)
    point_name = 'aero_point_{}'.format(i)
    prob.model.add_subsystem(point_name, aero_group)

    # Connect flow properties to the analysis point
    prob.model.connect('v', point_name + '.v')
    prob.model.connect('alpha', point_name + '.alpha')
    prob.model.connect('M', point_name + '.M')
    prob.model.connect('re', point_name + '.re')
    prob.model.connect('rho', point_name + '.rho')
    prob.model.connect('cg', point_name + '.cg')

    # Connect the parameters within the model for each aero point
    for surface in surfaces:

        name = surface['name']

        # Connect the mesh from the geometry component to the analysis point
        prob.model.connect(name[:-1] + '.def_mesh', point_name + '.' + name[:-1] + '.def_mesh')

        # Perform the connections with the modified names within the
        # 'aero_states' group.
        prob.model.connect(name[:-1] + '.def_mesh', point_name + '.aero_states.' + name + 'def_mesh')

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
