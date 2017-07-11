from __future__ import division, print_function
import sys
import numpy as np

from openaerostruct.geometry.utils import generate_mesh
from openaerostruct.structures.groups import SpatialBeamAlone
from openaerostruct.geometry.bsplines import Bsplines

from openmdao.api import IndepVarComp, Problem, Group, NewtonSolver, ScipyIterativeSolver, LinearBlockGS, NonlinearBlockGS, DirectSolver, DenseJacobian, LinearRunOnce, PetscKSP, ScipyOptimizer# TODO, SqliteRecorder, CaseReader, profile
from openmdao.api import view_model
from six import iteritems


# Create a dictionary to store options about the surface
mesh_dict = {'num_y' : 7,
             'wing_type' : 'CRM',
             'symmetry' : True,
             'num_twist_cp' : 5}

mesh, twist_cp = generate_mesh(mesh_dict)

surf_dict = {
            # Wing definition
            'name' : 'wing',        # name of the surface
            'symmetry' : True,     # if true, model one half of wing
                                    # reflected across the plane y = 0

            # Structural values are based on aluminum 7075
            'E' : 70.e9,            # [Pa] Young's modulus of the spar
            'G' : 30.e9,            # [Pa] shear modulus of the spar
            'yield' : 500.e6 / 2.5, # [Pa] yield stress divided by 2.5 for limiting case
            'mrho' : 3.e3,          # [kg/m^3] material density
            'fem_origin' : 0.35,    # normalized chordwise location of the spar
            't_over_c' : 0.15,      # maximum airfoil thickness

            'exact_failure_constraint' : False,
            }

surf_dict.update({'mesh' : mesh})

surf_dict['num_x'], surf_dict['num_y'] = surf_dict['mesh'].shape[:2]
num_thickness_cp = 3

surfaces = [surf_dict]

# Create the problem and assign the model group
prob = Problem()

# Loop over each surface in the surfaces list
for surface in surfaces:
    ny = surface['num_y']

    indep_var_comp = IndepVarComp()
    indep_var_comp.add_output('thickness_cp', val=np.ones(num_thickness_cp) * .1)
    indep_var_comp.add_output('loads', val=np.ones(ny) * 2e5)

    struct_group = SpatialBeamAlone(surface=surface)

    # Add indep_vars to the structural group
    struct_group.add_subsystem('indep_vars',
         indep_var_comp,
         promotes=['*'])

    # Add the B-spline componetn that translates the thickness control points
    # into thicknesses for each FEM element
    struct_group.add_subsystem('thickness_bsp', Bsplines(
        in_name='thickness_cp', out_name='thickness',
        num_cp=num_thickness_cp, num_pt=int(ny-1)),
        promotes_inputs=['thickness_cp'],
        promotes_outputs=['thickness'])

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

# prob.check_partials(compact_print=True)

print("\nWing structural weight:", prob['wing.structural_weight'])
