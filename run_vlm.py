""" Example runscript to perform aerodynamics-only optimization.

Call as `python run_vlm.py 0` to run a single analysis, or
call as `python run_vlm.py 1` to perform optimization.

To run with multiple lifting surfaces instead of a single one,
Call as `python run_vlm.py 0m` to run a single analysis, or
call as `python run_vlm.py 1m` to perform optimization.

"""

from __future__ import division
import sys
from time import time
import numpy

from openmdao.api import IndepVarComp, Problem, Group, ScipyOptimizer, SqliteRecorder
from openmdao.devtools.partition_tree_n2 import view_tree
from geometry import GeometryMesh, Bspline, gen_mesh, get_inds
from transfer import TransferDisplacements
from vlm import VLMStates, VLMFunctionals
from b_spline import get_bspline_mtx

# Define the aircraft properties
execfile('CRM.py')

# Single lifting surface
if not sys.argv[1].endswith('m'):
    num_x = 2  # number of chordwise node points
    num_y = 21  # number of spanwise node points, can only be odd numbers
    span = 10.  # full wingspan
    chord = 1.  # root chord
    cosine_spacing = .5  # spacing distribution; 0 is uniform, 1 is cosine
    mesh = gen_mesh(num_x, num_y, span, chord, cosine_spacing)
    num_twist = numpy.max([int((num_y - 1) / 5), 5])

    mesh = mesh.reshape(-1, mesh.shape[-1])
    aero_ind = numpy.atleast_2d(numpy.array([num_x, num_y]))
    fem_ind = [num_y]

# Multiple lifting surfaces
else:
    # Wing mesh generation
    num_x = 3  # number of chordwise node points
    num_y = 5  # number of spanwise node points, can only be odd numbers
    span = 10.  # full wingspan
    chord = 1.  # root chord
    cosine_spacing = .5  # spacing distribution; 0 is uniform, 1 is cosine
    mesh_wing = gen_mesh(num_x, num_y, span, chord, cosine_spacing)
    num_twist = numpy.max([int((num_y - 1) / 5), 5])

    # Add wing indices to aero_ind and fem_ind
    mesh_wing = mesh_wing.reshape(-1, mesh_wing.shape[-1])
    aero_ind = numpy.atleast_2d(numpy.array([num_x, num_y]))

    # Tail mesh generation
    nx = 2  # number of chordwise node points
    ny = 3  # number of spanwise node points, can only be odd numbers
    span = 3.  # full tailspan
    chord = 1.  # root chord
    cosine_spacing = .5  # spacing distribution; 0 is uniform, 1 is cosine
    mesh_tail = gen_mesh(nx, ny, span, chord, cosine_spacing)

    mesh_tail = mesh_tail.reshape(-1, mesh_tail.shape[-1])
    mesh_tail[:, 0] += 1e1

    aero_ind = numpy.vstack((aero_ind, numpy.atleast_2d(numpy.array([nx, ny]))))
    fem_ind = [num_y, ny]

    mesh = numpy.vstack((mesh_wing, mesh_tail))

# Compute the aero and fem indices
aero_ind, fem_ind = get_inds(aero_ind, fem_ind)

# Set additional mesh parameters
dihedral = 0.  # dihedral angle in degrees
sweep = 0.  # shearing sweep angle in degrees
taper = 1.  # taper ratio

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
    ('fem_ind', fem_ind)
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
         VLMStates(aero_ind),
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
# prob.driver.add_desvar('sweep', lower=-10., upper=30.)
# prob.driver.add_desvar('dihedral', lower=-10., upper=20.)
# prob.driver.add_desvar('taper', lower=.5, upper=2.)

# Set the objective (minimize CD on the main wing)
prob.driver.add_objective('CD_wing', scaler=1e4)

# Set the constraint (CL = 0.5 for the main wing)
prob.driver.add_constraint('CL_wing', equals=0.5)

# Record optimization history to a database
# Data saved here can be examined using `plot_all.py`
prob.driver.add_recorder(SqliteRecorder('vlm.db'))

# Can finite difference over the entire model
# Generally faster than using component derivatives
prob.root.deriv_options['type'] = 'fd'

# Setup the problem and produce an N^2 diagram
prob.setup()
view_tree(prob, outfile="aero.html", show_browser=False)

st = time()
prob.run_once()
if sys.argv[1].startswith('0'):  # run analysis once
    # Uncomment the following line to check derivatives.
    # prob.check_partial_derivatives(compact_print=True)
    pass
elif sys.argv[1].startswith('1'):  # perform optimization
    st = time()
    prob.run()
print "run time", time() - st
print
print 'alpha', prob['alpha'], "; CL", prob['CL'], "; CD", prob['CD'], "; num", num_y
