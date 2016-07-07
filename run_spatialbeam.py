""" Example script to run structural-only optimization.
Call as `python run_spatialbeam.py 0` to run a single analysis, or
call as `python run_spatialbeam.py 1` to perform optimization. """

from __future__ import division
import numpy
import sys
from time import time

from openmdao.api import IndepVarComp, Problem, Group, ScipyOptimizer, SqliteRecorder
from geometry import GeometryMesh, gen_crm_mesh, get_inds, gen_mesh
from spatialbeam import SpatialBeamStates, SpatialBeamFunctionals, radii
from materials import MaterialsTube
from openmdao.devtools.partition_tree_n2 import view_tree

try:
    from openmdao.api import pyOptSparseDriver
    SNOPT = True
except:
    SNOPT = False

# Create the mesh with 2 inboard points and 3 outboard points.
# This will be mirrored to produce a mesh with 7 spanwise points,
# or 6 spanwise panels
mesh = gen_crm_mesh(n_points_inboard=2, n_points_outboard=3, num_x=2)

if sys.argv[1].endswith('m'):
    num_x = 3
    num_y = 5
    span = 10.
    chord = 1.
    cosine_spacing = .5
    mesh_wing = gen_mesh(num_x, num_y, span, chord, cosine_spacing)
    num_twist = numpy.max([int((num_y - 1) / 5), 5])

    r = radii(mesh_wing)
    mesh_wing = mesh_wing.reshape(-1, mesh_wing.shape[-1])
    aero_ind = numpy.atleast_2d(numpy.array([num_x, num_y]))
    fem_ind = [num_y]

    nx = 2
    ny = 3
    span = 3.
    chord = 1.
    cosine_spacing = .5
    mesh_tail = gen_mesh(nx, ny, span, chord, cosine_spacing)

    mesh_tail = mesh_tail.reshape(-1, mesh_tail.shape[-1])
    mesh_tail[:, 0] += 1e1

    aero_ind = numpy.vstack((aero_ind, numpy.atleast_2d(numpy.array([nx, ny]))))
    mesh = numpy.vstack((mesh_wing, mesh_tail))

    fem_ind.append(ny)
    aero_ind, fem_ind = get_inds(aero_ind, fem_ind)

else:
    num_x = 2
    num_y = 5
    span = 10.
    chord = 5.
    cosine_spacing = .5
    mesh = gen_mesh(num_x, num_y, span, chord, cosine_spacing)
    num_twist = numpy.max([int((num_y - 1) / 5), 5])

    r = radii(mesh)
    mesh = mesh.reshape(-1, mesh.shape[-1])
    aero_ind = numpy.atleast_2d(numpy.array([num_x, num_y]))
    fem_ind = [num_y]
    aero_ind, fem_ind = get_inds(aero_ind, fem_ind)

num_twist = 5
t = r/20

# Define the material properties
execfile('aluminum.py')

# Define the loads
loads = numpy.zeros((num_y, 6))
loads[0, 2] = loads[-1, 2] = 1e3 # tip load of 1 kN
loads[:, 2] = 1e3 # load of 1 kN at each node

span = 58.7630524 # [m] baseline CRM

root = Group()

des_vars = [
    ('twist', numpy.zeros(num_twist)),
    ('dihedral', 0.),
    ('sweep', 0.),
    ('span', span),
    ('taper', 1.),
    ('r', r),
    ('t', t),
    ('loads', loads),
    ('fem_ind', fem_ind),
    ('aero_ind', aero_ind),
]

root.add('des_vars',
         IndepVarComp(des_vars),
         promotes=['*'])
root.add('mesh',
         GeometryMesh(mesh, aero_ind, num_twist),
         promotes=['*'])
root.add('tube',
         MaterialsTube(aero_ind),
         promotes=['*'])
root.add('spatialbeamstates',
         SpatialBeamStates(aero_ind, fem_ind, E, G),
         promotes=['*'])
root.add('spatialbeamfuncs',
         SpatialBeamFunctionals(aero_ind, fem_ind, E, G, stress, mrho),
         promotes=['*'])

prob = Problem()
prob.root = root

prob.driver = ScipyOptimizer()
prob.driver.options['optimizer'] = 'SLSQP'
prob.driver.options['disp'] = True

if SNOPT:
    prob.driver = pyOptSparseDriver()
    prob.driver.options['optimizer'] = "SNOPT"
    prob.driver.opt_settings = {'Major optimality tolerance': 1.0e-8,
                                'Major feasibility tolerance': 1.0e-8}

prob.driver.add_desvar('t',
                       lower=numpy.ones((num_y-1)) * 0.003,
                       upper=numpy.ones((num_y-1)) * 0.25)
prob.driver.add_objective('energy')
prob.driver.add_constraint('weight', upper=1e5)

prob.driver.add_recorder(SqliteRecorder('spatialbeam.db'))

# prob.root.deriv_options['type'] = 'fd'
prob.setup()
view_tree(prob, outfile="spatialbeam.html", show_browser=False)

st = time()
prob.run_once()
if sys.argv[1].startswith('0'):
    # Uncomment this line to check derivatives.
    prob.check_partial_derivatives(compact_print=True)
    pass
elif sys.argv[1].startswith('1'):
    prob.run()
print "weight", prob['weight']
print "run time", time()-st
