""" Example script to run structural-only optimization.
Call as `python run_spatialbeam.py 0` to run a single analysis, or
call as `python run_spatialbeam.py 1` to perform optimization. """

from __future__ import division
import numpy
import sys
from time import time

from openmdao.api import IndepVarComp, Problem, Group, ScipyOptimizer, SqliteRecorder
from geometry import GeometryMesh, Bspline, gen_crm_mesh, get_inds, gen_mesh
from spatialbeam import SpatialBeamStates, SpatialBeamFunctionals, radii
from materials import MaterialsTube
from openmdao.devtools.partition_tree_n2 import view_tree
from b_spline import get_bspline_mtx

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
    num_x = 2
    num_y = 5
    span = 5.
    chord = 5.
    cosine_spacing = 0.
    mesh_wing = gen_mesh(num_x, num_y, span, chord, cosine_spacing)
    mesh_wing[:, :, 1] = mesh_wing[:, :, 1] - span/2
    num_twist = numpy.max([int((num_y - 1) / 5), 5])

    r_wing = radii(mesh_wing)
    mesh_wing = mesh_wing.reshape(-1, mesh_wing.shape[-1])
    aero_ind = numpy.atleast_2d(numpy.array([num_x, num_y]))
    fem_ind = [num_y]

    nx = 2
    ny = 5
    span = 5.
    chord = 5.
    cosine_spacing = 0.
    mesh_tail = gen_mesh(nx, ny, span, chord, cosine_spacing)
    mesh_tail[:, :, 1] = mesh_tail[:, :, 1] + span/2

    r_tail = radii(mesh_tail)
    mesh_tail = mesh_tail.reshape(-1, mesh_tail.shape[-1])

    aero_ind = numpy.vstack((aero_ind, numpy.atleast_2d(numpy.array([nx, ny]))))
    mesh = numpy.vstack((mesh_wing, mesh_tail))
    r = numpy.hstack((r_wing, r_tail))

    fem_ind.append(ny)
    aero_ind, fem_ind = get_inds(aero_ind, fem_ind)

else:
    num_x = 2
    num_y = 9
    span = 10.
    chord = 5.
    cosine_spacing = 0.
    mesh = gen_mesh(num_x, num_y, span, chord, cosine_spacing)
    num_twist = numpy.max([int((num_y - 1) / 5), 5])

    r = radii(mesh)
    mesh = mesh.reshape(-1, mesh.shape[-1])
    aero_ind = numpy.atleast_2d(numpy.array([num_x, num_y]))
    fem_ind = [num_y]
    aero_ind, fem_ind = get_inds(aero_ind, fem_ind)



num_twist = 5
num_thickness = num_twist
t = r/20
r /= 5

# Define the material properties
execfile('aluminum.py')

# Define the loads
tot_n_fem = numpy.sum(fem_ind[:, 0])
num_surf = fem_ind.shape[0]
loads = numpy.zeros((tot_n_fem, 6))
loads[0, 2] = loads[-1, 2] = 1e3 # tip load of 1 kN

span = 58.7630524 # [m] baseline CRM

root = Group()
jac_twist = get_bspline_mtx(num_twist, num_y)
jac_thickness = get_bspline_mtx(num_thickness, tot_n_fem-num_surf)

des_vars = [
    ('twist_cp', numpy.zeros(num_twist)),
    ('thickness_cp', numpy.ones(num_thickness)*numpy.max(t)),
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
root.add('twist_bsp',
         Bspline('twist_cp', 'twist', jac_twist),
         promotes=['*'])
root.add('thickness_bsp',
         Bspline('thickness_cp', 'thickness', jac_thickness),
         promotes=['*'])
root.add('mesh',
         GeometryMesh(mesh, aero_ind),
         promotes=['*'])
root.add('tube',
         MaterialsTube(aero_ind, fem_ind),
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

prob.driver.add_desvar('thickness_cp',
                       lower=numpy.ones((num_thickness)) * 0.0003,
                       upper=numpy.ones((num_thickness)) * 0.25,
                       scaler=1e5)
# prob.driver.add_objective('energy')
# prob.driver.add_constraint('weight', upper=1e5)
prob.driver.add_objective('weight')
prob.driver.add_constraint('failure', upper=0.0)

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
