""" Example script to run aerodynamics-only optimization.
Call as `python run_vlm.py 0` to run a single analysis, or
call as `python run_vlm.py 1` to perform optimization. """

from __future__ import division
import numpy
import sys
from time import time

from openmdao.api import IndepVarComp, Problem, Group, ScipyOptimizer, SqliteRecorder
from geometry import GeometryMesh, Bspline, gen_crm_mesh, gen_mesh, get_inds
from transfer import TransferDisplacements, TransferLoads
from vlm import VLMStates, VLMFunctionals
from openmdao.devtools.partition_tree_n2 import view_tree
from b_spline import get_bspline_mtx

try:
    from openmdao.api import pyOptSparseDriver
    SNOPT = True
except:
    SNOPT = False

# Define the aircraft properties
execfile('CRM.py')

if sys.argv[1].endswith('m'):
    num_x = 3
    num_y = 3
    span = 10.
    chord = 1.
    cosine_spacing = .5
    mesh_wing = gen_mesh(num_x, num_y, span, chord, cosine_spacing)
    num_twist = numpy.max([int((num_y - 1) / 5), 5])

    mesh_wing = mesh_wing.reshape(-1, mesh_wing.shape[-1])
    aero_ind = numpy.atleast_2d(numpy.array([num_x, num_y]))

    nx = 2
    ny = 3
    span = 3.
    chord = 1.
    cosine_spacing = .5
    mesh_tail = gen_mesh(nx, ny, span, chord, cosine_spacing)

    mesh_tail = mesh_tail.reshape(-1, mesh_tail.shape[-1])
    mesh_tail[:, 0] += 1e1

    aero_ind = numpy.vstack((aero_ind, numpy.atleast_2d(numpy.array([nx, ny]))))
    fem_ind = [num_y, ny]

    mesh = numpy.vstack((mesh_wing, mesh_tail))

else:
    num_x = 2
    num_y = 21
    span = 10.
    chord = 1.
    cosine_spacing = .5
    mesh = gen_mesh(num_x, num_y, span, chord, cosine_spacing)
    num_twist = numpy.max([int((num_y - 1) / 5), 5])

    mesh = mesh.reshape(-1, mesh.shape[-1])
    aero_ind = numpy.atleast_2d(numpy.array([num_x, num_y]))
    fem_ind = [num_y]

aero_ind, fem_ind = get_inds(aero_ind, fem_ind)
tot_n_fem = numpy.sum(fem_ind[:, 0])

root = Group()

jac = get_bspline_mtx(num_twist, num_y)
des_vars = [
    ('twist_cp', numpy.zeros(num_twist)),
    ('dihedral', 0.),
    ('sweep', 0.),
    ('span', span),
    ('taper', 1.),
    ('v', v),
    ('alpha', alpha),
    ('rho', rho),
    ('disp', numpy.zeros((tot_n_fem, 6))),
    ('aero_ind', aero_ind),
    ('fem_ind', fem_ind)
]


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

prob = Problem()
prob.root = root

prob.driver = ScipyOptimizer()
prob.driver.options['optimizer'] = 'SLSQP'
prob.driver.options['disp'] = True
prob.driver.options['tol'] = 1.0e-8

if SNOPT:
    prob.driver = pyOptSparseDriver()
    prob.driver.options['optimizer'] = "SNOPT"
    prob.driver.opt_settings = {'Major optimality tolerance': 1.0e-8,
                                'Major feasibility tolerance': 1.0e-8}

prob.driver.add_desvar('twist_cp', lower=-10., upper=15., scaler=1e0)
# prob.driver.add_desvar('alpha', lower=-10., upper=10.)
# prob.driver.add_desvar('sweep', lower=-10., upper=30.)
# prob.driver.add_desvar('dihedral', lower=-10., upper=20.)
# prob.driver.add_desvar('taper', lower=.5, upper=2.)
prob.driver.add_objective('CD_wing', scaler=1e4)
prob.driver.add_constraint('CL_wing', equals=0.5)

# setup data recording
prob.driver.add_recorder(SqliteRecorder('vlm.db'))

# prob.root.deriv_options['type'] = 'fd'
prob.setup()

view_tree(prob, outfile="aero.html", show_browser=False)

st = time()
prob.run_once()
if sys.argv[1].startswith('0'):
    # Uncomment this line to check derivatives.
    prob.check_partial_derivatives(compact_print=True)
    pass
elif sys.argv[1].startswith('1'):
    st = time()
    prob.run()
print "run time", time() - st
print
print 'alpha', prob['alpha'], "; CL", prob['CL'], "; CD", prob['CD'], "; num", num_y
