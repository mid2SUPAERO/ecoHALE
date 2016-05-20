from __future__ import division
import numpy
import sys

from openmdao.api import IndepVarComp, Problem, Group, ScipyOptimizer, SqliteRecorder, pyOptSparseDriver, setup_profiling, activate_profiling
from geometry import GeometryMesh, mesh_gen, LinearInterp
from transfer import TransferDisplacements, TransferLoads
from weissinger import WeissingerStates, WeissingerFunctionals
from openmdao.devtools.partition_tree_n2 import view_tree

# Create the mesh with 2 inboard points and 3 outboard points
mesh = mesh_gen(n_points_inboard=2, n_points_outboard=3)
num_y = mesh.shape[1]

# Define the aircraft properties
execfile('CRM.py')

if 1:
    num_x = 2
    num_y = 11
    span = 10.
    chord = 2.
    mesh = numpy.zeros((num_x, num_y, 3))
    ny2 = (num_y + 1) / 2
    half_wing = numpy.zeros((ny2))
    beta = numpy.linspace(0, numpy.pi/2, ny2)
    half_wing = (.5 * numpy.cos(beta))**2
    full_wing = numpy.hstack((-half_wing[:-1], half_wing[::-1]))

    for ind_x in xrange(num_x):
        for ind_y in xrange(num_y):
            mesh[ind_x, ind_y, :] = [ind_x / (num_x-1) * chord + numpy.random.rand(),
                                     full_wing[ind_y] * span,
                                     0 + numpy.random.rand()]

            mesh[ind_x, ind_y, :] = [ind_x / (num_x-1) * chord, full_wing[ind_y] * span, 0]

if 0:
    num_x = 2
    num_y = 3
    span = 10.
    chord = 2.
    mesh = numpy.zeros((num_x, num_y, 3))

    for ind_x in xrange(num_x):
        for ind_y in xrange(num_y):
            mesh[ind_x, ind_y, :] = [ind_x / (num_x-1) * chord, ind_y / (num_y-1), 0]


disp = numpy.zeros((num_y, 6))

root = Group()

des_vars = [
    ('twist', numpy.zeros(num_y)),
    # ('twist', 12*(numpy.random.rand((num_y))-.5)),
    ('span', span),
    ('v', v),
    ('alpha', alpha),
    ('rho', rho),
    ('disp', numpy.zeros((num_y, 6)))
]

# print numpy.random.rand((num_y))

root.add('des_vars',
         IndepVarComp(des_vars),
         promotes=['*'])
#root.add('linear_twist',
#         LinearInterp(num_y, 'twist'),
#         promotes=['*'])
root.add('mesh',
         GeometryMesh(mesh),
         promotes=['*'])
root.add('def_mesh',
         TransferDisplacements(num_y),
         promotes=['*'])
root.add('weissingerstates',
         WeissingerStates(num_y),
         promotes=['*'])
root.add('weissingerfuncs',
         WeissingerFunctionals(num_y, CL0, CD0),
         promotes=['*'])
root.add('loads',
         TransferLoads(num_y),
         promotes=['*'])

prob = Problem()
prob.root = root

prob.driver = ScipyOptimizer()
prob.driver.options['optimizer'] = 'SLSQP'
prob.driver.options['disp'] = True
prob.driver.options['tol'] = 1.0e-8

if 1:
    prob.driver = pyOptSparseDriver()
    prob.driver.options['optimizer'] = "SNOPT"
    prob.driver.opt_settings = {'Major optimality tolerance': 1.0e-7,
                                'Major feasibility tolerance': 1.0e-7}

prob.driver.add_desvar('twist',lower=-5., upper=10., scaler=1e0) # test
#prob.driver.add_desvar('alpha', lower=-10., upper=10.)
prob.driver.add_objective('CD', scaler=1e4)
prob.driver.add_constraint('CL', equals=0.5)
# setup data recording
prob.driver.add_recorder(SqliteRecorder('weissinger.db'))

setup_profiling(prob)
activate_profiling()

prob.setup()
view_tree(prob, outfile="aerostruct.html", show_browser=False)

prob.run_once()
import time
if sys.argv[1] == '0':
    st = time.time()
    prob.check_partial_derivatives(compact_print=True)
    # prob.check_total_derivatives()
    print "run time", time.time() - st
    print
    print prob['CL'], prob['CD']
elif sys.argv[1] == '1':
    prob.run()
    print 'alpha', prob['alpha'], "; CL ", prob['CL'], "; CD ", prob['CD']
    print prob['twist']
