from __future__ import division
import numpy
import sys

from openmdao.api import IndepVarComp, Problem, Group, ScipyOptimizer, SqliteRecorder
from geometry import GeometryMesh, mesh_gen, LinearInterp
from transfer import TransferDisplacements, TransferLoads
from weissinger import WeissingerStates, WeissingerFunctionals

# Create the mesh with 2 inboard points and 3 outboard points
mesh = mesh_gen(n_points_inboard=2, n_points_outboard=3)
num_y = mesh.shape[1]

if 1:
    num_x = 2
    num_y = 11
    span = 10.
    chord = 1.
    mesh = numpy.zeros((num_x, num_y, 3))
    for ind_x in xrange(num_x):
        for ind_y in xrange(num_y):
            mesh[ind_x, ind_y, :] = [ind_x / (num_x-1) * chord, ind_y / (num_y - 1) * span, 0]

# Define the aircraft properties
execfile('CRM.py')

disp = numpy.zeros((num_y, 6))

root = Group()

des_vars = [
    ('twist', numpy.zeros(num_y)), 
    ('span', span),
    ('v', v),
    ('alpha', alpha), 
    ('rho', rho), 
    ('disp', numpy.zeros((num_y, 6)))
]

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
# prob.driver.options['tol'] = 1.0e-12

prob.driver.add_desvar('twist',lower=-5., upper=10., scaler=1e4)
#prob.driver.add_desvar('alpha', lower=-10., upper=10.)
prob.driver.add_objective('CD', scaler=1e4)
prob.driver.add_constraint('CL', equals=0.5)

# setup data recording
prob.driver.add_recorder(SqliteRecorder('weissinger.db'))

prob.setup()

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
