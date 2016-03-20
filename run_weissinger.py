from __future__ import division
import numpy
import sys

from openmdao.api import IndepVarComp, Problem, Group, ScipyOptimizer, SqliteRecorder
from geometry import GeometryMesh, mesh_gen
from transfer import TransferDisplacements, TransferLoads
from weissinger import WeissingerStates, WeissingerFunctionals

mesh = mesh_gen(n_points_inboard=2, n_points_outboard=3)
num_y = mesh.shape[1]

span = 58.7630524 # baseline CRM

v = 200.
alpha = 0.5
rho = 1.225

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
         WeissingerFunctionals(num_y),
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

prob.driver.add_desvar('twist',lower=-5.,
                       upper=5.)
prob.driver.add_desvar('alpha', lower=-10., upper=10.)
prob.driver.add_objective('CD', scaler=1e4)
prob.driver.add_constraint('CL', equals=0.3)

# setup data recording
prob.driver.add_recorder(SqliteRecorder('weissinger.db'))

# prob.root.fd_options['force_fd'] = True

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
