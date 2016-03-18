from __future__ import division
import numpy
import sys

from openmdao.api import IndepVarComp, Problem, Group, ScipyOptimizer, SqliteRecorder
from geometry import GeometryMesh
from transfer import TransferDisplacements, TransferLoads

from weissinger import WeissingerGroup

num_y = 5
# span = 232.02
span = 100.
chord = 10.

v = 200.
alpha = 3.
rho = 1.225

disp = numpy.zeros((num_y, 6))

root = Group()

des_vars = [
    ('twist', numpy.zeros(num_y)), 
    ('span', span),
    ('chord', numpy.ones(num_y)*chord),
    ('v', v),
    ('alpha', alpha), 
    ('rho', rho), 
    ('disp', numpy.zeros((num_y, 6)))
]

# root.add('twist',
#          IndepVarComp('twist', numpy.zeros((num_y))),
#          promotes=['*'])
# root.add('span',
#          IndepVarComp('span', span),
#          promotes=['*'])
# root.add('v',
#          IndepVarComp('v', v),
#          promotes=['*'])
# root.add('alpha',
#          IndepVarComp('alpha', alpha),
#          promotes=['*'])
# root.add('rho',
#          IndepVarComp('rho', rho),
#          promotes=['*'])
# root.add('disp',
#          IndepVarComp('disp', disp),
#          promotes=['*'])

root.add('des_vars', 
         IndepVarComp(des_vars), 
         promotes=['*'])
root.add('mesh',
         GeometryMesh(num_y),
         promotes=['*'])
root.add('def_mesh',
         TransferDisplacements(num_y),
         promotes=['*'])
root.add('weissinger',
         WeissingerGroup(num_y),
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

prob.driver.add_desvar('twist',lower=numpy.ones((num_y)) * -10.,
                       upper=numpy.ones((num_y)) * 10.)
# prob.driver.add_desvar('chord',lower=numpy.ones((num_y)) * 1,
#                        upper=numpy.ones((num_y)) * 12)
prob.driver.add_desvar('alpha', lower=-10., upper=10.)
prob.driver.add_objective('CD')
prob.driver.add_constraint('CL', equals=0.8)

#setup data recording
prob.driver.add_recorder(SqliteRecorder('weissinger.db'))

prob.setup()
prob.run_once()

if sys.argv[1] == '0':
    # prob.check_partial_derivatives(compact_print=True)
    prob.check_total_derivatives()
    prob.run_once()
    print
    print prob['CL'], prob['CD']
elif sys.argv[1] == '1':
    prob.run()
    print prob['alpha'], prob['CL'], prob['CD']
    print prob['twist']
    print prob['chord']
