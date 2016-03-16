from __future__ import division
import numpy
import sys

from openmdao.api import IndepVarComp, Problem, Group, ScipyOptimizer
from geometry import GeometryMesh
from transfer import TransferDisplacements

from weissinger import WeissingerGroup

num_y = 3
span = 232.02
chord = 39.37

v = 200.
alpha = 3.
rho = 1.225

disp = numpy.zeros((num_y, 6))

root = Group()

root.add('twist',
         IndepVarComp('twist', numpy.zeros((num_y))),
         promotes=['*'])
root.add('v',
         IndepVarComp('v', v),
         promotes=['*'])
root.add('alpha',
         IndepVarComp('alpha', alpha),
         promotes=['*'])
root.add('rho',
         IndepVarComp('rho', rho),
         promotes=['*'])
root.add('disp',
         IndepVarComp('disp', disp),
         promotes=['*'])

root.add('mesh',
         GeometryMesh(num_y, span, chord),
         promotes=['*'])
root.add('def_mesh',
         TransferDisplacements(num_y),
         promotes=['*'])
root.add('weissinger',
         WeissingerGroup(num_y),
         promotes=['*'])

prob = Problem()
prob.root = root

prob.driver = ScipyOptimizer()
prob.driver.options['optimizer'] = 'SLSQP'
prob.driver.options['disp'] = True
# prob.driver.options['tol'] = 1.0e-12

prob.driver.add_desvar('twist',lower=numpy.ones((num_y)) * -10.,
                       upper=numpy.ones((num_y)) * 10.)
prob.driver.add_desvar('alpha', lower=-10., upper=10., scaler=100)
prob.driver.add_objective('CD')
prob.driver.add_constraint('CL', equals=0.5)

prob.setup()
prob.run_once()

if sys.argv[1] == '0':
    prob.check_partial_derivatives()
    prob.run_once()
    print
    print prob['CL'], prob['CD']
elif sys.argv[1] == '1':
    prob.run()
    print prob['alpha'], prob['CL'], prob['CD']
    print prob['twist']
