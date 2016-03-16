from __future__ import division
import numpy
import sys

from openmdao.api import IndepVarComp, Problem, Group, ScipyOptimizer
from geometry import GeometryMesh
from spatialbeam import SpatialBeamGroup


num_y = 3
span = 232.02
chord = 39.37
cons = numpy.array([int((num_y-1)/2)])

E = 200.e9
G = 30.e9
r = 0.3 * numpy.ones(num_y-1)
t = 0.02 * numpy.ones(num_y-1)

loads = numpy.zeros((num_y, 6))
loads[0, 2] = loads[-1, 2] = 1e3
loads[:, 2] = 1e3

root = Group()

root.add('twist',
         IndepVarComp('twist', numpy.zeros((num_y))),
         promotes=['*'])
root.add('r',
         IndepVarComp('r', r),
         promotes=['*'])
root.add('t',
         IndepVarComp('t', t),
         promotes=['*'])
root.add('loads',
         IndepVarComp('loads', loads),
         promotes=['*'])

root.add('mesh',
         GeometryMesh(num_y, span, chord),
         promotes=['*'])
root.add('spatialbeam',
         SpatialBeamGroup(num_y, cons, E, G),
         promotes=['*'])

prob = Problem()
prob.root = root

prob.driver = ScipyOptimizer()
prob.driver.options['optimizer'] = 'SLSQP'
prob.driver.options['disp'] = True
# prob.driver.options['tol'] = 1.0e-12

prob.driver.add_desvar('t',
                       lower=numpy.ones((num_y)) * 0.001,
                       upper=numpy.ones((num_y)) * 0.25)
prob.driver.add_objective('energy')
prob.driver.add_constraint('weight', upper = 0.5)

prob.setup()
prob.run_once()

if sys.argv[1] == '0':
    prob.check_partial_derivatives()
    prob.run_once()
    print
    print prob['A']
    print prob['Iy']
    print prob['Iz']
    print prob['J']
    print
    print prob['disp']
elif sys.argv[1] == '1':
    prob.run()
