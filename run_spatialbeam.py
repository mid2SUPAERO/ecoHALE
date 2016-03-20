from __future__ import division
import numpy
import sys

from openmdao.api import IndepVarComp, Problem, Group, ScipyOptimizer
from geometry import GeometryMesh, mesh_gen
from spatialbeam import SpatialBeamGroup

mesh = mesh_gen(n_points_inboard=2, n_points_outboard=3)
num_y = mesh.shape[1]

span = 232.02
chord = 39.37
cons = numpy.array([int((num_y-1)/2)])

E = 200.e9
G = 30.e9
stress = 20.e6
mrho = 3.e3
r = 0.3 * numpy.ones(num_y-1)
t = 0.02 * numpy.ones(num_y-1)

loads = numpy.zeros((num_y, 6))
loads[0, 2] = loads[-1, 2] = 1e3
loads[:, 2] = 1e3

root = Group()

des_vars = [
    ('twist', numpy.zeros(num_y)), 
    ('span', span),
    ('r', r), 
    ('t', t), 
    ('loads', loads) 
]

root.add('des_vars', 
         IndepVarComp(des_vars), 
         promotes=['*'])
root.add('mesh',
         GeometryMesh(mesh),
         promotes=['*'])
root.add('spatialbeam',
         SpatialBeamGroup(num_y, cons, E, G, stress, mrho),
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
prob.driver.add_constraint('weight', upper=0.5)

prob.setup()
prob.run_once()

if sys.argv[1] == '0':
    prob.check_partial_derivatives(compact_print=True)
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
