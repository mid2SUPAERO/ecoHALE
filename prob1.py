from __future__ import division
import numpy
import sys
import time

from openmdao.api import IndepVarComp, Problem, Group, ScipyOptimizer, SqliteRecorder
from geometry import GeometryMesh, mesh_gen, LinearInterp
from spatialbeam import SpatialBeamStates, SpatialBeamFunctionals, radii
from materials import MaterialsTube
from model_helpers import view_tree

num_y = 10
span = 60
chord = 4

mesh = numpy.zeros((2, num_y,3))
for ind_x in xrange(2):
    for ind_y in xrange(num_y):
        mesh[ind_x, ind_y, :] = [ind_x * chord, ind_y / (num_y-1) * span, 0]

r = radii(mesh)
t = r/10

# Define the material properties
execfile('aluminum.py')

# Define the loads
loads = numpy.zeros((num_y, 6))
# loads[0, 2] = loads[-1, 2] = 1e4 # tip load of 1 kN
loads[:, 2] = 1e4 # load of 1 kN at each node

span = 58.7630524 # [m] baseline CRM

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
root.add('tube',
         MaterialsTube(num_y),
         promotes=['*'])
root.add('spatialbeamstates',
         SpatialBeamStates(num_y, E, G),
         promotes=['*'])
root.add('spatialbeamfuncs',
         SpatialBeamFunctionals(num_y, E, G, stress, mrho),
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
prob.driver.add_constraint('weight', upper=1e5)

prob.root.fd_options['force_fd'] = True
prob.root.fd_options['form'] = 'complex_step'
prob.root.fd_options['step_size'] = 1e-10

prob.driver.add_recorder(SqliteRecorder('spatialbeam.db'))

prob.setup()

view_tree(prob, outfile="prob1.html", show_browser=False)

# prob.run_once()
# prob.check_partial_derivatives(compact_print=True)

st = time.time()
prob.run()
print "run time", time.time() - st
