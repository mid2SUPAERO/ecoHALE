from __future__ import division
import numpy
import sys
import time

from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from openmdao.api import IndepVarComp, Problem, Group, ScipyOptimizer, SqliteRecorder
from geometry import GeometryMesh, gen_crm_mesh, LinearInterp
from spatialbeam import SpatialBeamStates, SpatialBeamFunctionals, radii
from materials import MaterialsTube
from openmdao.api import view_model
from run_classes import OASProblem

# Set problem type
prob_dict = {'type' : 'struct'}

# Instantiate problem and add default surface
OAS_prob = OASProblem(prob_dict)
OAS_prob.add_surface({'name' : 'wing',
                      'num_y' : 5,
                      'span_cos_spacing' : 0,
                      'symmetry' : True})
# Get the created surface
surface = OAS_prob.surfaces[0]

num_y = surface['num_y']

r = radii(surface['mesh'])
thickness = r / 10

# Define the loads
loads = numpy.zeros((num_y, 6))
# loads[0, 2] = loads[-1, 2] = 1e4 # tip load of 1 kN
loads[:, 2] = 1e4 # load of 1 kN at each node

root = Group()

des_vars = [
    ('twist', numpy.zeros(num_y)),
    ('span', surface['span']),
    ('r', r),
    ('thickness', thickness),
    ('loads', loads)
]

root.add('des_vars',
         IndepVarComp(des_vars),
         promotes=['*'])
root.add('mesh',
         GeometryMesh(surface),
         promotes=['*'])
root.add('tube',
         MaterialsTube(surface),
         promotes=['*'])
root.add('spatialbeamstates',
         SpatialBeamStates(surface),
         promotes=['*'])
root.add('spatialbeamfuncs',
         SpatialBeamFunctionals(surface),
         promotes=['*'])

prob = Problem()
prob.root = root


prob.driver = ScipyOptimizer()
prob.driver.options['optimizer'] = 'SLSQP'
prob.driver.options['disp'] = True
# prob.driver.options['tol'] = 1.0e-12

prob.driver.add_desvar('thickness',
                       lower=numpy.ones((num_y)) * 0.001,
                       upper=numpy.ones((num_y)) * 0.25)
prob.driver.add_objective('energy')
prob.driver.add_constraint('weight', upper=1e5)

prob.root.deriv_options['type'] = 'cs' # Use this if you haven't compiled the Fortran
# prob.root.deriv_options['type'] = 'fd' # Use this if you've compiled the Fortran
#
prob.root.deriv_options['form'] = 'central'
prob.root.deriv_options['step_size'] = 1e-10

prob.driver.add_recorder(SqliteRecorder('spatialbeam.db'))

prob.setup()

view_model(prob, outfile="prob1.html", show_browser=False)

# prob.run_once()
# prob.check_partial_derivatives(compact_print=True)

st = time.time()
prob.run()
print "run time: {} secs".format(time.time() - st)
