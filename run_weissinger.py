""" Example script to run aero-only optimization """

from __future__ import division
import numpy
import sys
import warnings
warnings.filterwarnings("ignore")

from openmdao.api import IndepVarComp, Problem, Group, ScipyOptimizer, SqliteRecorder, pyOptSparseDriver, profile
from geometry import GeometryMesh, gen_crm_mesh, gen_mesh
from transfer import TransferDisplacements, TransferLoads
from weissinger import WeissingerStates, WeissingerFunctionals, get_mesh_data
from openmdao.devtools.partition_tree_n2 import view_tree

numpy.random.seed(12345)

# Create the mesh with 2 inboard points and 3 outboard points
num_x = 3
mesh = gen_crm_mesh(n_points_inboard=4, n_points_outboard=6, num_x=num_x)
num_y = mesh.shape[1]

# Define the aircraft properties
execfile('CRM.py')

if sys.argv[1].startswith('00'):
    num_x = 3
    num_y = 81
    span = 5.
    chord = 1.
    amt_of_cos = 0.
    mesh = gen_mesh(num_x, num_y, span, chord, amt_of_cos)
    num_twist = numpy.max([int((num_y - 1) / 5), 5])

    mesh = mesh.reshape(-1, mesh.shape[-1])

    translate = numpy.zeros((mesh.shape))
    translate[:, 1] = numpy.max(mesh[:, 1])
    mesh_1 = mesh + translate
    mesh_2 = mesh - translate
    mesh = numpy.vstack((mesh_1, mesh_2))

    mesh_ind = numpy.atleast_2d(numpy.array([num_x, num_y]))
    mesh_ind = numpy.vstack((mesh_ind, mesh_ind))
    mesh_ind = get_mesh_data(mesh_ind)

else:
    num_x = 3
    num_y = 161
    span = 10.
    chord = 1.
    amt_of_cos = 0.
    mesh = gen_mesh(num_x, num_y, span, chord, amt_of_cos)
    num_twist = numpy.max([int((num_y - 1) / 5), 5])

    mesh = mesh.reshape(-1, mesh.shape[-1])
    mesh_ind = numpy.atleast_2d(numpy.array([num_x, num_y]))
    mesh_ind = get_mesh_data(mesh_ind)


disp = numpy.zeros((num_y, 6))

root = Group()

des_vars = [
    ('twist', numpy.zeros(num_twist)),
    ('dihedral', 0.),
    ('sweep', 0.),
    ('span', span),
    ('taper', 1.),
    ('v', v),
    ('alpha', alpha),
    ('rho', rho),
    ('disp', numpy.zeros((num_y, 6)))
]


root.add('des_vars',
         IndepVarComp(des_vars),
         promotes=['*'])
root.add('mesh',
         GeometryMesh(mesh, mesh_ind, num_twist),
         promotes=['*'])
root.add('def_mesh',
         TransferDisplacements(mesh_ind),
         promotes=['*'])
root.add('weissingerstates',
         WeissingerStates(mesh_ind),
         promotes=['*'])
root.add('weissingerfuncs',
         WeissingerFunctionals(mesh_ind, CL0, CD0, num_twist),
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
    prob.driver.opt_settings = {'Major optimality tolerance': 1.0e-8,
                                'Major feasibility tolerance': 1.0e-8}

prob.driver.add_desvar('twist', lower=-10., upper=15., scaler=1e0)
# prob.driver.add_desvar('alpha', lower=-10., upper=10.)
# prob.driver.add_desvar('sweep', lower=-10., upper=30.)
# prob.driver.add_desvar('dihedral', lower=-10., upper=20.)
# prob.driver.add_desvar('taper', lower=.5, upper=2.)
prob.driver.add_objective('CD', scaler=1e4)
prob.driver.add_constraint('CL', equals=0.5)
# setup data recording
prob.driver.add_recorder(SqliteRecorder('weissinger.db'))

# profile.setup(prob)
# profile.start()

prob.root.deriv_options['type'] = 'fd'
prob.setup()

view_tree(prob, outfile="aero.html", show_browser=False)

import time
st = time.time()
prob.run_once()
if sys.argv[1].startswith('0'):
    # prob.check_partial_derivatives(compact_print=True)
    # prob.check_total_derivatives()
    print "run time", time.time() - st
    print
    print 'alpha', prob['alpha'], "; L", prob['L'], "; D", prob['D'], "; num", num_y
    print 'alpha', prob['alpha'], "; CL", prob['CL'], "; CD", prob['CD'], "; num", num_y
    print
    print 'L/D', prob['L'] / prob['D']
elif sys.argv[1] == '1':
    st = time.time()
    prob.run()
    print 'alpha', prob['alpha'], "; CL", prob['CL'], "; CD", prob['CD'], "; num", num_y
    print
    print "run time", time.time() - st
    print
    print 'L/D', prob['L'] / prob['D']
