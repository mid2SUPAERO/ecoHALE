""" Example script to run aero-only optimization """

from __future__ import division
import numpy
import sys

from openmdao.api import IndepVarComp, Problem, Group, ScipyOptimizer, SqliteRecorder, pyOptSparseDriver, profile
from geometry import GeometryMesh, mesh_gen, LinearInterp
from transfer import TransferDisplacements, TransferLoads
from weissinger import WeissingerStates, WeissingerFunctionals
from openmdao.devtools.partition_tree_n2 import view_tree

numpy.random.seed(12345)

# Create the mesh with 2 inboard points and 3 outboard points
mesh = mesh_gen(n_points_inboard=4, n_points_outboard=51)
num_y = mesh.shape[1]
num_twist = 3

# Define the aircraft properties
execfile('CRM.py')

if 1:
    num_x = 11
    num_y = 31
    # num_twist = int((num_y - 1) / 5)
    span = 10.
    chord = 1.
    mesh = numpy.zeros((num_x, num_y, 3))
    ny2 = (num_y + 1) / 2
    half_wing = numpy.zeros((ny2))
    beta = numpy.linspace(0, numpy.pi/2, ny2)

    # mixed spacing with w as a weighting factor
    cosine = .5 * numpy.cos(beta)**1 #  cosine spacing
    uniform = numpy.linspace(0, .5, ny2)[::-1] #  uniform spacing
    w = 0
    half_wing = cosine * w + (1 - w) * uniform

    # # concentrated nodes in center of both sides of wing
    # ny3 = (ny2 - 1) / 3
    # half_wing = numpy.hstack((
    #                          numpy.linspace(0, .2, ny3+1, endpoint=False),
    #                          numpy.linspace(.2, .3, ny3, endpoint=False),
    #                          numpy.linspace(.3, .5, ny3)))
    # half_wing = half_wing[::-1]

    full_wing = numpy.hstack((-half_wing[:-1], half_wing[::-1])) * span
    chords = numpy.sqrt(1 - half_wing**2/(span/2)**2) * chord/2
    chords[0] += 1e-5
    # chords = numpy.max(chords) * numpy.linspace(1, .2, ny2)
    chords = numpy.hstack((chords[:-1], chords[::-1]))

    for ind_x in xrange(num_x):
        for ind_y in xrange(num_y):
            mesh[ind_x, ind_y, :] = [ind_x / (num_x-1) * chord, full_wing[ind_y], 0] # straight elliptical spacing
            # mesh[ind_x, ind_y, :] = [(-1)**(ind_x+1) * chords[ind_y], full_wing[ind_y], 0] # elliptical chord


disp = numpy.zeros((num_y, 6))

root = Group()

des_vars = [
    # ('twist', numpy.zeros(num_twist) * 10 * numpy.random.rand(num_twist)),
    ('twist', numpy.array([0., 0, 0.])),
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
         GeometryMesh(mesh, num_twist),
         promotes=['*'])
root.add('def_mesh',
         TransferDisplacements(num_x, num_y),
         promotes=['*'])
root.add('weissingerstates',
         WeissingerStates(num_x, num_y),
         promotes=['*'])
root.add('weissingerfuncs',
         WeissingerFunctionals(num_x, num_y, CL0, CD0),
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

prob.driver.add_desvar('twist',lower=-10., upper=15., scaler=1e0)
# prob.driver.add_desvar('alpha', lower=-10., upper=10.)
prob.driver.add_objective('CD', scaler=1e4)
prob.driver.add_constraint('CL', equals=0.5)
# setup data recording
prob.driver.add_recorder(SqliteRecorder('weissinger.db'))

profile.setup(prob)
profile.start()

prob.root.deriv_options['type'] = 'fd'

prob.setup()
view_tree(prob, outfile="aerostruct.html", show_browser=False)

import time
st = time.time()
prob.run_once()
if sys.argv[1] == '0':
    # prob.check_partial_derivatives(compact_print=True)
    # prob.check_total_derivatives()
    print "run time", time.time() - st
    print
    print 'alpha', prob['alpha'], "; L", prob['L'], "; D", prob['D'], "; num", num_y
    print 'alpha', prob['alpha'], "; CL", prob['CL'], "; CD", prob['CD'], "; num", num_y
    print
    print 'L/D', prob['L'] / prob['D']
    print
    # print numpy.sum(prob['sec_forces'], axis=0)
    # print prob['sec_forces']
    # print
    # print prob['normals']
    # print
elif sys.argv[1] == '1':
    st = time.time()
    prob.run()
    print 'alpha', prob['alpha'], "; CL", prob['CL'], "; CD", prob['CD'], "; num", num_y
    print prob['twist']
    print
    print "run time", time.time() - st
    norm = prob['normals']
    print norm
