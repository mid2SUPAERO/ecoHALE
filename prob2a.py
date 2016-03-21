from __future__ import division
import numpy
import sys
import time
import pylab as plt

from openmdao.api import IndepVarComp, Problem, Group, ScipyOptimizer, Newton, ScipyGMRES, LinearGaussSeidel, NLGaussSeidel, SqliteRecorder, DirectSolver
from geometry import GeometryMesh, mesh_gen
from transfer import TransferDisplacements, TransferLoads
from weissinger import WeissingerStates, WeissingerFunctionals
from spatialbeam import SpatialBeamStates, SpatialBeamFunctionals, radii
from materials import MaterialsTube
from functionals import FunctionalBreguetRange, FunctionalEquilibrium

from model_helpers import view_tree
from gs_newton import HybridGSNewton
from plot_tools import adjust_spines
import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['axes.edgecolor'] = 'gray'
mpl.rcParams['axes.linewidth'] = 0.5

# control problem size here, by chaning number of mesh points
mesh = mesh_gen(n_points_inboard=2, n_points_outboard=3)

num_y = mesh.shape[1]

cons = numpy.array([int((num_y-1)/2)])

W0 = 1.e5
CT = 0.01
a = 200
M = 0.75
R = 2000

v = a * M
span = 58.7630524 # baseline CRM
alpha = 1.
rho = 1.225

E = 200.e9
G = 30.e9
stress = 20.e6
mrho = 3.e3
r = radii(mesh)
# t = 0.05 * numpy.ones(num_y-1)
t = 0.05 * numpy.ones(num_y-1)
t = r/10

root = Group()


des_vars = [
    ('span', span),
    ('twist', numpy.zeros(num_y)),
    ('v', v),
    ('alpha', alpha),
    ('rho', rho),
    ('r', r),
    ('t', t),
]

root.add('des_vars',
         IndepVarComp(des_vars),
         promotes=['*'])
root.add('tube',
         MaterialsTube(num_y),
         promotes=['*'])

coupled = Group() # add components for MDA to this group
coupled.add('mesh',
            GeometryMesh(mesh),
            promotes=['*'])
coupled.add('def_mesh',
            TransferDisplacements(num_y),
            promotes=['*'])
coupled.add('weissingerstates',
            WeissingerStates(num_y),
            promotes=['*'])
coupled.add('loads',
            TransferLoads(num_y),
            promotes=['*'])
coupled.add('spatialbeamstates',
            SpatialBeamStates(num_y, cons, E, G),
            promotes=['*'])

coupled.nl_solver = Newton()
coupled.nl_solver.options['iprint'] = 1
coupled.nl_solver.options['atol'] = 1e-8
coupled.nl_solver.options['rtol'] = 1e-8

# Krylov Solver - LNGS preconditioning
coupled.ln_solver = ScipyGMRES()
coupled.ln_solver.preconditioner = LinearGaussSeidel()
coupled.weissingerstates.ln_solver = LinearGaussSeidel()
coupled.spatialbeamstates.ln_solver = LinearGaussSeidel()


root.add('coupled',
         coupled,
         promotes=['*'])
root.add('weissingerfuncs',
         WeissingerFunctionals(num_y),
         promotes=['*'])
root.add('spatialbeamfuncs',
         SpatialBeamFunctionals(num_y, E, G, stress, mrho),
         promotes=['*'])
root.add('fuelburn',
         FunctionalBreguetRange(W0, CT, a, R, M),
         promotes=['*'])
root.add('eq_con',
         FunctionalEquilibrium(W0),
         promotes=['*'])

prob = Problem()
prob.root = root

prob.setup()
# view_tree(prob, outfile="my_aerostruct_n2.html", show_browser=True) # generate the n2 diagram diagram

# always need to run before you compute derivatives!
prob.run_once()

prob.root.fd_options['force_fd'] = True
prob.root.fd_options['step_type'] = 'relative'
prob.root.fd_options['form'] = 'forward'
prob.root.fd_options['step_size'] = 1e-6

st = time.time()
jac = prob.calc_gradient(['twist','alpha','t'], ['fuelburn'], return_format="dict")
run_time = time.time() - st

print "runtime: ", run_time
print 
print "d_fuelburn/d_alpha", jac['fuelburn']['alpha']
print "norm(d_fuelburn/twist)", numpy.linalg.norm(jac['fuelburn']['twist'])

