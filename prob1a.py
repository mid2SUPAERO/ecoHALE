from __future__ import division
import numpy
import sys
import time

from openmdao.api import IndepVarComp, Problem, Group, ScipyOptimizer, Newton, ScipyGMRES, LinearGaussSeidel, NLGaussSeidel, SqliteRecorder
from geometry import GeometryMesh, mesh_gen
from transfer import TransferDisplacements, TransferLoads
from weissinger import WeissingerStates, WeissingerFunctionals
from spatialbeam import SpatialBeamStates, SpatialBeamFunctionals, radii
from materials import MaterialsTube
from functionals import FunctionalBreguetRange, FunctionalEquilibrium
from lu_group import LUGroup, LUSolver

from model_helpers import view_tree
from gs_newton import HybridGSNewton

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

coupled = Group()


# Nonlinear Gauss Seidel 
coupled.nl_solver = NLGaussSeidel()   
coupled.nl_solver.options['iprint'] = 1
coupled.nl_solver.options['atol'] = 1e-5
coupled.nl_solver.options['rtol'] = 1e-12
    

root.add('coupled',
         coupled,
         promotes=['*'])

# Add functional components here

prob = Problem()
prob.root = root
prob.print_all_convergence() # makes OpenMDAO print out solver convergence data

# change file name to save data from each experiment separately
prob.driver.add_recorder(SqliteRecorder('prob1a.db')) 

prob.setup()
# uncomment this to see an n2 diagram of your problem
# view_tree(prob, outfile="aerostruct_n2.html", show_browser=True) 

st = time.time()
prob.run_once()
print "runtime: ", time.time() - st



