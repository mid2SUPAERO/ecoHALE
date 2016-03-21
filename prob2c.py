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

from model_helpers import view_tree
from gs_newton import HybridGSNewton

# Create the mesh with 2 inboard points and 3 outboard points
mesh = mesh_gen(n_points_inboard=2, n_points_outboard=3)
num_y = mesh.shape[1]
r = radii(mesh)
t = r/10

# Define the aircraft properties
execfile('CRM.py')

# Define the material properties
execfile('aluminum.py')

# Create the top-level system
root = Group()

# Define the independent variables
indep_vars = [
    ('span', span),
    ('twist', numpy.zeros(num_y)), 
    ('v', v),
    ('alpha', alpha), 
    ('rho', rho),
    ('r', r),  
    ('t', t), 
]

root.add('indep_vars', 
         IndepVarComp(indep_vars), 
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

#######################################################
# Newton solver on the root group (comment out when using newton on coupled)
#######################################################

# root.nl_solver = Newton()
# root.nl_solver.options['iprint'] = 1
# root.nl_solver.line_search.options['iprint'] = 1
# root.ln_solver = ScipyGMRES()
# root.ln_solver.options['iprint'] = 1
# root.ln_solver.preconditioner = LinearGaussSeidel()
# coupled.ln_solver.options['maxiter'] = 2
# coupled.weissingerstates.ln_solver = LinearGaussSeidel()
# coupled.spatialbeamstates.ln_solver = LinearGaussSeidel()

#######################################################
# Newton Solver just on the coupled group (comment out when using newton root)
#######################################################
coupled.nl_solver = Newton()
coupled.nl_solver.options['iprint'] = 1
coupled.nl_solver.line_search.options['iprint'] = 1

#######################################################
# Linear Solver Options for Newton
#######################################################

# # Linear Gauss Seidel Solver
# coupled.ln_solver = LinearGaussSeidel()
# coupled.ln_solver.options['maxiter'] = 100

# Krylov Solver - No preconditioning
# coupled.ln_solver = ScipyGMRES()
# coupled.ln_solver.options['iprint'] = 1

# Krylov Solver - LNGS preconditioning
coupled.ln_solver = ScipyGMRES()
coupled.ln_solver.options['iprint'] = 1
coupled.ln_solver.preconditioner = LinearGaussSeidel()
coupled.weissingerstates.ln_solver = LinearGaussSeidel()
coupled.spatialbeamstates.ln_solver = LinearGaussSeidel()

#######################################################

# Direct Inversion Solver
# coupled.ln_solver = DirectSolver()
    
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
coupled.nl_solver.options['iprint'] = 1 # makes OpenMDAO print out solver convergence data
coupled.ln_solver.options['iprint'] = 1 # makes OpenMDAO print out solver convergence data


prob.driver.add_recorder(SqliteRecorder('prob1c.db'))

prob.setup()
# view_tree(prob, outfile="aerostruct_n2.html", show_browser=True) # generate the n2 diagram diagram


st = time.time()
prob.run_once()
print "runtime: ", time.time() - st


