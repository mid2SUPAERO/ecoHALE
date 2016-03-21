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

############################################################
# Change mesh size here
############################################################
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

indep_vars_comp = IndepVarComp(indep_vars)
tube_comp = MaterialsTube(num_y)

mesh_comp = GeometryMesh(mesh)
spatialbeamstates_comp = SpatialBeamStates(num_y, E, G)
def_mesh_comp = TransferDisplacements(num_y)
weissingerstates_comp = WeissingerStates(num_y)
loads_comp = TransferLoads(num_y)

weissingerfuncs_comp = WeissingerFunctionals(num_y, CL0, CD0)
spatialbeamfuncs_comp = SpatialBeamFunctionals(num_y, E, G, stress, mrho)
fuelburn_comp = FunctionalBreguetRange(W0, CT, a, R, M)
eq_con_comp = FunctionalEquilibrium(W0)

root.add('indep_vars', 
         indep_vars_comp,
         promotes=['*'])
root.add('tube',
         tube_comp,
         promotes=['*'])

# Add components to the MDA here
coupled = Group()
coupled.add('mesh', 
    mesh_comp, 
    promotes=["*"])
coupled.add('spatialbeamstates', 
    spatialbeamstates_comp, 
    promotes=["*"])
coupled.add('def_mesh', 
    def_mesh_comp, 
    promotes=["*"])
coupled.add('weissingerstates', 
    weissingerstates_comp, 
    promotes=["*"])
coupled.add('loads', 
    loads_comp, 
    promotes=["*"])


############################################################
# Comment/uncomment these solver blocks to try different 
# nonlinear solver methods
############################################################

## Nonlinear Gauss Seidel 
coupled.nl_solver = NLGaussSeidel()   
coupled.nl_solver.options['iprint'] = 1
coupled.nl_solver.options['atol'] = 1e-5
coupled.nl_solver.options['rtol'] = 1e-12

## Newton Solver
# coupled.nl_solver = Newton()
# coupled.nl_solver.options['iprint'] = 1
# coupled.nl_solver.line_search.options['iprint'] = 1
# coupled.ln_solver = ScipyGMRES()
# coupled.ln_solver.options['iprint'] = 1
# coupled.ln_solver.preconditioner = LinearGaussSeidel()
# coupled.weissingerstates.ln_solver = LinearGaussSeidel()
# coupled.spatialbeamstates.ln_solver = LinearGaussSeidel()
    
## Hybrid NLGS-Newton
# coupled.nl_solver = HybridGSNewton()   
# coupled.nl_solver.nlgs.options['iprint'] = 1
# coupled.nl_solver.nlgs.options['maxiter'] = 5
# coupled.nl_solver.newton.options['atol'] = 1e-7
# coupled.nl_solver.newton.options['rtol'] = 1e-7
# coupled.nl_solver.newton.options['iprint'] = 1


# linear solver configuration
coupled.ln_solver = ScipyGMRES()
coupled.ln_solver.options['iprint'] = 1
coupled.ln_solver.preconditioner = LinearGaussSeidel()
coupled.weissingerstates.ln_solver = LinearGaussSeidel()
coupled.spatialbeamstates.ln_solver = LinearGaussSeidel()
    
# adds the MDA to root (do not remove!)
root.add('coupled',
         coupled,
         promotes=['*'])

# Add functional components here
root.add('weissingerfuncs', 
        weissingerfuncs_comp, 
        promotes=['*'])
root.add('spatialbeamfuncs', 
        spatialbeamfuncs_comp, 
        promotes=['*'])
root.add('fuelburn', 
        fuelburn_comp, 
        promotes=['*'])
root.add('eq_con', 
        eq_con_comp, 
        promotes=['*'])

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



