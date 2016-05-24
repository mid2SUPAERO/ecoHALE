from __future__ import division
import numpy
import sys
import time

from openmdao.api import IndepVarComp, Problem, Group, ScipyOptimizer, Newton, ScipyGMRES, LinearGaussSeidel, NLGaussSeidel, SqliteRecorder, DirectSolver
from geometry import GeometryMesh, mesh_gen
from transfer import TransferDisplacements, TransferLoads
from weissinger import WeissingerStates, WeissingerFunctionals
from spatialbeam import SpatialBeamStates, SpatialBeamFunctionals, radii
from materials import MaterialsTube
from functionals import FunctionalBreguetRange, FunctionalEquilibrium

from openmdao.devtools.partition_tree_n2 import view_tree
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

## Hybrid NLGS-Newton on the coupled group
coupled.nl_solver = HybridGSNewton()
coupled.nl_solver.nlgs.options['iprint'] = 1
coupled.nl_solver.nlgs.options['maxiter'] = 5
coupled.nl_solver.newton.options['atol'] = 1e-8
coupled.nl_solver.newton.options['rtol'] = 1e-8
coupled.nl_solver.newton.options['iprint'] = 1


# Krylov Solver - LNGS preconditioning
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

prob.setup()
# view_tree(prob, outfile="my_aerostruct_n2.html", show_browser=True) # generate the n2 diagram diagram

# always need to run before you compute derivatives!
prob.run_once()


#############################################################
# Comment out the following code to run analytic derivatives
##############################################################
prob.root.deriv_options['type'] = 'cs'
prob.root.deriv_options['step_type'] = 'relative'
prob.root.deriv_options['form'] = 'forward'
prob.root.deriv_options['step_size'] = 1e-6
#####################################################

print "------------------------------------------------"
print "Solving for Derivatives"
print "------------------------------------------------"
st = time.time()
jac = prob.calc_gradient(['twist','alpha','t'], ['fuelburn'], return_format="dict")
run_time = time.time() - st

print "runtime: ", run_time
print
print "d_fuelburn/d_alpha", jac['fuelburn']['alpha']
print "norm(d_fuelburn/twist)", numpy.linalg.norm(jac['fuelburn']['twist'])
