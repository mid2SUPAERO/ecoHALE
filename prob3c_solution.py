from __future__ import division
import numpy
import sys
import time

from openmdao.api import IndepVarComp, Problem, Group, ScipyOptimizer, Newton, ScipyGMRES, LinearGaussSeidel, NLGaussSeidel, SqliteRecorder, DirectSolver
from geometry import GeometryMesh, gen_crm_mesh, get_mesh_data
from transfer import TransferDisplacements, TransferLoads
from vlm import VLMStates, VLMFunctionals
from spatialbeam import SpatialBeamStates, SpatialBeamFunctionals, radii
from materials import MaterialsTube
from functionals import FunctionalBreguetRange, FunctionalEquilibrium

from openmdao.devtools.partition_tree_n2 import view_tree
from gs_newton import HybridGSNewton

try:
    from openmdao.api import pyOptSparseDriver
    SNOPT = True
except:
    SNOPT = False

############################################################
# Change mesh size here
############################################################
# Create the mesh with 2 inboard points and 3 outboard points
mesh = gen_crm_mesh(n_points_inboard=2, n_points_outboard=3)
num_x, num_y, _ = mesh.shape
num_twist = numpy.max([int((num_y - 1) / 5), 5])
r = radii(mesh)
t = r/10

mesh = mesh.reshape(-1, mesh.shape[-1])
aero_ind = numpy.atleast_2d(numpy.array([num_x, num_y]))
aero_ind = get_mesh_data(aero_ind)

# Define the aircraft properties
execfile('CRM.py')

# Define the material properties
execfile('aluminum.py')

# Create the top-level system
root = Group()

# Define the independent variables
indep_vars = [
    ('span', span),
    ('twist', numpy.zeros(num_twist)),
    ('v', v),
    ('alpha', alpha),
    ('rho', rho),
    ('r', r),
    ('t', t),
    ('aero_ind', aero_ind)
]

indep_vars_comp = IndepVarComp(indep_vars)
tube_comp = MaterialsTube(aero_ind)

mesh_comp = GeometryMesh(mesh, aero_ind, num_twist)
spatialbeamstates_comp = SpatialBeamStates(aero_ind, E, G)
def_mesh_comp = TransferDisplacements(aero_ind)
vlmstates_comp = VLMStates(aero_ind)
loads_comp = TransferLoads(aero_ind)

vlmfuncs_comp = VLMFunctionals(aero_ind, CL0, CD0, num_twist)
spatialbeamfuncs_comp = SpatialBeamFunctionals(aero_ind, E, G, stress, mrho)
fuelburn_comp = FunctionalBreguetRange(W0, CT, a, R, M, aero_ind)
eq_con_comp = FunctionalEquilibrium(W0, aero_ind)

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
coupled.add('vlmstates',
    vlmstates_comp,
    promotes=["*"])
coupled.add('loads',
    loads_comp,
    promotes=["*"])

## Hybrid NLGS-Newton on the coupled group
coupled.nl_solver = HybridGSNewton()
coupled.nl_solver.nlgs.options['iprint'] = 1
coupled.nl_solver.nlgs.options['maxiter'] = 5
coupled.nl_solver.newton.options['atol'] = 1e-6
coupled.nl_solver.newton.options['rtol'] = 1e-6
coupled.nl_solver.newton.options['iprint'] = 1


# Krylov Solver - LNGS preconditioning
coupled.ln_solver = ScipyGMRES()
coupled.ln_solver.options['iprint'] = 1
coupled.ln_solver.preconditioner = LinearGaussSeidel()
coupled.vlmstates.ln_solver = LinearGaussSeidel()
coupled.spatialbeamstates.ln_solver = LinearGaussSeidel()


# adds the MDA to root (do not remove!)
root.add('coupled',
         coupled,
         promotes=['*'])

# Add functional components here
root.add('vlmfuncs',
        vlmfuncs_comp,
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

# prob.root.deriv_options['type'] = 'cs'
# prob.root.deriv_options['form'] = 'central'

prob.driver = ScipyOptimizer()
prob.driver.options['optimizer'] = 'SLSQP'
prob.driver.options['disp'] = True
prob.driver.options['tol'] = 1.0e-3
prob.driver.options['maxiter'] = 40

if SNOPT:
    prob.driver = pyOptSparseDriver()
    prob.driver.options['optimizer'] = "SNOPT"
    prob.driver.opt_settings = {'Major optimality tolerance': 1.0e-7,
                                'Major feasibility tolerance': 1.0e-7}

prob.driver.add_recorder(SqliteRecorder('aerostruct.db'))

###############################################################
# Add design vars
###############################################################
prob.driver.add_desvar('twist',lower= -10.,
                       upper=10., scaler=1e0)
prob.driver.add_desvar('alpha', lower=-10., upper=10., scaler=1e0)
prob.driver.add_desvar('t',
                       lower= 0.003,
                       upper= 0.25, scaler=1000)

###############################################################
# Add constraints, and objectives
###############################################################
prob.driver.add_objective('fuelburn')
prob.driver.add_constraint('failure', upper=0.0)
prob.driver.add_constraint('eq_con', equals=0.0)

prob.setup()
# view_tree(prob, outfile="my_aerostruct_n2.html", show_browser=True) # generate the n2 diagram diagram

# always need to run before you compute derivatives!
prob.run()

print 'Optimized twist:', prob['twist']
print 'Optimized alpha:', prob['alpha']
print 'Optimized thicknesses:', prob['t']
