""" Example runscript to perform aerostructural optimization. """

from __future__ import division
import numpy
import sys

from openmdao.api import IndepVarComp, Problem, Group, ScipyOptimizer, Newton, ScipyGMRES, LinearGaussSeidel, NLGaussSeidel, SqliteRecorder

from openmdao.devtools.partition_tree_n2 import view_tree
from geometry import GeometryMesh, Bspline, gen_crm_mesh, gen_mesh, get_inds
from transfer import TransferDisplacements, TransferLoads
from vlm import VLMStates, VLMFunctionals
from spatialbeam import SpatialBeamStates, SpatialBeamFunctionals, radii
from materials import MaterialsTube
from functionals import FunctionalBreguetRange, FunctionalEquilibrium

from gs_newton import HybridGSNewton
from b_spline import get_bspline_mtx


############################################################
# Change mesh size here
############################################################
# Create the mesh with 2 inboard points and 3 outboard points
mesh = gen_crm_mesh(n_points_inboard=2, n_points_outboard=3)
num_x, num_y = mesh.shape[:2]
num_twist = numpy.max([int((num_y - 1) / 5), 5])

r = radii(mesh)
mesh = mesh.reshape(-1, mesh.shape[-1])
aero_ind = numpy.atleast_2d(numpy.array([num_x, num_y]))
fem_ind = [num_y]
aero_ind, fem_ind = get_inds(aero_ind, fem_ind)
num_thickness = num_twist
t = r/10

# Define the aircraft properties
execfile('CRM.py')

# Define the material properties
execfile('aluminum.py')

# Create the top-level system
root = Group()
tot_n_fem = numpy.sum(fem_ind[:, 0])
num_surf = fem_ind.shape[0]
jac_twist = get_bspline_mtx(num_twist, num_y)
jac_thickness = get_bspline_mtx(num_thickness, tot_n_fem-num_surf)

# Define the independent variables
indep_vars = [
    ('span', span),
    ('twist_cp', numpy.zeros(num_twist)),
    ('thickness_cp', numpy.ones(num_thickness)*numpy.max(t)),
    ('v', v),
    ('alpha', alpha),
    ('rho', rho),
    ('r', r),
    ('t', t),
    ('aero_ind', aero_ind)
]

indep_vars_comp = IndepVarComp(indep_vars)
twist_comp = Bspline('twist_cp', 'twist', jac_twist)
thickness_comp = Bspline('thickness_cp', 'thickness', jac_thickness)
tube_comp = MaterialsTube(fem_ind)

mesh_comp = GeometryMesh(mesh, aero_ind)
spatialbeamstates_comp = SpatialBeamStates(aero_ind, fem_ind, E, G)
def_mesh_comp = TransferDisplacements(aero_ind, fem_ind)
vlmstates_comp = VLMStates(aero_ind)
loads_comp = TransferLoads(aero_ind, fem_ind)

vlmfuncs_comp = VLMFunctionals(aero_ind, CL0, CD0)
spatialbeamfuncs_comp = SpatialBeamFunctionals(aero_ind, fem_ind, E, G, stress, mrho)
fuelburn_comp = FunctionalBreguetRange(W0, CT, a, R, M, aero_ind)
eq_con_comp = FunctionalEquilibrium(W0, aero_ind)

root.add('indep_vars',
         indep_vars_comp,
         promotes=['*'])
root.add('twist_bsp',
         twist_comp,
         promotes=['*'])
root.add('thickness_bsp',
         thickness_comp,
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

try:  # Use SNOPT optimizer if installed
    from openmdao.api import pyOptSparseDriver
    prob.driver = pyOptSparseDriver()
    prob.driver.options['optimizer'] = "SNOPT"
    prob.driver.opt_settings = {'Major optimality tolerance': 1.0e-7,
                                'Major feasibility tolerance': 1.0e-7}
except:  # Use SLSQP optimizer if SNOPT not installed
    prob.driver = ScipyOptimizer()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['disp'] = True
    prob.driver.options['tol'] = 1.0e-3
    prob.driver.options['maxiter'] = 40

prob.driver.add_recorder(SqliteRecorder('aerostruct.db'))

###############################################################
# Add design vars
###############################################################
prob.driver.add_desvar('twist_cp',lower= -10.,
                       upper=10., scaler=1e0)
prob.driver.add_desvar('alpha', lower=-10., upper=10., scaler=1e0)
prob.driver.add_desvar('thickness_cp',
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
