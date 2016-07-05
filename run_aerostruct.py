""" !!! Note that this is currently non-functioning.
Example script to run aerostructural optimization.
Call as `python run_aerostruct.py` to check derivatives, or
call as `python run_aerostruct.py 0` to run a single analysis, or
call as `python run_aerostruct.py 1` to perform optimization. """

from __future__ import division
import numpy
import sys
from time import time

from openmdao.api import IndepVarComp, Problem, Group, ScipyOptimizer, Newton, ScipyGMRES, LinearGaussSeidel, NLGaussSeidel, SqliteRecorder, profile
from geometry import GeometryMesh, gen_crm_mesh, gen_mesh, get_mesh_data
from transfer import TransferDisplacements, TransferLoads
from weissinger import WeissingerStates, WeissingerFunctionals
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

if 1:
    # Use the CRM mesh
    # Create the mesh with 2 inboard points and 3 outboard points
    mesh = gen_crm_mesh(n_points_inboard=2, n_points_outboard=3)
    num_x, num_y, _ = mesh.shape
    num_twist = 5
else:
    # Use a rectangular wing mesh
    num_x = 2
    num_y = 11
    span = 10.
    chord = 1.
    cosine_spacing = 1.
    mesh = gen_mesh(num_x, num_y, span, chord, cosine_spacing)
    num_twist = numpy.max([int((num_y - 1) / 5), 5])

r = radii(mesh)
t = r/15

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

root.add('indep_vars',
         IndepVarComp(indep_vars),
         promotes=['*'])
root.add('tube',
         MaterialsTube(aero_ind),
         promotes=['*'])

coupled = Group()
coupled.add('mesh',
            GeometryMesh(mesh, aero_ind, num_twist),
            promotes=['*'])
coupled.add('def_mesh',
            TransferDisplacements(aero_ind),
            promotes=['*'])
coupled.add('weissingerstates',
            WeissingerStates(aero_ind),
            promotes=['*'])
coupled.add('loads',
            TransferLoads(aero_ind),
            promotes=['*'])
coupled.add('spatialbeamstates',
            SpatialBeamStates(aero_ind, E, G),
            promotes=['*'])

coupled.nl_solver = Newton()
coupled.nl_solver.options['iprint'] = 1
coupled.ln_solver = ScipyGMRES()
coupled.ln_solver.options['iprint'] = 1
coupled.ln_solver.preconditioner = LinearGaussSeidel()
coupled.weissingerstates.ln_solver = LinearGaussSeidel()
coupled.spatialbeamstates.ln_solver = LinearGaussSeidel()

coupled.nl_solver = NLGaussSeidel()   ### Uncomment this out to use NLGS
coupled.nl_solver.options['iprint'] = 1
coupled.nl_solver.options['atol'] = 1e-5
coupled.nl_solver.options['rtol'] = 1e-12

coupled.nl_solver = HybridGSNewton()   ### Uncomment this out to use Hybrid GS Newton
coupled.nl_solver.nlgs.options['iprint'] = 1
coupled.nl_solver.nlgs.options['maxiter'] = 5
coupled.nl_solver.newton.options['atol'] = 1e-7
coupled.nl_solver.newton.options['rtol'] = 1e-7
coupled.nl_solver.newton.options['iprint'] = 1

root.add('coupled',
         coupled,
         promotes=['*'])
root.add('weissingerfuncs',
         WeissingerFunctionals(aero_ind, CL0, CD0, num_twist),
         promotes=['*'])
root.add('spatialbeamfuncs',
         SpatialBeamFunctionals(aero_ind, E, G, stress, mrho),
         promotes=['*'])
root.add('fuelburn',
         FunctionalBreguetRange(W0, CT, a, R, M, aero_ind),
         promotes=['*'])
root.add('eq_con',
         FunctionalEquilibrium(W0, aero_ind),
         promotes=['*'])

prob = Problem()
prob.root = root
prob.print_all_convergence()

prob.driver = ScipyOptimizer()
prob.driver.options['optimizer'] = 'SLSQP'
prob.driver.options['disp'] = True
prob.driver.options['tol'] = 1.0e-8

if SNOPT:
    prob.driver = pyOptSparseDriver()
    prob.driver.options['optimizer'] = "SNOPT"
    prob.driver.opt_settings = {'Major optimality tolerance': 1.0e-7,
                                'Major feasibility tolerance': 1.0e-7}

prob.driver.add_desvar('twist',lower= -10.,
                       upper=10., scaler=1e0)
#prob.driver.add_desvar('alpha', lower=-10., upper=10., scaler=1000)
prob.driver.add_desvar('t',
                       lower= 0.001,
                       upper= 0.25, scaler=1000)
prob.driver.add_objective('fuelburn')
prob.driver.add_constraint('failure', upper=0.0)
prob.driver.add_constraint('eq_con', equals=0.0)

prob.driver.add_recorder(SqliteRecorder('aerostruct.db'))

profile.setup(prob)
profile.start()

prob.setup()
view_tree(prob, outfile="aerostruct.html", show_browser=False)

st = time()
prob.run_once()
if len(sys.argv) == 1:
    pass
elif sys.argv[1] == '0':
    prob.check_partial_derivatives(compact_print=True)
elif sys.argv[1] == '1':
    prob.run()
print "runtime: ", time() - st
