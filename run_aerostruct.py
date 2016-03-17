from __future__ import division
import numpy
import sys
import time

from openmdao.api import IndepVarComp, Problem, Group, ScipyOptimizer, Newton, ScipyGMRES, LinearGaussSeidel, NLGaussSeidel
from geometry import GeometryMesh
from transfer import TransferDisplacements, TransferLoads
from weissinger import WeissingerGroup
from spatialbeam import SpatialBeamGroup
from functionals import FunctionalBreguetRange, FunctionalEquilibrium
#from gs_newton import HybridGSNewton

num_y = 5
span = 60
chord = 4
cons = numpy.array([int((num_y-1)/2)])

W0 = 5.e5
CT = 0.01
a = 200
M = 0.75
R = 2000

v = a * M
alpha = 3.
rho = 1.225

E = 200.e9
G = 30.e9
stress = 20.e6
mrho = 3.e3
r = 0.3 * numpy.ones(num_y-1)
t = 0.05 * numpy.ones(num_y-1)

root = Group()

root.add('twist',
         IndepVarComp('twist', numpy.zeros((num_y))),
         promotes=['*'])
root.add('v',
         IndepVarComp('v', v),
         promotes=['*'])
root.add('alpha',
         IndepVarComp('alpha', alpha),
         promotes=['*'])
root.add('rho',
         IndepVarComp('rho', rho),
         promotes=['*'])
root.add('r',
         IndepVarComp('r', r),
         promotes=['*'])
root.add('t',
         IndepVarComp('t', t),
         promotes=['*'])

coupled = Group()
coupled.add('mesh',
            GeometryMesh(num_y, span, chord),
            promotes=['*'])
coupled.add('def_mesh',
            TransferDisplacements(num_y),
            promotes=['*'])
coupled.add('weissinger',
            WeissingerGroup(num_y),
            promotes=['*'])
coupled.add('loads',
            TransferLoads(num_y),
            promotes=['*'])
coupled.add('spatialbeam',
            SpatialBeamGroup(num_y, cons, E, G, stress, mrho),
            promotes=['*'])

coupled.nl_solver = Newton()
coupled.nl_solver.options['iprint'] = 1
coupled.nl_solver.line_search.options['iprint'] = 1
coupled.ln_solver = ScipyGMRES()
coupled.ln_solver.options['iprint'] = 1
coupled.ln_solver.preconditioner = LinearGaussSeidel()
coupled.weissinger.ln_solver = LinearGaussSeidel()
coupled.spatialbeam.ln_solver = LinearGaussSeidel()

coupled.nl_solver = NLGaussSeidel()   ### Uncomment this out to use NLGS
coupled.nl_solver.options['iprint'] = 1
coupled.nl_solver.options['atol'] = 1e-12
coupled.nl_solver.options['rtol'] = 1e-12

# coupled.nl_solver = HybridGSNewton()   ### Uncomment this out to use Hybrid GS Newton
# coupled.nl_solver.nlgs.options['iprint'] = 1
# coupled.nl_solver.nlgs.options['maxiter'] = 3
# coupled.nl_solver.newton.options['rtol'] = 1e-9


root.add('coupled',
         coupled,
         promotes=['*'])
root.add('fuelburn',
         FunctionalBreguetRange(W0, CT, a, R, M),
         promotes=['*'])
root.add('eq_con',
         FunctionalEquilibrium(W0),
         promotes=['*'])

prob = Problem()
prob.root = root
prob.print_all_convergence()

prob.driver = ScipyOptimizer()
prob.driver.options['optimizer'] = 'SLSQP'
prob.driver.options['disp'] = True
# prob.driver.options['tol'] = 1.0e-12

prob.driver.add_desvar('twist',lower=numpy.ones((num_y)) * -10.,
                       upper=numpy.ones((num_y)) * 10.)
prob.driver.add_desvar('alpha', lower=-10., upper=10., scaler=100)
prob.driver.add_objective('CD')
prob.driver.add_constraint('CL', equals=0.5)

prob.setup()

if len(sys.argv) == 1:
    st = time.time()
    prob.run_once()
    print "runtime: ", time.time() - st
elif sys.argv[1] == '0':
    prob.run_once()
    prob.check_partial_derivatives(compact_print=True)
elif sys.argv[1] == '1':
    prob.run()
