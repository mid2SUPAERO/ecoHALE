from __future__ import division
import numpy
import sys

from openmdao.api import IndepVarComp, Problem, Group, ScipyOptimizer, Newton, ScipyGMRES
from geometry import GeometryMesh
from transfer import TransferDisplacements, TransferLoads
from weissinger import WeissingerGroup
from spatialbeam import SpatialBeamGroup


num_y = 3
span = 232.02
chord = 39.37
cons = numpy.array([int((num_y-1)/2)])

v = 200.
alpha = 3.
rho = 1.225

E = 200.e9
G = 30.e9
r = 0.3 * numpy.ones(num_y-1)
t = 0.02 * numpy.ones(num_y-1)

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
            SpatialBeamGroup(num_y, cons, E, G),
            promotes=['*'])
coupled.nl_solver = Newton()
coupled.ln_solver = ScipyGMRES()

root.add('coupled',
         coupled,
         promotes=['*'])

prob = Problem()
prob.root = root

prob.setup()
prob.run_once()
