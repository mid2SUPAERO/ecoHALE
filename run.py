from __future__ import division
import numpy

from openmdao.api import IndepVarComp, Problem, Group
from geometry import Mesh
from weissinger import WeissingerPreproc, WeissingerCirculations


class AeroGroup(Group):

    def __init__(self):
        super(AeroGroup, self).__init__()

        num_y = 3
        span = 50
        chord = 5

        v = 100.
        alpha = 4.
        rho = 1.

        self.add('twist',
                 IndepVarComp('twist', numpy.zeros((num_y))),
                 promotes=['*'])
        self.add('v',
                 IndepVarComp('v', v),
                 promotes=['*'])
        self.add('alpha',
                 IndepVarComp('alpha', alpha),
                 promotes=['*'])
        self.add('mesh',
                 Mesh(num_y, span, chord),
                 promotes=['*'])
        self.add('preproc', WeissingerPreproc(num_y), promotes=['*'])
        self.add('circ', WeissingerCirculations(num_y), promotes=['*'])

        

if __name__ == "__main__":
#    numpy.seterr(all='raise')

    
   top = Problem()
   top.root = AeroGroup()

   top.setup()
   top.run()

   #data = top.check_total_derivatives()
   data = top.check_partial_derivatives()

   print 'What the 2x2 should be:'
   print top.root.circ.mtx
