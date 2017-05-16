from __future__ import print_function, division
import numpy as np

import unittest

from openmdao.api import Problem, Group
from openaerostruct.structures.create_rhs import CreateRHS

class Test(unittest.TestCase):

    def test(self):
        OASprob = OASProblem({'type' : 'struct'})
        OASprob.add_surface()
        surface = OASprob.surfaces[0]

        prob = Problem(model=CreateRHS(
            surface=surface))
        prob.setup()

        prob.run_model()

        check = prob.check_partial_derivs(compact_print=True)
        self.assertTrue(
            check['']['forces', 'loads']['rel error'].forward < 1e-6)


if __name__ == '__main__':
    unittest.main()
