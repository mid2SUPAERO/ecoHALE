from __future__ import print_function, division
import numpy as np

import unittest

from openmdao.api import Problem, Group
from openaerostruct.structures.weight import Weight

from run_classes import OASProblem

class Test(unittest.TestCase):

    def test(self):
        OASprob = OASProblem({'type' : 'struct'})
        OASprob.add_surface()
        surface = OASprob.surfaces[0]

        prob = Problem(model=Weight(surface=surface))
        prob.setup()

        prob.run_model()

        data = prob.check_partial_derivs(compact_print=True)

        new_dict = {}
        for key1 in data.keys():
            for key2 in data[key1].keys():
                for key3 in data[key1][key2].keys():
                    if 'rel' in key3:
                        error = np.linalg.norm(data[key1][key2][key3])
                        new_key = key1+'_'+key2[0]+'_'+key2[1]+'_'+key3
                        new_dict.update({new_key : error})

        for key in new_dict.keys():
            error = new_dict[key]
            if not np.isnan(error):
                self.assertAlmostEqual(0., error, places=2)


if __name__ == '__main__':
    unittest.main()
