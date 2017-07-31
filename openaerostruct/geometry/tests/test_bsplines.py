from __future__ import print_function, division
from numpy.testing import assert_almost_equal, assert_equal

import unittest

from openmdao.api import Problem

from openaerostruct.geometry.bsplines import Bsplines, get_bspline_mtx

from openaerostruct.utils.testing import run_test


class Test(unittest.TestCase):

    def test(self):
        num_cp = 20
        num_pt = 100

        jac = get_bspline_mtx(num_cp, num_pt)

        comp = Bsplines(num_cp=num_cp, num_pt=num_pt, jac=jac, in_name='x', out_name='y')

        run_test(self, comp)


if __name__ == '__main__':
    unittest.main()
