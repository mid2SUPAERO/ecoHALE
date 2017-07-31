from __future__ import print_function, division
from numpy.testing import assert_almost_equal, assert_equal

import unittest

from openmdao.api import Problem

from openaerostruct.geometry.monotonic_constraint import MonotonicConstraint

from openaerostruct.utils.testing import run_test


class Test(unittest.TestCase):

    def test(self):

        surface = {'symmetry' : True,
                   'num_y' : 5}

        comp = MonotonicConstraint(var_name='x', surface=surface)

        run_test(self, comp)


if __name__ == '__main__':
    unittest.main()
