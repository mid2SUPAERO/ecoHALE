from __future__ import print_function, division
from numpy.testing import assert_almost_equal, assert_equal

import unittest

from openmdao.api import Problem

from openaerostruct.geometry.monotonic_constraint import MonotonicConstraint

from openaerostruct.utils.testing import run_test, get_default_prob_dict, get_default_surfaces


class Test(unittest.TestCase):

    def test(self):
        surfaces = get_default_surfaces()

        comp = MonotonicConstraint(var_name='x', surface=surfaces[0])

        run_test(self, comp)


if __name__ == '__main__':
    unittest.main()
