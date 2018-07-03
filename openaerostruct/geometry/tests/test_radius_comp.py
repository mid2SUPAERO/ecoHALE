from __future__ import print_function, division
from numpy.testing import assert_almost_equal, assert_equal

import unittest

from openmdao.api import Problem

from openaerostruct.geometry.radius_comp import RadiusComp
from openaerostruct.utils.testing import run_test, get_default_surfaces


class Test(unittest.TestCase):

    def test(self):
        surface = get_default_surfaces()[0]

        comp = RadiusComp(surface=surface)

        run_test(self, comp)


if __name__ == '__main__':
    unittest.main()
