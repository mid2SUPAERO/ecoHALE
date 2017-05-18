import unittest

from openaerostruct.aerodynamics.forces import Forces
from openaerostruct.utils.testing import run_test, get_default_prob_dict, get_default_surfaces


class Test(unittest.TestCase):

    def test(self):
        surfaces = get_default_surfaces()
        prob_dict = get_default_prob_dict()

        comp = Forces(surfaces=surfaces)

        run_test(self, comp)


if __name__ == '__main__':
    unittest.main()
