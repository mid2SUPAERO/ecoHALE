import unittest

from openaerostruct.aerodynamics.assemble_aic import AssembleAIC
from openaerostruct.utils.testing import run_test, get_default_surfaces


class Test(unittest.TestCase):

    def test(self):
        surfaces = get_default_surfaces()

        comp = AssembleAIC(surfaces=surfaces)

        run_test(self, comp)


if __name__ == '__main__':
    unittest.main()
