import unittest

from openaerostruct.functionals.equilibrium import Equilibrium
from openaerostruct.utils.testing import run_test


class Test(unittest.TestCase):

    def test(self):
        surfaces = [{'name' : 'wing'}, {'name' : 'tail'}]

        comp = Equilibrium(surfaces=surfaces)

        run_test(self, comp)


if __name__ == '__main__':
    unittest.main()
