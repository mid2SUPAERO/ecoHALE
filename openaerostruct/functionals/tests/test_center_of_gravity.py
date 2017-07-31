import unittest

from openaerostruct.functionals.center_of_gravity import CenterOfGravity
from openaerostruct.utils.testing import run_test


class Test(unittest.TestCase):

    def test(self):
        surfaces = [{'name' : 'wing'}, {'name' : 'tail'}]

        comp = CenterOfGravity(surfaces=surfaces)

        run_test(self, comp)


if __name__ == '__main__':
    unittest.main()
