import unittest

from openaerostruct.aerodynamics.convert_velocity import ConvertVelocity
from openaerostruct.utils.testing import run_test, get_default_surfaces


class Test(unittest.TestCase):

    def test(self):
        surfaces = get_default_surfaces()

        comp = ConvertVelocity(surfaces=surfaces)

        run_test(self, comp)


if __name__ == '__main__':
    unittest.main()
