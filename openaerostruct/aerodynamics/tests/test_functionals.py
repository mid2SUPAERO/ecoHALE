import unittest

from openaerostruct.aerodynamics.functionals import VLMFunctionals
from openaerostruct.utils.testing import run_test, get_default_surfaces


class Test(unittest.TestCase):

    def test(self):
        surfaces = get_default_surfaces()

        comp = VLMFunctionals(surface=surfaces[0])

        run_test(self, comp)


if __name__ == '__main__':
    unittest.main()