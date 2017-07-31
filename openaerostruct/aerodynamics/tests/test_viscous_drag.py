import unittest

from openaerostruct.aerodynamics.viscous_drag import ViscousDrag
from openaerostruct.utils.testing import run_test, get_default_surfaces


class Test(unittest.TestCase):

    def test(self):
        surface = get_default_surfaces()[0]

        comp = ViscousDrag(surface=surface, with_viscous=True)

        run_test(self, comp)


if __name__ == '__main__':
    unittest.main()
