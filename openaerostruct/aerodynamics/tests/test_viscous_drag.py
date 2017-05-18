import unittest

from openaerostruct.aerodynamics.viscous_drag import ViscousDrag
from openaerostruct.utils.testing import run_test, get_default_surf_dict


class Test(unittest.TestCase):

    def test(self):
        surface = get_default_surf_dict()

        comp = ViscousDrag(surface=surface, with_viscous=True)

        run_test(self, comp)


if __name__ == '__main__':
    unittest.main()
