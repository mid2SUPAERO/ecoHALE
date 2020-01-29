import unittest

from openaerostruct.aerodynamics.lift_drag import LiftDrag
from openaerostruct.utils.testing import run_test, get_default_surfaces


class Test(unittest.TestCase):

    def test(self):
        surfaces = get_default_surfaces()

        comp = LiftDrag(surface=surfaces[0])

        run_test(self, comp)


if __name__ == '__main__':
    unittest.main()
