import unittest

from openaerostruct.structures.non_intersecting_thickness import NonIntersectingThickness
from openaerostruct.utils.testing import run_test, get_default_surf_dict


class Test(unittest.TestCase):

    def test(self):
        surface = get_default_surf_dict()

        comp = NonIntersectingThickness(surface=surface)

        run_test(self, comp)


if __name__ == '__main__':
    unittest.main()
