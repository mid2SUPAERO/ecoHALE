import unittest

from openaerostruct.structures.fem import FEM
from openaerostruct.utils.testing import run_test, get_default_surfaces


class Test(unittest.TestCase):

    def test(self):
        surface = get_default_surfaces()[0]

        ny = surface['mesh'].shape[1]
        comp = FEM(size= ny * 6)

        run_test(self, comp)


if __name__ == '__main__':
    unittest.main()
