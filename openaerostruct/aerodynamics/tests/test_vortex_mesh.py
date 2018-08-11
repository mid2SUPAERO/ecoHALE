import unittest

from openaerostruct.aerodynamics.vortex_mesh import VortexMesh
from openaerostruct.utils.testing import run_test, get_default_surfaces

import numpy as np
np.set_printoptions(linewidth=200)


class Test(unittest.TestCase):

    def test(self):
        surfaces = get_default_surfaces()

        comp = VortexMesh(surfaces=surfaces)

        run_test(self, comp)


if __name__ == '__main__':
    unittest.main()
