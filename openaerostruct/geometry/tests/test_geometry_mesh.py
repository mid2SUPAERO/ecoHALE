from __future__ import print_function, division
import numpy as np

import unittest

from openmdao.api import Problem, Group

from openaerostruct.geometry.geometry_mesh import GeometryMesh


class Test(unittest.TestCase):

    def test(self):
        surface = {}
        surface['symmetry'] = False
        span = 1
        ny = 5
        mesh = np.zeros((2, ny, 3))
        mesh[0, :, 0] = -1.
        mesh[:, :, 1] = np.linspace(0, span, ny)
        surface['mesh'] = mesh
        surface['name'] = 'wing'
        surface['num_y'] = ny
        surface['initial_geo'] = ['taper']
        surface['span'] = span
        surface['t_over_c'] = .12

        prob = Problem(model=GeometryMesh(
            surface=surface, desvars={'taper': .4}))
        prob.setup()

        prob.run_model()

        prob.check_partial_derivs(compact_print=True)


if __name__ == '__main__':
    unittest.main()
