from __future__ import print_function, division
import numpy as np

import unittest

from openmdao.api import Problem, Group

from openaerostruct.geometry.geometry_mesh import GeometryMesh

from openaerostruct.utils.testing import run_test, get_default_prob_dict, get_default_surfaces


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

        comp = GeometryMesh(surface=surface, desvars={'taper': .4})

        run_test(self, comp)


if __name__ == '__main__':
    unittest.main()
