import unittest

import numpy as np

from openmdao.api import Problem
from openmdao.utils.assert_utils import assert_check_partials

from openaerostruct.aerodynamics.convert_velocity import ConvertVelocity
from openaerostruct.utils.testing import run_test, get_default_surfaces


class Test(unittest.TestCase):

    def test(self):
        surfaces = get_default_surfaces()

        comp = ConvertVelocity(surfaces=surfaces)

        run_test(self, comp)

    def test_rotation_option_derivatives(self):
        surfaces = get_default_surfaces()

        comp = ConvertVelocity(surfaces=surfaces, rotational=True)

        prob = Problem()
        prob.model.add_subsystem('comp', comp)
        prob.setup(force_alloc_complex=True)

        prob['comp.omega'] = np.array(([[.3, .4, -.1], [0, np.pi/2, 0]]))
        prob['comp.cg'] = np.array([.1, .6, .4])
        prob['comp.coll_pts'] = np.random.random(prob['comp.coll_pts'].shape)
        prob['comp.beta'] = 15.0
        prob.run_model()

        check = prob.check_partials(compact_print=True, method='cs', step=1e-40)

        assert_check_partials(check)

if __name__ == '__main__':
    unittest.main()
