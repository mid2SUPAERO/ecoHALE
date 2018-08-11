import unittest

from openaerostruct.aerodynamics.eval_mtx import EvalVelMtx
from openaerostruct.utils.testing import run_test, get_default_surfaces


class Test(unittest.TestCase):

    def test(self):
        surfaces = get_default_surfaces()

        comp = EvalVelMtx(surfaces=surfaces, num_eval_points=2, eval_name='test_name')

        run_test(self, comp)


if __name__ == '__main__':
    unittest.main()
