import unittest

from openaerostruct.functionals.total_lift_drag import TotalLiftDrag
from openaerostruct.utils.testing import run_test, get_default_prob_dict, get_default_surfaces


class Test(unittest.TestCase):

    def test(self):
        surfaces = get_default_surfaces()
        prob_dict = get_default_prob_dict()

        comp = TotalLiftDrag(surfaces=surfaces, prob_dict=prob_dict)

        run_test(self, comp)

        prob_dict['S_ref_total'] = 5

        comp = TotalLiftDrag(surfaces=surfaces, prob_dict=prob_dict)

        run_test(self, comp)


if __name__ == '__main__':
    unittest.main()
