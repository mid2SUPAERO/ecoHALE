import unittest

from openaerostruct.functionals.breguet_range import BreguetRange
from openaerostruct.utils.testing import run_test


class Test(unittest.TestCase):

    def test(self):
        surfaces = [{'name': 'wing'}]
        prob_dict = {
            'g': 9.81,
            'CT': 9.80665 * 17.e-6,
            'a': 295.4,
            'R': 14.3e6,
            'M': 0.84,
            'W0': 1e6,
            'beta': 0.5,
        }

        comp = BreguetRange(surfaces=surfaces, prob_dict=prob_dict)

        run_test(self, comp)


if __name__ == '__main__':
    unittest.main()
