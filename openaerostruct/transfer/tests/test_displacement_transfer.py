import unittest

from openaerostruct.transfer.displacement_transfer import DisplacementTransfer
from openaerostruct.utils.testing import run_test, get_default_prob_dict, get_default_surfaces


class Test(unittest.TestCase):

    def test(self):
        surfaces = get_default_surfaces()
        prob_dict = get_default_prob_dict()

        comp = DisplacementTransfer(surface=surfaces[0])

        run_test(self, comp)


if __name__ == '__main__':
    unittest.main()
