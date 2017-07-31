import unittest

from openaerostruct.functionals.moment_coefficient import MomentCoefficient
from openaerostruct.utils.testing import run_test


class Test(unittest.TestCase):

    def test(self):
        wing_dict = {'name' : 'wing',
                     'num_y' : 7,
                     'num_x' : 2,
                     'symmetry' : True}
        tail_dict = {'name' : 'tail',
                     'num_y' : 5,
                     'num_x' : 3,
                     'symmetry' : False}
                     
        surfaces = [wing_dict, tail_dict]

        comp = MomentCoefficient(surfaces=surfaces)

        run_test(self, comp)


if __name__ == '__main__':
    unittest.main()
