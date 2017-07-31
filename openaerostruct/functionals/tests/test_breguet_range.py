import unittest

from openaerostruct.functionals.breguet_range import BreguetRange
from openaerostruct.utils.testing import run_test


class Test(unittest.TestCase):

    def test(self):
        surfaces = [{'name' : 'wing'}, {'name' : 'tail'}]

        comp = BreguetRange(surfaces=surfaces)

        run_test(self, comp)
        

if __name__ == '__main__':
    unittest.main()
