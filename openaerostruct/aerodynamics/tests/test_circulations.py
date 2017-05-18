import unittest

from openaerostruct.aerodynamics.circulations import Circulations
from openaerostruct.utils.testing import run_test


class Test(unittest.TestCase):

    def test(self):
        comp = Circulations(size=10)

        run_test(self, comp)


if __name__ == '__main__':
    unittest.main()
