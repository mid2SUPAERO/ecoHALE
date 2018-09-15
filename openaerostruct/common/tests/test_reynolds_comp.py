import unittest

from openaerostruct.common.reynolds_comp import ReynoldsComp
from openaerostruct.utils.testing import run_test


class Test(unittest.TestCase):

    def test(self):
        comp = ReynoldsComp()

        run_test(self, comp, method='cs', complex_flag=True)


if __name__ == '__main__':
    unittest.main()
