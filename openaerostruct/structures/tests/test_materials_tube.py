import unittest

from openaerostruct.structures.materials_tube import MaterialsTube
from openaerostruct.utils.testing import run_test, get_default_surfaces


class Test(unittest.TestCase):

    def test(self):
        surface = get_default_surfaces()[0]

        comp = MaterialsTube(surface=surface)

        run_test(self, comp)


if __name__ == '__main__':
    unittest.main()
