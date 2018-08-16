import unittest

from openaerostruct.aerodynamics.wave_drag import WaveDrag
from openaerostruct.utils.testing import run_test, get_default_surfaces


class Test(unittest.TestCase):

    def test(self):
        surface = get_default_surfaces()[0]
        comp = WaveDrag(surface=surface, with_wave=True)

        run_test(self, comp, complex_flag=True)


if __name__ == '__main__':
    unittest.main()
