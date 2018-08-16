import unittest
import numpy as np
from openaerostruct.aerodynamics.wave_drag import WaveDrag
from openaerostruct.utils.testing import run_test, get_default_surfaces
from openmdao.api import Group, IndepVarComp, BsplinesComp

class Test(unittest.TestCase):

    def test(self):
        surface = get_default_surfaces()[0]
        surface['with_wave'] = True

        surface['t_over_c_cp'] = np.array([1., 2., 3444.])

        ny = surface['num_y']
        nx = surface['num_x']
        n_cp = len(surface['t_over_c_cp'])

        group = Group()

        indep_var_comp = IndepVarComp()
        indep_var_comp.add_output('t_over_c', val=np.arange(ny-1))
        group.add_subsystem('indep_var_comp', indep_var_comp, promotes=['*'])

        comp = WaveDrag(surface=surface, with_wave=True)
        group.add_subsystem('wavedrag', comp, promotes=['*'])

        run_test(self, group, complex_flag=True)

if __name__ == '__main__':
    unittest.main()
