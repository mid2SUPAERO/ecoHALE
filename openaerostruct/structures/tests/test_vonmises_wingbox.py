import unittest
import numpy as np

from openmdao.api import Group, IndepVarComp
from openaerostruct.structures.vonmises_wingbox import VonMisesWingbox
from openaerostruct.utils.testing import run_test, get_default_surfaces

class Test(unittest.TestCase):

    def test(self):
        surface = get_default_surfaces()[0]

        surface['strength_factor_for_upper_skin'] = 1.0

        comp = VonMisesWingbox(surface=surface)

        group = Group()

        indep_var_comp = IndepVarComp()

        ny = surface['num_y']

        nodesval = np.array([[0., 0., 0.],
                            [0., 1., 0.],
                            [0., 2., 0.],
                            [0., 3., 0.]])

        indep_var_comp.add_output('nodes', val=nodesval)
        indep_var_comp.add_output('disp', val=np.ones((ny, 6),
                       dtype=complex))
        indep_var_comp.add_output('Qz', val=np.ones((ny - 1), dtype=complex))
        indep_var_comp.add_output('Iz', val=np.ones((ny - 1), dtype=complex))
        indep_var_comp.add_output('J', val=np.ones((ny - 1), dtype=complex))
        indep_var_comp.add_output('A_enc', val=np.ones((ny - 1), dtype=complex))
        indep_var_comp.add_output('spar_thickness', val=np.ones((ny - 1), dtype=complex))
        indep_var_comp.add_output('skin_thickness', val=np.ones((ny - 1), dtype=complex))
        indep_var_comp.add_output('htop', val=np.ones((ny - 1), dtype=complex))
        indep_var_comp.add_output('hbottom', val=np.ones((ny - 1), dtype=complex))
        indep_var_comp.add_output('hfront', val=np.ones((ny - 1), dtype=complex))
        indep_var_comp.add_output('hrear', val=np.ones((ny - 1), dtype=complex))

        group.add_subsystem('indep_var_comp', indep_var_comp, promotes=['*'])
        group.add_subsystem('vonmises_wingbox', comp, promotes=['*'])

        run_test(self, group,  complex_flag=True)


if __name__ == '__main__':
    unittest.main()
