import unittest
import numpy as np

from openmdao.api import Group, IndepVarComp

from openaerostruct.structures.assemble_k import AssembleK
from openaerostruct.utils.testing import run_test, get_default_surfaces

np.random.seed(314)

class Test(unittest.TestCase):

    def test(self):
        surface = get_default_surfaces()[0]
        group = Group()
        comp = AssembleK(surface=surface)
        indep_var_comp = IndepVarComp()

        ny = surface['num_y']
        nx = surface['num_x']

        indep_var_comp.add_output('nodes', val=np.random.random_sample((ny, 3)), units='m')
        indep_var_comp.add_output('A', val=np.random.random_sample((ny - 1)), units='m**2')
        indep_var_comp.add_output('Iy', val=np.random.random_sample((ny - 1)), units='m**4')
        indep_var_comp.add_output('Iz', val=np.random.random_sample((ny - 1)), units='m**4')
        indep_var_comp.add_output('J', val=np.random.random_sample((ny - 1)), units='m**4')

        group.add_subsystem('indep_var_comp', indep_var_comp, promotes=['*'])
        group.add_subsystem('assemble_k', comp, promotes=['*'])

        run_test(self, group)


if __name__ == '__main__':
    unittest.main()
