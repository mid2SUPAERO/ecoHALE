import unittest

from openmdao.api import Group, IndepVarComp, Problem
from openaerostruct.structures.fuel_loads import FuelLoads
from openaerostruct.utils.testing import run_test, get_default_surfaces
import numpy as np

class Test(unittest.TestCase):

    def test_0(self):
        surface = get_default_surfaces()[0]

        comp = FuelLoads(surface=surface)

        group = Group()

        indep_var_comp = IndepVarComp()

        ny = surface['num_y']

        nodesval = np.array([[0., 0., 0.],
                            [0., 1., 0.],
                            [0., 2., 0.],
                            [0., 3., 0.]], dtype=complex)

        indep_var_comp.add_output('nodes', val=nodesval)

        group.add_subsystem('indep_var_comp', indep_var_comp, promotes=['*'])
        group.add_subsystem('load', comp, promotes=['*'])

        run_test(self, group, complex_flag=True)

if __name__ == '__main__':
    unittest.main()
