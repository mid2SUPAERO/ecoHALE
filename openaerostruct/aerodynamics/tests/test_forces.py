import unittest
import numpy as np

from openmdao.api import Group, IndepVarComp

from openaerostruct.aerodynamics.forces import Forces
from openaerostruct.utils.testing import run_test, get_default_surfaces


class Test(unittest.TestCase):

    def test(self):
        surfaces = get_default_surfaces()

        group = Group()

        comp = Forces(surfaces=surfaces)

        indep_var_comp = IndepVarComp()

        indep_var_comp.add_output('wing_widths', val=np.ones((surfaces[0]['num_y']-1)), units='m')

        group.add_subsystem('indep_var_comp', indep_var_comp)
        group.add_subsystem('forces', comp)

        group.connect('indep_var_comp.wing_widths', 'forces.wing_widths')

        run_test(self, group)


if __name__ == '__main__':
    unittest.main()
