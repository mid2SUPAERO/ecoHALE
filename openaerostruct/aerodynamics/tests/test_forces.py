import unittest
import numpy as np

from openmdao.api import Group, IndepVarComp

from openaerostruct.aerodynamics.forces import Forces
from openaerostruct.utils.testing import run_test, get_default_surfaces

@unittest.skipUnless(0, "This test is deprecated as the component has been replaced.")
class Test(unittest.TestCase):

    def test(self):
        surfaces = get_default_surfaces()

        group = Group()

        comp = Forces(surfaces=surfaces)

        indep_var_comp = IndepVarComp()

        indep_var_comp.add_output('wing_widths', val=np.ones((surfaces[0]['num_y']-1)), units='m')
        indep_var_comp.add_output('tail_widths', val=np.ones((surfaces[1]['num_y']-1)), units='m')
        indep_var_comp.add_output('wing_def_mesh', val=surfaces[0]['mesh'], units='m')
        indep_var_comp.add_output('tail_def_mesh', val=surfaces[1]['mesh'], units='m')
        indep_var_comp.add_output('M', val=.3)

        group.add_subsystem('indep_var_comp', indep_var_comp, promotes=['*'])
        group.add_subsystem('forces', comp, promotes=['*'])

        run_test(self, group)


if __name__ == '__main__':
    unittest.main()
