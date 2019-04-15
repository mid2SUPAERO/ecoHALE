import unittest
import numpy as np

from openmdao.api import Group, IndepVarComp
from openaerostruct.structures.add_point_masses import AddPointMasses
from openaerostruct.utils.testing import run_test, get_default_surfaces

class Test(unittest.TestCase):

    def test(self):
        surface = get_default_surfaces()[0]

        # turn down some of these properties, so the absolute deriv error isn't magnified
        surface['E'] = 7
        surface['G'] = 3
        surface['yield'] = .02
        surface['n_point_masses'] = 2

        comp = AddPointMasses(surface=surface)

        group = Group()

        indep_var_comp = IndepVarComp()

        ny = surface['mesh'].shape[1]

        nodesval = np.array([[0., 0., 0.],
                            [0., 1., 0.],
                            [0., 2., 0.],
                            [0., 3., 0.]])

        point_masses = np.array([[2./9.8, 0.]])

        point_mass_locations = np.array([[2., 0., 0.],
                                         [3., 1., 0.]])

        indep_var_comp.add_output('nodes', val=nodesval)
        indep_var_comp.add_output('point_masses', val=point_masses)
        indep_var_comp.add_output('point_mass_locations', val=point_mass_locations)

        group.add_subsystem('indep_var_comp', indep_var_comp, promotes=['*'])
        group.add_subsystem('add_point_masses', comp, promotes=['*'])

        run_test(self, group,  complex_flag=True, step=1e-8, atol=2e-5, compact_print=True)


if __name__ == '__main__':
    unittest.main()
