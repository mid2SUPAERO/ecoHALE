import unittest
import numpy as np

from openmdao.utils.assert_utils import assert_rel_error
from openmdao.api import Group, IndepVarComp
from openaerostruct.structures.add_point_masses import AddPointMasses
from openaerostruct.utils.testing import run_test, get_default_surfaces

class Test(unittest.TestCase):

    def test_derivs(self):
        surface = get_default_surfaces()[0]

        surface['n_point_masses'] = 2

        comp = AddPointMasses(surface=surface)

        group = Group()

        indep_var_comp = IndepVarComp()

        ny = surface['mesh'].shape[1]

        nodesval = np.array([[0., 0., 0.],
                            [0., 1., 0.],
                            [0., 2., 0.],
                            [0., 3., 0.]])

        point_masses = np.array([[2., 1.]])

        point_mass_locations = np.array([[2., 0., 0.],
                                         [3., 1., 0.]])

        indep_var_comp.add_output('nodes', val=nodesval, units='m')
        indep_var_comp.add_output('point_masses', val=point_masses, units='kg')
        indep_var_comp.add_output('point_mass_locations', val=point_mass_locations, units='m')

        group.add_subsystem('indep_var_comp', indep_var_comp, promotes=['*'])
        group.add_subsystem('add_point_masses', comp, promotes=['*'])

        prob = run_test(self, group,  complex_flag=True, step=1e-8, atol=1e-5, compact_print=True)

        truth_array = np.array([0, 0, -16.87832073, -2.47465336, 36.23129481, 0.])

        assert_rel_error(self, prob['comp.loads_from_point_masses'][0, :], truth_array, 1e-6)

    def test_simple_values(self):
        surface = get_default_surfaces()[0]

        surface['n_point_masses'] = 1

        comp = AddPointMasses(surface=surface)

        group = Group()

        indep_var_comp = IndepVarComp()

        ny = surface['mesh'].shape[1]

        nodesval = np.array([[0., 0., 0.],
                            [0., 1., 0.],
                            [0., 2., 0.],
                            [0., 3., 0.]])

        point_masses = np.array([[1/9.8]])

        point_mass_locations = np.array([[.55012, 0., 0.]])

        indep_var_comp.add_output('nodes', val=nodesval, units='m')
        indep_var_comp.add_output('point_masses', val=point_masses, units='kg')
        indep_var_comp.add_output('point_mass_locations', val=point_mass_locations, units='m')

        group.add_subsystem('indep_var_comp', indep_var_comp, promotes=['*'])
        group.add_subsystem('add_point_masses', comp, promotes=['*'])

        prob = run_test(self, group,  complex_flag=True, step=1e-8, atol=1e-5, compact_print=True)

        truth_array = np.array([0, 0, -1., 0., 0.55012, 0.])

        assert_rel_error(self, prob['comp.loads_from_point_masses'][0, :], truth_array, 1e-6)


if __name__ == '__main__':
    unittest.main()
