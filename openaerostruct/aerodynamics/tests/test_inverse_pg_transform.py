import unittest
import numpy as np

from openmdao.api import Group, IndepVarComp

from openaerostruct.aerodynamics.inverse_pg_transform import InversePGTransform
from openaerostruct.utils.testing import run_test, get_default_surfaces


class Test(unittest.TestCase):

    def test(self):
        surfaces = get_default_surfaces()

        group = Group()

        comp = InversePGTransform(surfaces=surfaces)

        indep_var_comp = IndepVarComp()

        for surface in surfaces:
            name = surface['name']
            indep_var_comp.add_output(name+'_sec_forces_pg', val=surface['mesh'][1:,1:,:], units='N')
            group.connect('indep_var_comp.'+name+'_sec_forces_pg',
                'inverse_pg_transform.'+name+'.sec_forces_pg')
            indep_var_comp.add_output(name+'_node_forces_pg', val=np.ones(surface['mesh'].shape), units='N')
            group.connect('indep_var_comp.'+name+'_node_forces_pg',
                'inverse_pg_transform.'+name+'.node_forces_pg')

        indep_var_comp.add_output('alpha', val=1.0, units='deg')
        group.connect('indep_var_comp.alpha', 'inverse_pg_transform.alpha')

        indep_var_comp.add_output('beta', val=-1.0, units='deg')
        group.connect('indep_var_comp.beta', 'inverse_pg_transform.beta')

        indep_var_comp.add_output('M', val=0.3)
        group.connect('indep_var_comp.M', 'inverse_pg_transform.M')

        group.add_subsystem('indep_var_comp', indep_var_comp)
        group.add_subsystem('inverse_pg_transform', comp)

        run_test(self, group)


if __name__ == '__main__':
    unittest.main()
