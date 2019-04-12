import unittest

import numpy as np

from openmdao.api import Group, IndepVarComp

from openaerostruct.aerodynamics.pg_transform import PGTransform
from openaerostruct.utils.testing import run_test, get_default_surfaces


class Test(unittest.TestCase):

    def test_derivatives(self):
        surfaces = get_default_surfaces()

        group = Group()

        for surface in surfaces:
            surface['symmetry'] = False

        comp = PGTransform(surfaces=surfaces)

        indep_var_comp = IndepVarComp()

        for surface in surfaces:
            name = surface['name']
            indep_var_comp.add_output(name+'_def_mesh', val=surface['mesh'], units='m')
            group.connect('indep_var_comp.'+name+'_def_mesh',
                'pg_transform.'+name+'.def_mesh')
            indep_var_comp.add_output(name+'_b_pts', val=surface['mesh'][1:,:,:], units='m')
            group.connect('indep_var_comp.'+name+'_b_pts',
                'pg_transform.'+name+'.b_pts')
            indep_var_comp.add_output(name+'_c_pts', val=surface['mesh'][1:,1:,:], units='m')
            group.connect('indep_var_comp.'+name+'_c_pts',
                'pg_transform.'+name+'.c_pts')
            indep_var_comp.add_output(name+'_normals', val=surface['mesh'][1:,1:,:])
            group.connect('indep_var_comp.'+name+'_normals',
                'pg_transform.'+name+'.normals')
            indep_var_comp.add_output(name+'_v_rot', val=surface['mesh'][1:,1:,:], units='m/s')
            group.connect('indep_var_comp.'+name+'_v_rot',
                'pg_transform.'+name+'.v_rot')

        indep_var_comp.add_output('alpha', val=1.0, units='deg')
        group.connect('indep_var_comp.alpha', 'pg_transform.alpha')

        indep_var_comp.add_output('beta', val=-1.0, units='deg')
        group.connect('indep_var_comp.beta', 'pg_transform.beta')

        indep_var_comp.add_output('M', val=0.3)
        group.connect('indep_var_comp.M', 'pg_transform.M')

        group.add_subsystem('indep_var_comp', indep_var_comp)
        group.add_subsystem('pg_transform', comp)

        run_test(self, group)


if __name__ == '__main__':
    unittest.main()
