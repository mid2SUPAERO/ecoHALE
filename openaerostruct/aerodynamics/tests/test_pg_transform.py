import unittest

import numpy as np

from openmdao.api import Problem
from openmdao.utils.assert_utils import assert_check_partials

from openaerostruct.aerodynamics.pg_transform import PGTransform
from openaerostruct.aerodynamics.pg_scale import ScaleFromPrandtlGlauert, ScaleToPrandtlGlauert
from openaerostruct.aerodynamics.pg_wind_rotation import RotateFromWindFrame, RotateToWindFrame
from openaerostruct.utils.testing import run_test, get_default_surfaces


class Test(unittest.TestCase):

    def test_to_wind_derivs(self):
        surfaces = get_default_surfaces()

        comp = RotateToWindFrame(surfaces=surfaces, rotational=True)

        prob = Problem()
        prob.model.add_subsystem('comp', comp)
        prob.setup(force_alloc_complex=True)

        prob['comp.alpha'] = np.random.random(prob['comp.alpha'].shape)
        prob['comp.beta'] = np.random.random(prob['comp.beta'].shape)
        prob['comp.coll_pts'] = np.random.random(prob['comp.coll_pts'].shape)
        prob['comp.bound_vecs'] = np.random.random(prob['comp.bound_vecs'].shape)
        prob['comp.rotational_velocities'] = np.random.random(prob['comp.rotational_velocities'].shape)
        prob['comp.wing_def_mesh'] = np.random.random(prob['comp.wing_def_mesh'].shape)
        prob['comp.tail_def_mesh'] = np.random.random(prob['comp.tail_def_mesh'].shape)
        prob['comp.tail_normals'] = np.random.random(prob['comp.tail_normals'].shape)
        prob['comp.wing_normals'] = np.random.random(prob['comp.wing_normals'].shape)

        prob.run_model()

        check = prob.check_partials(compact_print=True, method='cs', step=1e-40)

        assert_check_partials(check)

    def test_from_wind_derivs(self):
        surfaces = get_default_surfaces()

        comp = RotateFromWindFrame(surfaces=surfaces)

        prob = Problem()
        prob.model.add_subsystem('comp', comp)
        prob.setup(force_alloc_complex=True)

        prob['comp.alpha'] = np.random.random(prob['comp.alpha'].shape)
        prob['comp.beta'] = np.random.random(prob['comp.beta'].shape)
        prob['comp.wing_sec_forces_w_frame'] = np.random.random(prob['comp.wing_sec_forces_w_frame'].shape)
        prob['comp.tail_sec_forces_w_frame'] = np.random.random(prob['comp.tail_sec_forces_w_frame'].shape)

        prob.run_model()

        check = prob.check_partials(compact_print=True, method='cs', step=1e-40)

        assert_check_partials(check)

    def test_scale_to_pg(self):
        surfaces = get_default_surfaces()

        comp = ScaleToPrandtlGlauert(surfaces=surfaces, rotational=True)

        prob = Problem()
        prob.model.add_subsystem('comp', comp)
        prob.setup(force_alloc_complex=True)

        prob['comp.MachNumber'] = np.random.random(prob['comp.MachNumber'].shape)
        prob['comp.coll_pts_w_frame'] = np.random.random(prob['comp.coll_pts_w_frame'].shape)
        prob['comp.bound_vecs_w_frame'] = np.random.random(prob['comp.bound_vecs_w_frame'].shape)
        prob['comp.rotational_velocities_w_frame'] = np.random.random(prob['comp.rotational_velocities_w_frame'].shape)
        prob['comp.wing_def_mesh_w_frame'] = np.random.random(prob['comp.wing_def_mesh_w_frame'].shape)
        prob['comp.tail_def_mesh_w_frame'] = np.random.random(prob['comp.tail_def_mesh_w_frame'].shape)
        prob['comp.tail_normals_w_frame'] = np.random.random(prob['comp.tail_normals_w_frame'].shape)
        prob['comp.wing_normals_w_frame'] = np.random.random(prob['comp.wing_normals_w_frame'].shape)

        prob.run_model()

        check = prob.check_partials(compact_print=True, method='cs', step=1e-40)

        assert_check_partials(check)

    def test_scale_from_pg(self):
        surfaces = get_default_surfaces()

        comp = ScaleFromPrandtlGlauert(surfaces=surfaces)

        prob = Problem()
        prob.model.add_subsystem('comp', comp)
        prob.setup(force_alloc_complex=True)

        prob['comp.MachNumber'] = np.random.random(prob['comp.MachNumber'].shape)
        prob['comp.wing_sec_forces_pg'] = np.random.random(prob['comp.wing_sec_forces_pg'].shape)
        prob['comp.tail_sec_forces_pg'] = np.random.random(prob['comp.tail_sec_forces_pg'].shape)

        prob.run_model()

        check = prob.check_partials(compact_print=True, method='cs', step=1e-40)

        assert_check_partials(check)

    #def test_derivatives(self):
        #surfaces = get_default_surfaces()

        #group = Group()

        #for surface in surfaces:
            #surface['symmetry'] = False

        #comp = PGTransform(surfaces=surfaces)

        #indep_var_comp = IndepVarComp()

        #for surface in surfaces:
            #name = surface['name']
            #indep_var_comp.add_output(name+'_def_mesh', val=surface['mesh'], units='m')
            #group.connect('indep_var_comp.'+name+'_def_mesh',
                #'pg_transform.'+name+'.def_mesh')
            #indep_var_comp.add_output(name+'_b_pts', val=surface['mesh'][1:,:,:], units='m')
            #group.connect('indep_var_comp.'+name+'_b_pts',
                #'pg_transform.'+name+'.b_pts')
            #indep_var_comp.add_output(name+'_c_pts', val=surface['mesh'][1:,1:,:], units='m')
            #group.connect('indep_var_comp.'+name+'_c_pts',
                #'pg_transform.'+name+'.c_pts')
            #indep_var_comp.add_output(name+'_normals', val=surface['mesh'][1:,1:,:])
            #group.connect('indep_var_comp.'+name+'_normals',
                #'pg_transform.'+name+'.normals')
            #indep_var_comp.add_output(name+'_v_rot', val=surface['mesh'][1:,1:,:], units='m/s')
            #group.connect('indep_var_comp.'+name+'_v_rot',
                #'pg_transform.'+name+'.v_rot')

        #indep_var_comp.add_output('alpha', val=1.0, units='deg')
        #group.connect('indep_var_comp.alpha', 'pg_transform.alpha')

        #indep_var_comp.add_output('beta', val=-1.0, units='deg')
        #group.connect('indep_var_comp.beta', 'pg_transform.beta')

        #indep_var_comp.add_output('M', val=0.3)
        #group.connect('indep_var_comp.M', 'pg_transform.M')

        #group.add_subsystem('indep_var_comp', indep_var_comp)
        #group.add_subsystem('pg_transform', comp)

        #run_test(self, group)


if __name__ == '__main__':
    unittest.main()
