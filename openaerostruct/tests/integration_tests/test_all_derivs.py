from __future__ import print_function, division, absolute_import

import unittest

import itertools
from six import iteritems

import numpy as np
from numpy.testing import assert_almost_equal

from openmdao.api import Problem, Group, IndepVarComp, ScipyOptimizeDriver, view_model, ExecComp, SqliteRecorder

from openaerostruct.geometry.inputs_group import InputsGroup
from openaerostruct.structures.fea_bspline_group import FEABsplineGroup

from openaerostruct.aerodynamics.vlm_preprocess_group import VLMPreprocessGroup
from openaerostruct.aerodynamics.vlm_postprocess_group import VLMPostprocessGroup

from openaerostruct.structures.fea_preprocess_group import FEAPreprocessGroup
from openaerostruct.structures.fea_postprocess_group import FEAPostprocessGroup

from openaerostruct.aerodynamics.components.forces.vlm_panel_forces_comp import VLMPanelForcesComp
from openaerostruct.tests.utils import get_default_lifting_surfaces

def run_test(obj, comp, decimal=3):
    prob = Problem()
    prob.model.add_subsystem('comp', comp)
    prob.setup(force_alloc_complex=True)

    prob.run_model()
    check = prob.check_partials(compact_print=True)
    for key, subjac in iteritems(check[list(check.keys())[0]]):
        if subjac['magnitude'].fd > 1e-6:
            assert_almost_equal(
                subjac['rel error'].forward, 0., decimal=decimal, err_msg='deriv of %s wrt %s' % key)
            assert_almost_equal(
                subjac['rel error'].reverse, 0., decimal=decimal, err_msg='deriv of %s wrt %s' % key)


class Test(unittest.TestCase):

    def test(self):
        lifting_surfaces = get_default_lifting_surfaces()

        comp = VLMPanelForcesComp(num_nodes=1, lifting_surfaces=lifting_surfaces)

        run_test(self, comp)

if __name__ == '__main__':
    unittest.main()
