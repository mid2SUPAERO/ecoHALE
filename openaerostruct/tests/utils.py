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
from openaerostruct.common.lifting_surface import LiftingSurface

num_nodes = 1
g = 9.81

def get_default_lifting_surfaces():
    num_nodes = 1
    num_points_x = 2
    num_points_z_half = 2
    num_points_z = 2 * num_points_z_half - 1
    g = 9.81

    wing = LiftingSurface('wing')

    wing.initialize_mesh(num_points_x, num_points_z_half, airfoil_x=np.linspace(0., 1., num_points_x), airfoil_y=np.zeros(num_points_x))
    wing.set_mesh_parameters(distribution='sine', section_origin=.25)
    wing.set_structural_properties(E=70.e9, G=29.e9, spar_location=0.35, sigma_y=200e6, rho=2700)
    wing.set_aero_properties(factor2=.119, factor4=-0.064, cl_factor=1.05)

    wing.set_chord(1.)
    wing.set_twist(0.)
    wing.set_sweep(0.)
    wing.set_dihedral(0.)
    wing.set_span(5.)
    wing.set_thickness(0.05)
    wing.set_radius(0.1)

    lifting_surfaces = [('wing', wing)]

    return lifting_surfaces


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
