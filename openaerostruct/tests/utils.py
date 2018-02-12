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

num_nodes = 1
g = 9.81

def get_default_lifting_surfaces():
    num_points_x = 2
    num_points_z_half = 2
    num_points_z = 2 * num_points_z_half - 1
    lifting_surfaces = [
        ('wing', {
            'num_points_x': num_points_x, 'num_points_z_half': num_points_z_half,
            'airfoil_x': np.linspace(0., 1., num_points_x),
            'airfoil_y': np.zeros(num_points_x),
            'chord': 1., 'twist': 0. * np.pi / 180., 'sweep_x': 0., 'dihedral_y': 0., 'span': 5,
            'twist_bspline': (6, 2),
            'sec_z_bspline': (num_points_z_half, 2),
            'chord_bspline': (2, 2),
            'thickness_bspline': (6, 3),
            'thickness' : 0.05,
            'radius' : 0.1,
            'distribution': 'sine',
            'section_origin': 0.25,
            'spar_location': 0.35,
            'E': 70.e9,
            'G': 29.e9,
            'sigma_y': 200e6,
            'rho': 2700,
            'factor2' : 0.119,
            'factor4' : -0.064,
            'cl_factor' : 1.05,
            'W0' : (0.1381 * g - .350) * 1e6 + 300 * 80 * g,
            'a' : 295.4,
            'R' : 7000. * 1.852 * 1e3,
            'M' : .84,
            'CT' : g * 17.e-6,
        })
    ]

    return lifting_surfaces


def run_test(obj, comp, decimal=3):
    prob = Problem()
    prob.model.add_subsystem('comp', comp)
    prob.setup(force_alloc_complex=True)

    prob.run_model()
    check = prob.check_partials(compact_print=True, method='cs')
    for key, subjac in iteritems(check[list(check.keys())[0]]):
        if subjac['magnitude'].fd > 1e-6:
            assert_almost_equal(
                subjac['rel error'].forward, 0., decimal=decimal, err_msg='deriv of %s wrt %s' % key)
            assert_almost_equal(
                subjac['rel error'].reverse, 0., decimal=decimal, err_msg='deriv of %s wrt %s' % key)
