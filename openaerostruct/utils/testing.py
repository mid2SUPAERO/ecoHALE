from openmdao.api import Problem

from six import iteritems
from numpy.testing import assert_almost_equal
from openaerostruct.integration.integration import OASProblem


def run_test(obj, comp, decimal=3):
    prob = Problem(model=comp)
    prob.setup()

    prob.run_model()
    check = prob.check_partial_derivs(compact_print=True)
    for key, subjac in iteritems(check['']):
        if subjac['magnitude'].fd > 1e-6:
            assert_almost_equal(
                subjac['rel error'].forward, 0., decimal=decimal, err_msg='%s,%s' % key)
            assert_almost_equal(
                subjac['rel error'].reverse, 0., decimal=decimal, err_msg='%s,%s' % key)


def get_default_prob_dict():
    return OASProblem().get_default_prob_dict()

def get_default_surf_dict():
    return OASProblem().get_default_surf_dict()

def get_default_surfaces():
    surf_dict = OASProblem().get_default_surf_dict()
    return [surf_dict]
