from openmdao.api import Problem

from six import iteritems
from numpy.testing import assert_almost_equal


def run_test(obj, comp, decimal=3):
    prob = Problem(model=comp)
    prob.setup()

    prob.run_model()
    check = prob.check_partial_derivs(compact_print=True)
    for key, subjac in iteritems(check['']):
        if subjac['magnitude'].fd > 1e-6:
            assert_almost_equal(subjac['rel error'].forward, 0., decimal=decimal, err_msg=key)
            assert_almost_equal(subjac['rel error'].reverse, 0., decimal=decimal, err_msg=key)


def get_default_prob_dict():
    prob_dict = {
        'g': 9.81,
        'CT': 9.80665 * 17.e-6,
        'a': 295.4,
        'R': 14.3e6,
        'M': 0.84,
        'W0': 1e6,
        'beta': 0.5,
    }
    return prob_dict


def get_default_surfaces():
    surfaces = [{'name': 'wing'}]
    return surfaces
