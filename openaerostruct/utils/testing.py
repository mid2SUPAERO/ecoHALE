from openmdao.api import Problem

from six import iteritems
from numpy.testing import assert_almost_equal


def run_test(obj, comp, decimal=3):
    prob = Problem(model=comp)
    prob.setup()

    prob.run_model()
    check = prob.check_partial_derivs(compact_print=True)
    for key, subjac in iteritems(check['']):
        assert_almost_equal(subjac['rel error'].forward, 0., decimal=decimal, err_msg=key)
        assert_almost_equal(subjac['rel error'].reverse, 0., decimal=decimal, err_msg=key)
