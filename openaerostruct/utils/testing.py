from openmdao.api import Problem, Group, IndepVarComp, view_model

from six import iteritems
from numpy.testing import assert_almost_equal


def run_test(obj, comp, decimal=3):
    prob = Problem()
    prob.model.add_subsystem('comp', comp)
    prob.setup()

    view_model(prob, outfile='test.html', show_browser=False)
    prob.run_model()
    check = prob.check_partials(compact_print=True)
    for key, subjac in iteritems(check['comp']):
        if subjac['magnitude'].fd > 1e-6:
            assert_almost_equal(
                subjac['rel error'].forward, 0., decimal=decimal, err_msg='deriv of %s wrt %s' % key)
            assert_almost_equal(
                subjac['rel error'].reverse, 0., decimal=decimal, err_msg='deriv of %s wrt %s' % key)
