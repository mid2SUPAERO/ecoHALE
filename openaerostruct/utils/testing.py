from openmdao.api import Problem

import unittest


def run_test(obj, comp):
    prob = Problem(model=comp)
    prob.setup()

    prob.run_model()
    check = prob.check_partial_derivs(compact_print=True)
