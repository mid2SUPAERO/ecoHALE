from __future__ import division
import sys
from time import time
import numpy
import unittest

from run_classes import *

class TestVLM(unittest.TestCase):

    def test_aero_analysis_flat(self):
        OAS_prob = OASProblem({'optimize' : False})
        OAS_prob.add_surface()
        OAS_prob.run_aero()
        prob = OAS_prob.prob
        self.assertAlmostEqual(prob['_CL'], .65655138)

    def test_aero_analysis_flat_symmetry(self):
        v_dict = {'symmetry' : True}
        OAS_prob = OASProblem({'optimize' : False})
        OAS_prob.add_surface(v_dict)
        OAS_prob.run_aero()
        prob = OAS_prob.prob
        self.assertAlmostEqual(prob['_CL'], .65655138)

    def test_aero_optimization_flat(self):
        OAS_prob = OASProblem({'optimize' : True})
        OAS_prob.add_surface()
        OAS_prob.run_aero()
        prob = OAS_prob.prob
        self.assertAlmostEqual(prob['_CD'], .0164570988)

    def test_aerostruct_analysis(self):
        v_dict = {'num_y' : 13,
                  'num_x' : 2,
                  'wing_type' : 'CRM'}
        OAS_prob = OASProblem({'optimize' : False})
        OAS_prob.add_surface(v_dict)
        OAS_prob.run_as()
        prob = OAS_prob.prob
        self.assertAlmostEqual(prob['_CL'], .58245256)
        self.assertAlmostEqual(prob['_failure'], -.431801158)

    def test_aerostruct_analysis_symmetry(self):
        v_dict = {'symmetry' : True,
                  'num_y' : 13,
                  'num_x' : 2,
                  'wing_type' : 'CRM'}
        OAS_prob = OASProblem({'optimize' : False})
        OAS_prob.add_surface(v_dict)
        OAS_prob.run_as()
        prob = OAS_prob.prob
        self.assertAlmostEqual(prob['_CL'], .58245256)
        self.assertAlmostEqual(prob['_failure'], -.5011158763)

    def test_aerostruct_optimization(self):
        v_dict = {'num_y' : 13,
                  'num_x' : 2,
                  'wing_type' : 'CRM'}
        OAS_prob = OASProblem({'optimize' : True})
        OAS_prob.add_surface(v_dict)
        OAS_prob.run_as()
        prob = OAS_prob.prob
        self.assertAlmostEqual(prob['fuelburn'], 922881.03287074505, places=2)
        self.assertAlmostEqual(prob['_failure'], 1e-9)

    def test_aerostruct_optimization_symmetry(self):
        v_dict = {'symmetry' : True,
                  'num_y' : 13,
                  'num_x' : 2,
                  'wing_type' : 'CRM'}
        OAS_prob = OASProblem({'optimize' : True})
        OAS_prob.add_surface(v_dict)
        OAS_prob.run_as()
        prob = OAS_prob.prob
        self.assertAlmostEqual(prob['fuelburn'], 427046.16000037867, places=3)
        self.assertAlmostEqual(prob['_failure'], 1e-9)

if __name__ == "__main__":
    print
    print '===================================================='
    print '|                                                  |'
    print '|             RUNNING FULL TEST SUITE              |'
    print '|                                                  |'
    print '===================================================='
    print
    unittest.main()
