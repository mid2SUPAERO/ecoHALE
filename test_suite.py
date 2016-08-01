from __future__ import division
import sys
from time import time
import numpy

from openmdao.api import IndepVarComp, Problem, Group, ScipyOptimizer, SqliteRecorder
from openmdao.devtools.partition_tree_n2 import view_tree
from geometry import GeometryMesh, Bspline, gen_mesh, get_inds
from transfer import TransferDisplacements
from vlm import VLMStates, VLMFunctionals
from b_spline import get_bspline_mtx
import unittest

class TestVLM(unittest.TestCase):

    # Solve some simple VLM problems

    def get_default_dict(self):
        defaults = {'num_x' : 3,
                    'num_y' : 5,
                    'span' : 10.,
                    'chord' : 1.,
                    'cosine_spacing' : 1,
                    'dihedral' : 0.,
                    'sweep' : 0.,
                    'taper' : 1.,
                    'Re' : 0.,
                    'alpha' : 5.,
                    'optimize' : False,
                    'W0' : 0.5 * 2.5e6, # [N] (MTOW of B777 is 3e5 kg with fuel)
                    'CT' : 9.81 * 17.e-6, # [1/s] (9.81 N/kg * 17e-6 kg/N/s)
                    'R' : 14.3e6, # [m] maximum range
                    'M' : 0.84, # at cruise
                    'rho' : 0.38, # [kg/m^3] at 35,000 ft
                    'a' : 295.4, # [m/s] at 35,000 ft
                    'CL0' : 0.2,
                    'CD0' : 0.015,
                    }
        return defaults

    def run_aero_case(self, target_value, input_dict={}):

        v_dict = self.get_default_dict()
        v_dict.update(input_dict)

        for name in v_dict.keys():
            exec(name + ' = v_dict[ "' + name + '"]')

        v = a * M

        mesh = gen_mesh(num_x, num_y, span, chord, cosine_spacing)

        num_twist = numpy.max([int((num_y - 1) / 5), 5])

        mesh = mesh.reshape(-1, mesh.shape[-1])
        aero_ind = numpy.atleast_2d(numpy.array([num_x, num_y]))
        fem_ind = [num_y]

        # Compute the aero and fem indices
        aero_ind, fem_ind = get_inds(aero_ind, fem_ind)

        # Create the top-level system
        root = Group()

        # Define Jacobians for b-spline controls
        tot_n_fem = numpy.sum(fem_ind[:, 0])
        jac = get_bspline_mtx(num_twist, num_y)

        # Define the independent variables
        des_vars = [
            ('twist_cp', numpy.zeros(num_twist)),
            ('dihedral', dihedral),
            ('sweep', sweep),
            ('span', span),
            ('taper', taper),
            ('v', v),
            ('alpha', alpha),
            ('rho', rho),
            ('disp', numpy.zeros((tot_n_fem, 6))),
            ('aero_ind', aero_ind),
            ('fem_ind', fem_ind),
            ('Re', Re)
        ]

        # Add VLM components to the top-level system
        root.add('des_vars',
                 IndepVarComp(des_vars),
                 promotes=['*'])
        root.add('twist_bsp',
                 Bspline('twist_cp', 'twist', jac),
                 promotes=['*'])
        root.add('mesh',
                 GeometryMesh(mesh, aero_ind),
                 promotes=['*'])
        root.add('def_mesh',
                 TransferDisplacements(aero_ind, fem_ind),
                 promotes=['*'])
        root.add('vlmstates',
                 VLMStates(aero_ind),
                 promotes=['*'])
        print 'CLLLLL', CL0
        root.add('vlmfuncs',
                 VLMFunctionals(aero_ind, CL0, CD0),
                 promotes=['*'])

        # Set the optimization problem settings
        prob = Problem()
        prob.root = root

        try:  # Use SNOPT optimizer if installed
            from openmdao.api import pyOptSparseDriver
            prob.driver = pyOptSparseDriver()
            prob.driver.options['optimizer'] = "SNOPT"
            prob.driver.opt_settings = {'Major optimality tolerance': 1.0e-8,
                                        'Major feasibility tolerance': 1.0e-8}
        except:  # Use SLSQP optimizer if SNOPT not installed
            prob.driver = ScipyOptimizer()
            prob.driver.options['optimizer'] = 'SLSQP'
            prob.driver.options['disp'] = True
            prob.driver.options['tol'] = 1.0e-8

        # Add design variables for the optimizer to control
        # Note that the scaling is very important to get correct convergence
        prob.driver.add_desvar('twist_cp', lower=-10., upper=15., scaler=1e0)
        # prob.driver.add_desvar('alpha', lower=-10., upper=10.)
        prob.driver.add_desvar('sweep', lower=-10., upper=30.)
        prob.driver.add_desvar('dihedral', lower=-10., upper=20.)
        prob.driver.add_desvar('taper', lower=.5, upper=2.)

        # Set the objective (minimize CD on the main wing)
        prob.driver.add_objective('CD_wing', scaler=1e4)

        # Set the constraint (CL = 0.5 for the main wing)
        prob.driver.add_constraint('CL_wing', equals=0.5)

        # Record optimization history to a database
        # Data saved here can be examined using `plot_all.py`
        prob.driver.add_recorder(SqliteRecorder('test.db'))

        # Can finite difference over the entire model
        # Generally faster than using component derivatives
        prob.root.deriv_options['type'] = 'fd'

        # Setup the problem and produce an N^2 diagram
        prob.setup()

        prob.run_once()
        if not optimize:  # run analysis once
            self.assertAlmostEqual(prob['CL'], target_value)
            pass
        else:  # perform optimization
            prob.run()
            self.assertAlmostEqual(prob['CD_wing'], target_value)


    def test_aero_analysis_flat(self):
        self.run_aero_case(.65655138)  # Match the CL

    def test_aero_optimization_flat(self):
        v_dict = {'optimize' : True}
        self.run_aero_case(.0314570988, v_dict)  # Match the objective value





if __name__ == "__main__":
    unittest.main()
