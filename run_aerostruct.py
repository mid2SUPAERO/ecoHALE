""" Example runscript to perform aerostructural optimization.

Call as `python run_aerostruct.py 0` to run a single analysis, or
call as `python run_aerostruct.py 1` to perform optimization.

Call as `python run_aerostruct.py 0m` to run a single analysis with
multiple surfaces, or
call as `python run_aerostruct.py 1m` to perform optimization with
multiple surfaces.

"""

from __future__ import division
import sys
from time import time
import numpy

from run_classes import OASProblem

if __name__ == "__main__":

    # Set problem type
    prob_dict = {'type' : 'aerostruct'}

    if sys.argv[1].startswith('0'):  # run analysis once
        prob_dict.update({'optimize' : False})
    else:  # perform optimization
        prob_dict.update({'optimize' : True})

    # Instantiate problem and add default surface
    OAS_prob = OASProblem(prob_dict)

    # Create a dictionary to store options about the surface
    surf_dict = {'name' : 'wing',
                 'symmetry' : False,
                 'num_y' : 13,
                 'num_x' : 2,
                 'wing_type' : 'CRM',
                 'CL0' : 0.2,
                 'CD0' : 0.015}

    # Add the specified wing surface to the problem
    OAS_prob.add_surface(surf_dict)

    # Single lifting surface
    if not sys.argv[1].endswith('m'):

        # Setup problem and add design variables, constraint, and objective
        OAS_prob.setup()
        OAS_prob.add_desvar('wing.twist_cp', lower=-15., upper=15.)
        OAS_prob.add_desvar('wing.thickness_cp', lower=0.001, upper=0.25, scaler=1e2)
        OAS_prob.add_constraint('wing_perf.failure', upper=0.)

    # Multiple lifting surfaces
    else:

        # Add additional lifting surface
        surf_dict.update({'name' : 'tail',
                          'offset':numpy.array([0., 0., 10.])})
        OAS_prob.add_surface(surf_dict)

        # Setup problem and add design variables, constraints, and objective
        OAS_prob.setup()

        # Add design variables and constraints for both the wing and tail
        OAS_prob.add_desvar('wing.twist_cp', lower=-15., upper=15.)
        OAS_prob.add_desvar('wing.thickness_cp', lower=0.001, upper=0.25, scaler=1e2)
        OAS_prob.add_constraint('wing_perf.failure', upper=0.)
        OAS_prob.add_desvar('tail.twist_cp', lower=-15., upper=15.)
        OAS_prob.add_desvar('tail.thickness_cp', lower=0.001, upper=0.25, scaler=1e2)
        OAS_prob.add_constraint('tail_perf.failure', upper=0.)

    # Add design variables, constraint, and objective on the problem
    OAS_prob.add_desvar('alpha', lower=-10., upper=10.)
    OAS_prob.add_constraint('eq_con', equals=0.)
    OAS_prob.add_objective('fuelburn', scaler=1e-5)

    st = time()
    # Actually run the problem
    OAS_prob.run()
    print "\nTime:", time()-st, 'secs'

    print "\nFuelburn:", OAS_prob.prob['fuelburn']
