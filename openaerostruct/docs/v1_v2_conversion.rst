.. _V1_V2_Conversion:

OpenAeroStruct v1 to v2 Conversion
==================================

.. content-container ::

  .. embed-compare::
      openaerostruct.tests.test_v1_aero_opt.Test.test

    from __future__ import division, print_function
    from time import time
    import numpy as np

    from OpenAeroStruct import OASProblem


    # Set problem type
    prob_dict = {'type' : 'aero',
                 'optimize' : True}

    # Instantiate problem and add default surface
    OAS_prob = OASProblem(prob_dict)

    # Create a dictionary to store options about the surface
    surf_dict = {'num_y' : 5,
                 'num_x' : 3,
                 'wing_type' : 'rect',
                 'symmetry' : True,
                 'num_twist_cp' : 2}

    # Add the specified wing surface to the problem
    OAS_prob.add_surface(surf_dict)

    # Setup problem and add design variables, constraint, and objective
    OAS_prob.add_desvar('wing.twist_cp', lower=-10., upper=15.)
    OAS_prob.add_desvar('wing.sweep', lower=10., upper=30.)
    OAS_prob.add_desvar('wing.dihedral', lower=-10., upper=20.)
    OAS_prob.add_constraint('wing_perf.CL', equals=0.5)
    OAS_prob.add_objective('wing_perf.CD', scaler=1e4)
    OAS_prob.setup()

    # Actually run the problem
    OAS_prob.run()

    print("\nWing CL:", OAS_prob.prob['wing_perf.CL'])
    print("Wing CD:", OAS_prob.prob['wing_perf.CD'])
