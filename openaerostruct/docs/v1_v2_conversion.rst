.. _V1_V2_Conversion:

OpenAeroStruct v1 to v2 Conversion
==================================

There are quite a few differences between OpenAeroStruct v1 and v2, though the underlying analyses remain largely the same.
In this document, we'll example code using v1 on the left in red and the corresponding code to perform the same computations in v2 on the right in green.
This can serve as a Rosetta Stone to translate older v1 scripts to run in v2.
We'll first go through individual sets of commands then present a full example script that performs the same analyses in both versions.

.. content-container ::

  .. embed-compare::
      openaerostruct.tests.test_v1_aero_opt.Test.test
      dictionary
      prob.run_driver()

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
