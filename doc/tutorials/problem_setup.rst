.. _Problem Setup:

=============
Problem Setup
=============

This page serves as a simple introduction to the internal code of OpenAeroStruct and how it is organized.
Not every feature is explained here, so please contact the authors if you have a suggestion for how to make something more clear.

In general, you'll follow these five steps to set up and run a problem in OpenAeroStruct:

1. Initialize your problem
2. Add your lifting surface(s)
3. Add your design variables, constraints, and objective
4. Call setup() on the problem
5. Call run() to perform analysis/optimization

We'll now investigate these steps individually, using an aerodynamic optimization case as an example.

1. Initialize your problem
--------------------------
First, you will initialize the OpenAeroStruct problem by creating an `OASProblem` instance
with an input dictionary containing the problem-level settings you wish to use.
This includes flow conditions such as Reynolds number and alpha,
as well as aircraft info like specific fuel consumption and range.
Additionally, it contains execution options for the problem, such as
whether to run analysis or optimization and how to compute derivatives.
Please see :func:`OASProblem.get_default_prob_dict` within the :doc:`../source/run_classes` to see the defaults for the problem options dictionary.

Although some options are only used for aerostructural cases, each problem always
has every option defined.
The user-inputted options overwrite any of the default options.
Keywords are case-specific.

.. note::
  Depending on your problem size, using `force_fd = True` might lower
  the computation time for your optimization. This option simply computes
  the total derivatives by using finite-differencing over the entire model.

If you install `pyOptSparse <https://bitbucket.org/mdolab/pyoptsparse>`_, you can use `pyOptSparseDriver` within OpenMDAO.
This allows you to use a wider variety of optimizers.

Here is a sample code block for this step:

.. code-block:: python

  from __future__ import print_function
  from OpenAeroStruct import OASProblem

  # Set problem type and instantiate problem
  prob_dict = {'type' : 'aero',
               'optimize' : True}
  OAS_prob = OASProblem(prob_dict)


2. Add your lifting surface(s)
------------------------------
With your OASProblem instance created, you can now define surfaces and add them to the problem.
In the simplest case, you can add a single lifting surface that represents the wing of an aircraft.
Please see :func:`OASProblem.get_default_surf_dict` within the :doc:`../source/run_classes` to see the defaults for the surface options dictionary.

There are many options for each surface, and they are loosely organized into the following categories:

- Wing definition (mesh size, wing position, symmetry option, etc)
- Geometric variable definitions (span, dihedral, sweep, twist, etc)
- Aerodynamic performance (CL and CD at alpha=0)
- Airfoil properties (turbulence transition point, t/c, location of max t)
- Structural properties (E, G, yield stress, location of spar, etc)
- Options for constraints (KS aggregation, monotonic design variables)

Again, the user-inputted dictionary will override any defaults.
Here is a sample code block:

.. code-block:: python

  # Add lifting surface
  surf_dict = {'name' : 'wing',
               'symmetry' : True,
               'num_y' : 11,
               'num_x' : 3}
  OAS_prob.add_surface(surf_dict)

3. Add your design variables, constraints, and objective
--------------------------------------------------------
.. note::
  This step is only necessary when performing an optimization, with
  `optimize = True` in the problem dictionary.

With the problem and surfaces defined, we can now add a description of the
optimization problem.
The order of these commands does not matter.
These OpenAeroStruct methods simply call the OpenMDAO methods that are documented here: http://openmdao.readthedocs.io/en/latest/srcdocs/packages/core/driver.html

You can choose a certain set of parameters as design variables, including:

- alpha
- taper
- span
- dihedral
- sweep
- chord distribution
- twist distribution
- shear deformation in x direction
- shear deformation in y direction
- structural spar radii distribution
- structural spar thickness distribution

For the constraints and objective, you can choose any outputted variable.
Common examples include weight, fuel burn, CL, and CD.

Sample code block:

.. code-block:: python

  # Add design variables, constraint, and objective and setup problem
  OAS_prob.add_desvar('wing.twist_cp', lower=-10., upper=15.)
  OAS_prob.add_constraint('wing_perf.CL', equals=0.5)
  OAS_prob.add_objective('wing_perf.CD', scaler=1e4)


4. Call setup() on the problem
------------------------------
Depending on the user-defined problem type, this setup function calls
:func:`OASProblem.setup_aero`,
:func:`OASProblem.setup_struct`, or
:func:`OASProblem.setup_aerostruct`.
Each of these methods is different, but they mainly organize the OpenMDAO
components for each of the disciplines in the correct manner and then
setup the OpenMDAO problem.

For aero-only, that means that the lifitng surfaces are added and linked together
so we can compute the entire AIC matrix.
For struct-only, we can set up each spar individually because they have no effect
on each other.
For aerostructural cases, we must take care to add the aerodynamic and structural
components in the correct groups within the problem.
The mesh setup and performance components are outside the coupled group, whereas
the FEM and VLM solvers are within the coupled group so we can converge
the coupled aerostructural system.

.. code-block:: python

  OAS_prob.setup()


5. Call run() to perform analysis/optimization
----------------------------------------------

Lastly, we call :func:`OASProblem.run` to finalize OpenMDAO setup and actually run the problem.
Here we actually add the design variables, constraints, and objective to the OpenMDAO problem.
We also set the optimization history recording options and save a model of the problem layout in an .html file.
Check your run directory for a new .html file and examine this to see your problem layout.

If `optimize = False` in the problem dictionary, then we perform analysis on the initial geometry.
If `optimize = True`, then we run optimization with the given formulation and optimizer selected.
The outputted results of the analysis or optimization are available after running by accessing
the variables as shown below:

.. code-block:: python

  # Actually run the problem
  OAS_prob.run()

  print("\nWing CL:", OAS_prob.prob['wing_perf.CL'])
  print("Wing CD:", OAS_prob.prob['wing_perf.CD'])

If you are unsure of where the variables are located, you can consult the .html file that contains
the problem layout to see the problem hierarchy.
