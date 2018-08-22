.. _User_specified_sref:

Specifying a Reference Area
=============================

In order to use a specific reference area for calculating coefficients, rather than the default approach of area-weighted averaging of each lifting surface, a few modifications must be done. First, the ``user_supplied_S_ref`` flag needs to be set to ``True``, which can be found in ``total_aero_performance.py`` under the functionals folder. 

.. code-block:: python

  user_supplied_S_ref = True

Next, a new independent variable called ``S_ref_total`` needs to be created in the run script. Here it is set to the value of ``areaRef``. Then, for each aerodynamic point, this variable needs to be connected to the one in ``TotalAeroPerformance``.

.. code-block:: python

  indep_var_comp.add_output('S_ref_total', val=areaRef, units='m**2')
  prob.model.connect('S_ref_total', point_name + '.S_ref_total')