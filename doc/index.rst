.. OpenAeroStruct documentation master file, created by
   sphinx-quickstart on Wed Jul 13 13:18:23 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

OpenAeroStruct Documentation
==========================================

OpenAeroStruct is a lightweight tool to perform aerostructural optimization using OpenMDAO.
It couples a vortex-lattice method (VLM) and a 6 degrees of freedom 3-dimensional spatial beam model to simulate the aerodynamic and structural properties of lifting surfaces.
These simulations are wrapped in an optimizer using NASA's OpenMDAO framework.
The analysis and optimization results can be visualized using included tools, producing figures such as this:

[put in figure here]

To get started, you can try running the aerodynamic analysis by entering the following command into a terminal: `python run_vlm.py 0`.
This does not perform optimization, but simply obtains the aerodynamic properties of a simple rectangular wing.


.. todo:: Add information here. In the meantime, check out the Module Index below.

.. toctree::
   :maxdepth: 2



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
