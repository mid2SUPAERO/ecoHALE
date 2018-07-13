.. OpenAeroStruct documentation master file

OpenAeroStruct Documentation
============================

OpenAeroStruct is a lightweight tool that performs aerostructural optimization using OpenMDAO.
It couples a vortex-lattice method (VLM) and a 6 degrees of freedom 3-dimensional spatial beam model to simulate aerodynamic and structural analyses using lifting surfaces.
These simulations are wrapped with an optimizer using NASA's OpenMDAO framework.
The analysis and optimization results can be visualized using included tools, producing figures such as this:

.. figure:: example.png
   :align: center
   :width: 100%
   :alt: sample visualization of aerostructural system

Walkthroughs and Examples
=========================

.. toctree::
   :maxdepth: 2

   installation.rst
   quick_example.rst
   aero_walkthrough.rst
   struct_example.rst
   aerostructural_walkthrough.rst
   specialty_topics.rst

Other Useful Docs
=================

.. toctree::
   :maxdepth: 1

   v1_v2_conversion.rst
   how_to_contribute.rst
   _srcdocs/index.rst

   Notes
   -----

   This current version of this repository has grown past the previous Matlab implementation. If you are looking for a Matlab-capable version, please see https://github.com/samtx/OpenAeroStruct for the latest version.

   Known Issues
   ------------

   * The increase in accuracy of results when using a cosine-spaced mesh is not as large as it should be.
   * Aerostructural optimization sometimes fails to converge for certain geometries. The example provided in `run_aerostruct.py` should converge. The structural and aerodynamic values must make sense together, e.g. the beam thickness and radius must be able to support the aerodynamic loads.
