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

These first few doc pages go into detail about how to set up and run a problem in OpenAeroStruct.
Please review these at a minimum to understand how aerodynamic, structural, and aerostructural problems are constructed.

.. toctree::
   :maxdepth: 2

   installation.rst
   quick_example.rst
   aero_walkthrough.rst
   struct_example.rst
   aerostructural_walkthrough.rst

Once you have reviewed and understand these basic examples, you can move on to these more advanced examples below.

.. toctree::
   :maxdepth: 2

   more_examples.rst

Other Useful Docs
=================

.. toctree::
   :maxdepth: 2

   specialty_topics.rst
   v1_v2_conversion.rst
   how_to_contribute.rst

Source Docs
===========

.. toctree::
   :maxdepth: 1

   _srcdocs/index.rst

Notes
=====

This current version of this repository has grown past the previous Matlab implementation. If you are looking for a Matlab-capable version, please see https://github.com/samtx/OpenAeroStruct for the latest version.

Known Issues
============

* The increase in accuracy of results when using a cosine-spaced mesh is not as large as it should be.
* Aerostructural optimization sometimes fails to converge for certain geometries. The example provided in `run_aerostruct.py` should converge. The structural and aerodynamic values must make sense together, e.g. the beam thickness and radius must be able to support the aerodynamic loads.
