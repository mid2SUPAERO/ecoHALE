.. OpenAeroStruct documentation master file

OpenAeroStruct Documentation
==========================================

OpenAeroStruct is a lightweight tool that performs aerostructural optimization using OpenMDAO.
It couples a vortex-lattice method (VLM) and a 6 degrees of freedom 3-dimensional spatial beam model to simulate aerodynamic and structural analyses using lifting surfaces.
These simulations are wrapped with an optimizer using NASA's OpenMDAO framework.
The analysis and optimization results can be visualized using included tools, producing figures such as this:

.. image:: ../example.png

Installation
-----------------

To use OpenAeroStruct, you must first install OpenMDAO 2.3+ by installing via pip using:

.. code-block:: bash

    pip install openmdao

or by following the instructions at https://github.com/openmdao/openmdao. If you are unfamiliar with OpenMDAO and wish to modify the internals of OpenAeroStruct, you should examine the OpenMDAO documentation at http://openmdao.org/twodocs/versions/latest/index.html. The tutorials provided with OpenMDAO are helpful to understand the basics of using OpenMDAO to solve an optimization problem.

Next, clone the OpenAeroStruct repository:

.. code-block:: bash

    git clone https://github.com/mdolab/OpenAeroStruct.git

Then from within the OpenAeroStruct folder, pip install the package:

.. code-block:: bash

    pip install -e .

Lastly, there's an optional step to compile the Fortran to decrease the computational cost of running OpenAeroStruct. If you are using Linux, the default `config.mk` file should work. On Mac, you need to delete the original `config.mk` file and rename the `config-macOS.mk` file to `config.mk`.

You may need to modify the `PYTHON-CONFIG` option based on which Python version you're using. For example, if you're using Anaconda Python 2.7 or Python 3.6, you may need to change the value to `python2.7-config` or `python3.6-config` respectively. With the `config.mk` file set up correctly, call the following command from the root level of the OpenAroStruct directory:

.. code-block:: bash

    make

Note that the code will run without compiling the Fortran library, but it will run significantly faster when using Fortran.
The Fortran code has been tested extensively on Linux, partially on MacOS, and not at all on Windows.

Usage
-----

Check out these tests.

For each case, you can view the optimization results using `plot_all.py`. Examine its docstring for keyword information.

An example workflow would be:

.. code-block:: bash

    python run_aerostruct.py 1
    python plot_all.py aerostruct.db

The first command performs aerostructural optimization and the second visualizes the optimization history.

The keywords used for each file are explained in their respective docstrings at the top of the file.

Notes
-----

This current version of this repository has grown past the previous Matlab implementation. If you are looking for a Matlab-capable version, please see https://github.com/samtx/OpenAeroStruct for the latest version.

Known Issues
------------

* The increase in accuracy of results when using a cosine-spaced mesh is not as large as it should be.
* Aerostructural optimization sometimes fails to converge for certain geometries. The example provided in `run_aerostruct.py` should converge. The structural and aerodynamic values must make sense together, e.g. the beam thickness and radius must be able to support the aerodynamic loads.



Tutorials and Indices
======================

.. toctree::
   :maxdepth: 2

   installation.rst
   aero_walkthrough.rst
   struct_example.rst
   aerostructural_walkthrough.rst
   specialty_topics.rst
   v1_v2_conversion.rst
   how_to_contribute.rst

Check out the module index below to see the internal methods within each file and how they're used.

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
