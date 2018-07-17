.. _Installation:

Installation
============

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

You may need to modify the `PYTHON-CONFIG` option based on which Python version you're using. For example, if you're using Anaconda Python 2.7 or Python 3.6, you may need to change the value to `python2.7-config` or `python3.6-config` respectively. With the `config.mk` file set up correctly, navigate to the `openaerostructr/fortran` folder and run the following command:

.. code-block:: bash

    make

Note that the code will run without compiling the Fortran library, but it will run significantly faster when using Fortran.
The Fortran code has been tested extensively on Linux, partially on MacOS, and not at all on Windows.
