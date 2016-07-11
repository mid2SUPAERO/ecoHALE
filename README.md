# OpenAeroStruct

OpenAeroStruct is a lightweight Python tool to perform aerostructural optimization of an aircraft using OpenMDAO. It uses a vortex lattice method (VLM) for the aerodynamics analysis and a spatial beam model with 6-DOF per element for the structural analysis.

## Installation

To use OpenAeroStruct, you must first install OpenMDAO by following the instructions here: https://github.com/openmdao/openmdao. If you are unfamiliar with OpenMDAO and wish to modify the internals of OpenAeroStruct, you should examine the OpenMDAO documentation at http://openmdao.readthedocs.io/en/1.7.0/. The tutorials, especially The Sellar Problem, are helpful to understand the basics of an OpenMDAO optimization.

Next, clone the OpenAeroStruct repository:

    git clone https://github.com/hwangjt/OpenAeroStruct.git

Lastly, from within the OpenAeroStruct folder, make the Fortran files:

    make

Note that the code will run without compiling the Fortran library, but it will run significantly faster with the Fortran files compiled. 


## Usage

`run_vlm.py` is for aero-only analysis and optimization. It can use a single lifting surface or multiple separate lifting surfaces.

`run_spatialbeam.py` is for structural-only analysis and optimization. It can use a single structural component or multiple structural components.

`run_aerostruct.py` performs aerostructural analysis and optimization.


For each case, you can view the optimization results using `plot_all.py`. Examine its docstring for keyword information.

An example workflow would be:

    python run_aerostruct.py 1
    python plot_all.py as
    
## Known Issues

* Cannot use multiple lifting surfaces for aerostructural optimization.
* Aerostructural optimization sometimes fails to converge.
* The residual of the structural system solution is sometimes too large and prevents convergence of the optimization problem.
* Internal documentation is lacking.



