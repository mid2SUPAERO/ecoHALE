# OpenAeroStruct

OpenAeroStruct is a lightweight Python tool to perform aerostructural optimization of lifting surfaces using OpenMDAO. It uses a vortex lattice method (VLM) expanded from Phillips' Modern Adaption of Prandtl's Classic Lifting Line Theory (http://arc.aiaa.org/doi/pdfplus/10.2514/2.2649) for the aerodynamics analysis and a spatial beam model with 6-DOF per element for the structural analysis.

![Optimized CRM-type wing with 30 panels](/example.png?raw=true "Example Optimization Result and Visualization")

## Installation

To use OpenAeroStruct, you must first install OpenMDAO 1.7.0 by following the instructions here: https://github.com/openmdao/openmdao. If you are unfamiliar with OpenMDAO and wish to modify the internals of OpenAeroStruct, you should examine the OpenMDAO documentation at http://openmdao.readthedocs.io/en/1.7.0/. The tutorials provided with OpenMDAO, especially The Sellar Problem, are helpful to understand the basics of using OpenMDAO to solve an optimization problem.

Next, clone this repository:

    git clone https://github.com/johnjasa/OpenAeroStruct.git

Lastly, from within the OpenAeroStruct folder, make the Fortran files:

    make

Note that the code will run without compiling the Fortran library, but it will run significantly faster when using Fortran.

Note that the master branch is a development branch and may be unstable. Using a stable release is suggested. To do so, checkout a release using the git command:

    git checkout v0.2.0
    
This will use v0.2.0 of OpenAeroStruct, the most recent version.

## Usage

`run_vlm.py` is for aero-only analysis and optimization. It can use a single lifting surface or multiple separate lifting surfaces.

`run_spatialbeam.py` is for structural-only analysis and optimization. It can use a single structural component or multiple structural components, where each component represents a spar within a lifting surface.

`run_aerostruct.py` performs aerostructural analysis and optimization.


For each case, you can view the optimization results using `plot_all.py`. Examine its docstring for keyword information.

An example workflow would be:

    python run_aerostruct.py 1
    python plot_all.py as

The keywords used for each file are explained in their respective docstrings at the top of the file.

## Known Issues

* Aerostructural optimization sometimes fails to converge for certain geometries. The example provided in `run_aerostruct.py` should converge.
* Aerostructural optimization using multiple lifting surfaces does not converge.
* The residual of the structural system solution for very large problems is sometimes too large and prevents convergence of the optimization problem.
* Internal documentation is lacking.
* `plot_all.py` does not correctly display multiple structural components, but does work for multiple surfaces if using only aerodynamic optimization.
* Multiple surface optimization for structures is not optimally coded.
