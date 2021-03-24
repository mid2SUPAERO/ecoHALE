OpenAeroStruct
==============

[![Build Status](https://travis-ci.org/mdolab/OpenAeroStruct.svg?branch=master)](https://travis-ci.org/mdolab/OpenAeroStruct)
[![Coverage Status](https://coveralls.io/repos/github/mdolab/OpenAeroStruct/badge.svg?branch=master)](https://coveralls.io/github/mdolab/OpenAeroStruct?branch=master)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/mdolab/OpenAeroStruct.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/mdolab/OpenAeroStruct/context:python)

OpenAeroStruct is a lightweight tool that performs aerostructural optimization using OpenMDAO.
It couples a vortex-lattice method (VLM) and a 6 degrees of freedom 3-dimensional spatial beam model to simulate aerodynamic and structural analyses using lifting surfaces.
These simulations are wrapped with an optimizer using NASA's OpenMDAO framework.
The analysis and optimization results can be visualized using included tools, producing figures such as this:

![Example](openaerostruct/docs/example.png)

Install OpenAeroStruct by cloning this repository and entering the folder it generates.
Then do:

`pip install -e .`

If you encounter any problems while using OpenAeroStruct, please create an issue on this GitHub repository.

Documentation
-------------

Please see the [documentation](https://mdolab.github.io/OpenAeroStruct/) for more installation details, walkthroughs, and examples.

Citation
--------

For more background, theory, and figures, see the [OpenAeroStruct journal article](http://mdolab.engin.umich.edu/sites/default/files/OAS_SMO_preprint_0.pdf).
Please cite this article when using OpenAeroStruct in your research or curricula.

John P. Jasa, John T. Hwang, and Joaquim R. R. A. Martins. "Open-source coupled aerostructural optimization using Python." Structural and Multidisciplinary Optimization 57.4 (2018): 1815-1827. DOI: 10.1007/s00158-018-1912-8

```
@article{Jasa2018a,
	Author = {John P. Jasa and John T. Hwang and Joaquim R. R. A. Martins},
	Doi = {10.1007/s00158-018-1912-8},
	Journal = {Structural and Multidisciplinary Optimization},
	Month = {April},
	Number = {4},
	Pages = {1815--1827},
	Publisher = {Springer},
	Title = {Open-source coupled aerostructural optimization using {Python}},
	Volume = {57},
	Year = {2018}}
```

If using the wingbox model, please cite the following conference paper.

Shamsheer S. Chauhan and Joaquim R. R. A. Martins, “Low-Fidelity Aerostructural Optimization of Aircraft Wings with a Simplified Wingbox Model Using OpenAeroStruct,” Proceedings of the 6th International Conference on Engineering Optimization, EngOpt 2018, Springer, Lisbon, Portugal, September 2018, pp. 418–431. doi:10.1007/978-3-319-97773-7 38

```
@inproceedings{Chauhan2018b,
	Author = {Shamsheer S. Chauhan and Joaquim R. R. A. Martins},
	Address = {Lisbon, Portugal},
	Booktitle = {Proceedings of the 6th International Conference on Engineering Optimization, EngOpt 2018},
	Doi = {10.1007/978-3-319-97773-7_38},
	Pages = {418-431},
	Publisher = {Springer},
	Title = {Low-Fidelity Aerostructural Optimization of Aircraft Wings with a Simplified Wingbox Model Using {OpenAeroStruct}},
	Year = {2018}}
```

Version Information
-------------------
This version of OpenAeroStruct requires [OpenMDAO](https://github.com/OpenMDAO/openmdao) v2.5+.
If you are looking to use the previous version of OpenAeroStruct which uses OpenMDAO 1.7.4, use OpenAeroStruct v1.0 from [here](https://github.com/mdolab/OpenAeroStruct/releases).

License
-------
Copyright 2018 MDO Lab

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

ecoHALE tutorial
-------
The CO2 footprint optimization of a solar-powered High Altitude Long Endurance (HALE) drone is studied.
This optimization is performed using a modified version of OpenAeroStruct. 
This version includes material choice optimization for the structure of the HALE in the MDO approach. 
The simple HALE model developed shows acceptable agreement with more complex models.

Tutorial:
https://youtu.be/s5952bI_Jw0

![Convergence Graphs](openaerostruct/docs/convergence_graphs.png)
![CADmodel](openaerostruct/docs/CADmodel.jpg)