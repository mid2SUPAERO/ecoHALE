EcoHALE
=======

[![Build Status](https://travis-ci.org/mdolab/OpenAeroStruct.svg?branch=master)](https://travis-ci.org/mdolab/OpenAeroStruct)
[![Coverage Status](https://coveralls.io/repos/github/mdolab/OpenAeroStruct/badge.svg?branch=master)](https://coveralls.io/github/mdolab/OpenAeroStruct?branch=master)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/mdolab/OpenAeroStruct.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/mdolab/OpenAeroStruct/context:python)

EcoHALE is a low-fidelity multidisciplinary optimization tool for High Altitude Long Endurance (HALE) aircraft,
developed from OpenAeroStruct. While OpenAeroStruct is a tool for aerostructural optimization based on OpenMDAO,
a multidisciplinary design optimization framework.

The CO2 footprint optimization of a High Altitude Long Endurance (HALE) drone is studied. 
The  environmental  impact of a solar-powered drone comes mainly from the materials used and the building.
For this reason, this tool includes material choice optimization for the structure of the HALE in the MDO approach. 
The simple HALE model developed shows acceptable agreement with more complex models.

![Convergence Graphs](openaerostruct/docs/convergence_graphs.png)
![CADmodel](openaerostruct/docs/CADmodel.jpg)

Install EcoHALE version of OpenAeroStruct by cloning this repository and entering the folder it generates.
Then do:

`pip install -e .`

If you encounter any problems while using OpenAeroStruct, please create an issue on this GitHub repository.

OpenAeroStruct Documentation
----------------------------

Please see the [documentation](https://mdolab.github.io/OpenAeroStruct/) for more installation details, walkthroughs, and examples.

New features
------------

Improvements with respect to the original version of OpenAeroStruct:

   * Adapted for HALE drones.
   * Material choice among a discrete catalogue.
   * Battery modelling.
   * Solar panel modelling.
   * Buckling constraint.

Video Introduction
------------------

For a detailed introduction and a quick start tutorial please visit:

   https://youtu.be/s5952bI_Jw0

Version Information
-------------------

This version of OpenAeroStruct requires [OpenMDAO](https://github.com/OpenMDAO/openmdao) v2.5+.
If you are looking to use the previous version of OpenAeroStruct which uses OpenMDAO 1.7.4, use OpenAeroStruct v1.0 from [here](https://github.com/mdolab/OpenAeroStruct/releases).

How to use
----------

Please go to the folder [openaerostruct/HALE](https://github.com/mid2SUPAERO/ecoHALE/tree/dowloadEcohale/openaerostruct/HALE) and then:

   * The optimization can be launched from the file *fonctionPrincipale.py*. 
   * To modify some of the optimization parameters use *fonctionOptim.py*.
   * To compare the results of a multi-start optimization and check which of them have converged properly, use *seeResultsMain.py*.
   * The detailed results of one particular optimization can be analysed using *seeOptim.py*.
   * The file *CatiaParamGenerator.py* can be used to generate a TXT file that can be imported into *HALE.CATPart* in order to plot the aircraft.

For the paper entitled "HALE multidisciplinary ecodesign optimization with material selection" specific files have been created:

   * Section 3.1, "Mass minimization problem (validation withFBHALE)": 
     * *fonctionPrincipale_MASSminimization.py*
     * *fonctionOptim_MASSminimization.py*
     * *seeOptim_MASSminimization.py*
   * Section 3.2, "CO2 footprint minimization problem": 
     * *fonctionPrincipale_CO2minimization.py*
     * *fonctionOptim_CO2minimization.py*
     * *seeOptim_CO2minimization.py*
	
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