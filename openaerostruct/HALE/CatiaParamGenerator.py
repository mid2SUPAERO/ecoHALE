# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 19:44:19 2020

@author: Victor M. Guadano
"""

from openmdao.api import Problem, ScipyOptimizeDriver, SqliteRecorder, CaseReader
import matplotlib.pyplot as plt
import numpy as np
import os

cr = CaseReader("aerostructMrhoi505sk0.004sr0.00030000000000000003sn50tc0.13.db")
#cr = CaseReader("aerostructMrhoi505sk0.002sr0.0001sn50tc0.13.db")

driver_cases = cr.list_cases('driver')

iterations=len(driver_cases)

taper=[]
span=[]
t_c=[]
rootChord=[]
point_masses=[]
point_mass_locations=[]
engine_location=[]
twist=[]
mesh=[]
sparThickness=[]
skinThickness=[]

for i in range(iterations):
#for i in range(350,380):
    case = cr.get_case(driver_cases[i])
    design_vars = case.get_design_vars()
    objective= case.get_objectives()
    constraints= case.get_constraints()
    taper.append(case.inputs['wing.geometry.mesh.taper.taper'][0])
    span.append(case.inputs['wing.geometry.mesh.stretch.span'][0])
    rootChord.append(case.inputs['wing.geometry.mesh.scale_x.chord'][0])
    point_masses.append(case.outputs['point_masses'][0])
    point_mass_locations.append(case.outputs['point_mass_locations'][0])
    engine_location.append(design_vars['engine_location'][0])
    twist.append(case.outputs['wing.geometry.twist'][0])
    mesh.append(case.outputs['AS_point_0.coupled.wing.def_mesh'][0])
    sparThickness.append(case.outputs['wing.spar_thickness'][0])
    skinThickness.append(case.outputs['wing.skin_thickness'][0])
    t_c.append(case.outputs['wing.t_over_c'][0])
    

f = open("CatiaParam.txt","w+")

position = np.zeros(mesh[-1].shape[0])
chord = np.zeros(mesh[-1].shape[0])

for i in range(mesh[-1].shape[0]):
    position[i] = mesh[-1][i][1]
    chord[i] = rootChord[-1] * ((taper[-1] - 1) / (span[-1] / 2) * (-1) * position[i] + 1)
    
for i in range(mesh[-1].shape[0]):    
    f.write("Position" + str(i) + '\t' + "<" + str(1e3 * position[position.shape[0] - 1 - i]) + "mm>" + os.linesep)
    f.write("Chord" + str(i) + '\t' + "<" + str(1e3 * chord[position.shape[0] - 1 - i]) + "mm>" + os.linesep)
    f.write("Twist" + str(i) + '\t' + "<" + str(twist[-1][position.shape[0] - 1 - i]) + "deg>" + os.linesep)
    
    if i < (mesh[-1].shape[0] - 1):
        f.write("t_over_c" + str(i) + '\t' + "<" + str(t_c[-1][position.shape[0] - 2 - i]) + ">" + os.linesep)
        f.write("SparThickness" + str(i) + '\t' + "<" + str(1e3 * sparThickness[-1][position.shape[0] - 2 - i]) + "mm>" + os.linesep)
        f.write("SkinThickness" + str(i) + '\t' + "<" + str(1e3 * skinThickness[-1][position.shape[0] - 2 - i]) + "mm>" + os.linesep)
    else:
        f.write("t_over_c" + str(i) + '\t' + "<" + str(t_c[-1][0] * position[1] / position[0]) + ">" + os.linesep)
        f.write("SparThickness" + str(i) + '\t' + "<" + str(1e3 * sparThickness[-1][0] * position[1] / position[0]) + "mm>" + os.linesep)
        f.write("SkinThickness" + str(i) + '\t' + "<" + str(1e3 * skinThickness[-1][0] * position[1] / position[0]) + "mm>" + os.linesep)

f.write("EnginePosition" + '\t' + "<" + str(1e3 * point_mass_locations[-1][1]) + "mm>" + os.linesep)
f.write("Span" + '\t' + "<" + str(1e3 * span[-1]) + "mm>" + os.linesep)
f.write("Taper" + '\t' + "<" + str(taper[-1]) + ">" + os.linesep)    
    
f.close()