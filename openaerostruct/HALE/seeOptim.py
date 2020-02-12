# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 17:06:34 2019

@author: e.duriez
"""
from openmdao.api import Problem, ScipyOptimizeDriver, SqliteRecorder, CaseReader
import matplotlib.pyplot as plt
import numpy as np


#cr = CaseReader("chord1.4/aerostructMrhoi505sk0.003sr0.00030000000000000003sn25tc0.05.db")
cr = CaseReader("aerostructMrhoi500sk0.0012sr0.0001sn60tc0.11ed5.db")

driver_cases = cr.list_cases('driver')

iterations=len(driver_cases)
#$iterations=3

mrho=[]
masse=[]
co2=[]
taper=[]
span=[]
chord=[]
chordTip=[]
surf=[]
surface0=[]
surface1=[]
sparThicknessRoot=[]
sparThicknessTip=[]
skinThicknessRoot=[]
skinThicknessTip=[]
failure=[]
power=[]
lift=[]
tOverC1=[]
tOverC2=[]
buckling=[]

for i in range(iterations):
#for i in range(350,380):
    case = cr.get_case(driver_cases[i])
    design_vars = case.get_design_vars()
    objective= case.get_objectives()
    constraints= case.get_constraints()
    mrho.append(design_vars['mrho'][0])
    masse.append(case.outputs['wing.structural_mass'][0])
    co2.append(objective['emitted_co2'][0])
    taper.append(case.inputs['wing.geometry.mesh.taper.taper'][0])
    span.append(case.inputs['wing.geometry.mesh.stretch.span'][0])
    chord.append(case.inputs['wing.geometry.mesh.scale_x.chord'][0])
    chordTip.append(case.inputs['wing.geometry.mesh.scale_x.chord'][-1])
    surface0.append(case.outputs['AS_point_0.coupled.wing.S_ref'][0])
#    surface1.append(case.outputs['AS_point_1.coupled.wing.S_ref'][0])
    sparThicknessRoot.append(design_vars['wing.spar_thickness_cp'][-1])
    sparThicknessTip.append(design_vars['wing.spar_thickness_cp'][0])
    skinThicknessRoot.append(design_vars['wing.skin_thickness_cp'][-1])
    skinThicknessTip.append(design_vars['wing.skin_thickness_cp'][0])
#    failure.append(constraints['AS_point_1.wing_perf.failure'][0])
#    power.append(constraints['AS_point_1.enough_power'][0])
#    lift.append(constraints['AS_point_1.L_equals_W'][0])
    failure.append(constraints['AS_point_1.wing_perf.failure'][0])
    power.append(constraints['AS_point_0.enough_power'][0])
    lift.append(constraints['AS_point_0.L_equals_W'][0])
    tOverC1.append(case.outputs['wing.geometry.t_over_c_cp'])
    tOverC2.append(case.outputs['wing.t_over_c'][0])
    buckling.append(constraints['AS_point_1.wing_perf.buckling'][0])

chordEnd=np.multiply(chord,taper)
doublemeanchord=np.add(chord,chordEnd)
meanchord=[x/2 for x in doublemeanchord]
surf=np.multiply(meanchord,span)



print(mrho)
#print(masse)
print(co2)

plt.semilogy(masse)
plt.xlabel('iteration')
plt.ylabel('mass')
#plt.xlim((0,150))

plt.show()

plt.plot([i*1000 for i in mrho])
plt.xlabel('iteration')
plt.ylabel('density (kg/m3)')
plt.ylim((450,2050))
#plt.xlim((0,250))

plt.show()

#/plt.plot(co2)
plt.semilogy([i*10000 for i in co2])
plt.xlabel('iteration')
plt.ylabel('co2 (kg)')
#plt.xlim((0,150))

plt.show()

plt.plot(taper)
plt.xlabel('iteration')
plt.ylabel('taper')
#plt.xlim((0,150))
plt.ylim((0,0.5))

plt.show()

plt.plot(span)
plt.xlabel('iteration')
plt.ylabel('span (m)')
#plt.xlim((0,150))

plt.show()

plt.plot(chord)
plt.xlabel('iteration')
plt.ylabel('chord (m)')
#plt.xlim((0,150))
#plt.ylim((0,10))

plt.show()

plt.plot(skinThicknessRoot)
plt.plot(skinThicknessTip)
plt.xlabel('iteration')
plt.ylabel('skin (mm)')
#plt.xlim((250,300))
#plt.ylim((0,4))

plt.show()

plt.plot([i/10 for i in sparThicknessRoot])
plt.plot([i/10 for i in sparThicknessTip])
plt.xlabel('iteration')
plt.ylabel('spar (mm)')
#plt.xlim((250,300))
#plt.ylim((0,4))

plt.show()

plt.plot(failure)
plt.xlabel('iteration')
plt.ylabel('failure')
#plt.xlim((250,300))
plt.ylim((-1,1))

plt.show()

plt.plot(power)
plt.xlabel('iteration')
plt.ylabel('power')
#plt.xlim((250,300))
#plt.ylim((0,4))

plt.show()

plt.plot(lift)
plt.xlabel('iteration')
plt.ylabel('lift')
#plt.xlim((250,300))
#plt.ylim((0,4))

plt.show()

plt.plot(tOverC1)
plt.xlabel('iteration')
plt.ylabel('toverc1')
#plt.xlim((250,300))
#plt.ylim((0,4))

plt.show()

plt.plot(tOverC2)
plt.xlabel('iteration')
plt.ylabel('toverc2')
#plt.xlim((250,300))
#plt.ylim((0,4))

plt.show()

plt.plot(buckling)
plt.xlabel('iteration')
plt.ylabel('buckling')
#plt.xlim((250,300))
plt.ylim((-1,1))

plt.show()

#plt.plot(chordTip)
#plt.xlabel('iteration')
#plt.ylabel('chord at wing tip')
#plt.xlim((0,150))
#
#plt.show()
#
#plt.plot(surf)
#plt.xlabel('iteration')
#plt.ylabel('surface aprox')
#plt.xlim((0,150))
#
#plt.show()
#
#plt.plot(surface0)
#plt.xlabel('iteration')
#plt.ylabel('surface point0')
#plt.xlim((0,150))
#
#plt.show()
#
#plt.plot(surface1)
#plt.xlabel('iteration')
#plt.ylabel('surface point1')
#plt.xlim((0,150))
#
#plt.show()
