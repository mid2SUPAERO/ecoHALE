# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 17:06:34 2019

@author: e.duriez
"""
from openmdao.api import Problem, ScipyOptimizeDriver, SqliteRecorder, CaseReader
import matplotlib.pyplot as plt
import numpy as np

cr = CaseReader("aerostructMrhoi555sk0.002sr0.00030000000000000003sn50tc0.13.db")

driver_cases = cr.list_cases('driver')

iterations=len(driver_cases)

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
cl=[]
cd=[]
efficiency=[]
point_masses=[]
point_mass_locations=[]
engine_location=[]
totalWeight=[]
PVmass=[]
twist=[]
twistcp=[]
mesh=[]
chordcp=[]
forces=[]
loads=[]
vm=[]
sparThickness=[]
skinThickness=[]
tbs=[]
widths=[]

for i in range(iterations):
#for i in range(350,380):
    case = cr.get_case(driver_cases[i])
    design_vars = case.get_design_vars()
    objective= case.get_objectives()
    constraints= case.get_constraints()
    mrho.append(design_vars['mrho'])
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
    cl.append(case.outputs['AS_point_0.CL'][0])
    cd.append(case.outputs['AS_point_0.CD'][0])
    point_masses.append(case.outputs['point_masses'][0])
    point_mass_locations.append(case.outputs['point_mass_locations'][0])
    engine_location.append(design_vars['engine_location'][0])
    totalWeight.append(case.outputs['AS_point_0.total_perf.total_weight'][0])
    PVmass.append(case.outputs['AS_point_0.total_perf.PV_mass'][0])
    twistcp.append(design_vars['wing.twist_cp'])
    twist.append(case.outputs['wing.geometry.twist'][0])
    mesh.append(case.outputs['AS_point_0.coupled.wing.def_mesh'][0])
    chordcp.append(design_vars['wing.chord_cp'])
    loads.append(case.outputs['AS_point_0.coupled.wing_loads.loads'][0])
    forces.append(case.outputs['AS_point_0.coupled.aero_states.wing_sec_forces'])
    vm.append(case.outputs['AS_point_0.wing_perf.vonmises'][0])
    sparThickness.append(case.outputs['wing.spar_thickness'][0])
    skinThickness.append(case.outputs['wing.skin_thickness'][0])
    widths.append(case.outputs['AS_point_0.coupled.wing.widths'])
    
chordEnd=np.multiply(chord,taper)
doublemeanchord=np.add(chord,chordEnd)
meanchord=[x/2 for x in doublemeanchord]
surf=np.multiply(meanchord,span)

lift_dist = np.sum(forces[-1], axis=0)[:,2]/widths[-1]
AR = span[-1]**2/surface0[-1]
cl_cd = cl[-1]**(3/2)/cd[-1]
total_mass = totalWeight[-1]/9.81

#print(mrho)
#print(masse)
#print(co2)

plt.semilogy(masse)
plt.xlabel('iteration')
plt.ylabel('mass')
#plt.xlim((0,150))

plt.show()

plt.plot(mrho)
plt.xlabel('iteration')
plt.ylabel('mrho')
plt.ylim((0.5,0.6))
#plt.xlim((0,250))

plt.show()

#/plt.plot(co2)
plt.semilogy(co2)
plt.xlabel('iteration')
plt.ylabel('co2')
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
plt.ylabel('span')
#plt.xlim((0,150))

plt.show()

plt.plot(chord)
plt.xlabel('iteration')
plt.ylabel('chord')
#plt.xlim((0,150))
plt.ylim((0,10))

plt.show()

plt.plot(skinThicknessRoot)
plt.plot(skinThicknessTip)
plt.xlabel('iteration')
plt.ylabel('skin')
#plt.xlim((250,300))
#plt.ylim((0,4))

plt.show()

plt.plot(sparThicknessRoot)
plt.plot(sparThicknessTip)
plt.xlabel('iteration')
plt.ylabel('spar')
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

plt.plot(mesh[-1][:,1],twist[-1],'b')
plt.plot(-mesh[-1][:,1],twist[-1],'b')
plt.xlabel('span')
plt.ylabel('twist')
plt.xlim((-span[-1]/2,span[-1]/2))
plt.show()

plt.plot(mesh[-1][1:,1],sparThickness[-1],'b')
plt.plot(-mesh[-1][1:,1],sparThickness[-1],'b')
plt.xlabel('span')
plt.ylabel('spar thickness')
plt.xlim((-span[-1]/2,span[-1]/2))
plt.show()

plt.plot(mesh[-1][1:,1],skinThickness[-1],'b')
plt.plot(-mesh[-1][1:,1],skinThickness[-1],'b')
plt.xlabel('span')
plt.ylabel('skin thickness')
plt.xlim((-span[-1]/2,span[-1]/2))
plt.show()

plt.plot(mesh[-1][1:,1],tOverC2[-1],'b')
plt.plot(-mesh[-1][1:,1],tOverC2[-1],'b')
plt.xlabel('span')
plt.ylabel('t/c')
plt.xlim((-span[-1]/2,span[-1]/2))
plt.show()

ellip_lift = 2*totalWeight[-1]/(0.5*span[-1]*np.pi)*np.sqrt(1-(-mesh[-1][1:,1]/(0.5*span[-1]))**2)

plt.plot(mesh[-1][1:,1],lift_dist,'b')
plt.plot(-mesh[-1][1:,1],lift_dist,'b')
plt.plot(mesh[-1][1:,1],ellip_lift,'c--')
plt.plot(-mesh[-1][1:,1],ellip_lift,'c--')
plt.xlabel('span')
plt.ylabel('lift')
plt.xlim((-span[-1]/2,span[-1]/2))
plt.show()
