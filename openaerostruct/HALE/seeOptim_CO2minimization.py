# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 17:06:34 2019

@author: e.duriez and Victor M. Guadano
"""


from openmdao.api import Problem, ScipyOptimizeDriver, SqliteRecorder, CaseReader
import matplotlib.pyplot as plt
import numpy as np


cr = CaseReader("aerostructMrhoi600sk0.002sr0.002sn100tc0.13.db")  # File to read


driver_cases = cr.list_cases('driver')

iterations=len(driver_cases)

# Values for all the iterations:
co2=[]  # CO2 emissions [10^4 kg]

mrho=[]  # Material density [g/cm3]
taper=[]  # Taper ratio
span=[]  # Span [m]
chord=[]  # Root chord [m]
surf=[]  # Wing surface [m2]

sparThickness=[]  # Spar thickness distribution [m]
skinThickness=[]  # Skin thickness distribution [m]
sparThicknessRoot=[]  # Spar thickness at root section [mm]
sparThicknessTip=[]  # Spar thickness at tip section [mm]
skinThicknessRoot=[]  # Skin thickness at root section [mm]
skinThicknessTip=[]  # Skin thickness at tip section [mm] 
sparThickness_cp=[]  # Spar thickness at control points [mm]
skinThickness_cp=[]  # Skin thickness at control points [mm]

t_c=[]  # Thickness-over-chord ratio distribution 
t_c_cp=[]  # Thickness-over-chord ratio at control points 

twist=[]  # Twist distribution [deg]
twist_cp=[]  # Twist at control points [deg/10]

failure=[]  # Failure constraint
power=[]  # Power constraint
lift=[]  # Lift-equals-weight constarint
buckling=[]  # Buckling constraint

cl=[]  # Lift coefficient
cd=[]  # Drag coeffient

point_mass_locations=[]  # Position of each motor [m]
motor_location=[]  # Motor location over semi-span ratio

totalWeight=[]  # Total weight [N]
struct_mass=[]  # Structural mass [kg]
PVmass=[]  # Mass of solar panels and batteries [kg]
point_masses=[]  # Mass of each motor [kg]

mesh=[]  # Structural mess [m]
forces=[]  # Aerodynamic forces [N]
vm=[]  # von Mises stresses [Pa]
widths=[]  # Mesh widths (spanwise direction) [m]

for i in range(iterations):
    case = cr.get_case(driver_cases[i])
    design_vars = case.get_design_vars()
    objective= case.get_objectives()
    constraints= case.get_constraints()
    
    co2.append(objective['emitted_co2'][0])
    
    mrho.append(design_vars['mrho'])
    taper.append(case.inputs['wing.geometry.mesh.taper.taper'][0])
    span.append(case.inputs['wing.geometry.mesh.stretch.span'][0])
    chord.append(case.inputs['wing.geometry.mesh.scale_x.chord'][0])
    surf.append(case.outputs['AS_point_0.coupled.wing.S_ref'][0])
    
    sparThickness.append(case.outputs['wing.spar_thickness'][0])
    skinThickness.append(case.outputs['wing.skin_thickness'][0])    
    sparThicknessRoot.append(design_vars['wing.spar_thickness_cp'][-1])
    sparThicknessTip.append(design_vars['wing.spar_thickness_cp'][0])
    skinThicknessRoot.append(design_vars['wing.skin_thickness_cp'][-1])
    skinThicknessTip.append(design_vars['wing.skin_thickness_cp'][0])
    sparThickness_cp.append(design_vars['wing.spar_thickness_cp'])
    skinThickness_cp.append(design_vars['wing.skin_thickness_cp'])
    
    t_c.append(case.outputs['wing.t_over_c'][0])
    t_c_cp.append(case.outputs['wing.geometry.t_over_c_cp'])
    
    twist.append(case.outputs['wing.geometry.twist'][0])
    twist_cp.append(design_vars['wing.twist_cp'])
    
    failure.append(constraints['AS_point_1.wing_perf.failure'][0])
    power.append(constraints['AS_point_0.enough_power'][0])
    lift.append(constraints['AS_point_0.L_equals_W'][0])
    buckling.append(constraints['AS_point_1.wing_perf.buckling'][0])
    
    cl.append(case.outputs['AS_point_0.CL'][0])
    cd.append(case.outputs['AS_point_0.CD'][0])
    
    point_mass_locations.append(case.outputs['point_mass_locations'][0])
    motor_location.append(design_vars['motor_location'])
    
    totalWeight.append(case.outputs['AS_point_0.total_perf.total_weight'][0])
    struct_mass.append(case.outputs['wing.structural_mass'][0])
    PVmass.append(case.outputs['AS_point_0.total_perf.PV_mass'][0])
    point_masses.append(case.outputs['point_masses'][0])

    mesh.append(case.outputs['AS_point_0.coupled.wing.def_mesh'][0])
    forces.append(case.outputs['AS_point_0.coupled.aero_states.wing_sec_forces'])
    vm.append(case.outputs['AS_point_0.wing_perf.vonmises'][0])
    widths.append(case.outputs['AS_point_0.coupled.wing.widths'])
    
chordEnd=np.multiply(chord,taper)
doublemeanchord=np.add(chord,chordEnd)
meanchord=[x/2 for x in doublemeanchord]
surf=np.multiply(meanchord,span)

# Usuful final values
lift_dist = np.sum(forces[-1], axis=0)[:,2]/widths[-1]  # Final lift spanwise distribution [N]
AR = span[-1]**2/surf[-1]   # Aspect Ratio
eff = cl[-1]**(3/2)/cd[-1]    # Aerodynamic efficiency CL^(3/2)/CD
total_mass = totalWeight[-1]/9.81   # Total final mass [kg]
total_mass_it = [x/9.81 for x in totalWeight]   # Total mass per iteration [kg]

mrho_kgm3 = [x*1000 for x in mrho]      # Material density per iteration [kg/m^3]
co2_kg = [x*1e4 for x in co2]     # Total CO2 emissions per iteration [kg]


# CO2 emissions convergence graph 
plt.plot(co2_kg)
plt.xlabel('iteration')
plt.ylabel('CO2 emissions (kg)')
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
#plt.xlim((0,100))
#plt.ylim((5000,27500))
#plt.savefig('co2emissions.png', dpi=300)
plt.show()


# Total mass convergence graph 
plt.plot(total_mass_it)
plt.xlabel('iteration')
plt.ylabel('total mass (kg)')
#plt.xlim((0,500))
#plt.ylim((160,320))
#plt.savefig('totalMass.png', dpi=300)
plt.show()


# Material density convergence graph 
if mrho_kgm3[-1][1] == mrho_kgm3[1][1]: # Nmaterial = 1
    mrho_kgm3_1mat = [x[0] for x in mrho_kgm3]
    plt.plot(mrho_kgm3_1mat)
    plt.xlabel('iteration')
    plt.ylabel('material density (kg/m3)')
    #plt.xlim((0,450))
    #plt.ylim((480,600))
    #plt.savefig('materialDensity1mat.png', dpi=300)
else:                                   # Nmaterial = 2
    plt.plot(mrho_kgm3)
    plt.xlabel('iteration')
    plt.ylabel('material density (kg/m3)')
    #plt.xlim((0,100))
    #plt.ylim((475,700))
    plt.legend(['spar', 'skin'])
    #plt.savefig('materialDensity.png', dpi=300)
plt.show()


# Taper ratio convergence graph 
plt.plot(taper)
plt.xlabel('iteration')
plt.ylabel('taper')
#plt.xlim((0,500))
#plt.ylim((0.295,0.330))
#plt.savefig('taper.png', dpi=300)
plt.show()


# Span convergence graph 
plt.plot(span)
plt.xlabel('iteration')
plt.ylabel('span (m)')
#plt.xlim((0,500))
#plt.ylim((25,65))
#plt.savefig('span.png', dpi=300)
plt.show()


# Root chord convergence graph 
plt.plot(chord)
plt.xlabel('iteration')
plt.ylabel('root chord (m)')
#plt.xlim((0,500))
#plt.ylim((1.30,1.70))
#plt.savefig('chord.png', dpi=300)
plt.show()


# Skin thickness convergence graph (root and tip)
plt.plot(skinThicknessRoot, label='root')
plt.plot(skinThicknessTip, label='tip')
plt.xlabel('iteration')
plt.ylabel('skin thickness (mm)')
#plt.xlim((0,500))
#plt.ylim((0,5))
plt.legend()
#plt.savefig('skin.png', dpi=300)
plt.show()


# Spar thickness convergence graph (root and tip)
plt.plot(sparThicknessRoot, label='root')
plt.plot(sparThicknessTip, label='tip')
plt.xlabel('iteration')
plt.ylabel('spar thickness (mm)')
#plt.xlim((0,500))
#plt.ylim((0,4))
plt.legend()
#plt.savefig('spar.png', dpi=300)
plt.show()


# Failure constarint convergence graph
plt.plot(failure)
plt.xlabel('iteration')
plt.ylabel('failure')
#plt.xlim((250,300))
#plt.ylim((-1,1))
plt.show()


# Power constarint convergence graph
plt.plot(power)
plt.xlabel('iteration')
plt.ylabel('power')
#plt.xlim((250,300))
#plt.ylim((0,4))
plt.show()


# Lift-equals-weight constarint convergence graph
plt.plot(lift)
plt.xlabel('iteration')
plt.ylabel('lift')
#plt.xlim((250,300))
#plt.ylim((0,4))
plt.show()


# Buckling constarint convergence graph
plt.plot(buckling)
plt.xlabel('iteration')
plt.ylabel('buckling')
#plt.xlim((250,300))
#plt.ylim((-1,1))
plt.show()


# Twist distribution
plt.plot(mesh[-1][:,1],twist[-1],'b')
plt.plot(-mesh[-1][:,1],twist[-1],'b')
plt.xlabel('span (m)')
plt.ylabel('twist (deg)')
plt.xlim((-span[-1]/2,span[-1]/2))
plt.show()


# Spar thickness distribution
plt.plot(mesh[-1][1:,1],sparThickness[-1],'b')
plt.plot(-mesh[-1][1:,1],sparThickness[-1],'b')
plt.xlabel('span (m)')
plt.ylabel('spar thickness (m)')
plt.xlim((-span[-1]/2,span[-1]/2))
plt.show()


# Skin thickness distribution
plt.plot(mesh[-1][1:,1],skinThickness[-1],'b')
plt.plot(-mesh[-1][1:,1],skinThickness[-1],'b')
plt.xlabel('span (m)')
plt.ylabel('skin thickness (m)')
plt.xlim((-span[-1]/2,span[-1]/2))
plt.show()


# Thickness-to-chord ratio distribution
plt.plot(mesh[-1][1:,1],t_c[-1],'b')
plt.plot(-mesh[-1][1:,1],t_c[-1],'b')
plt.xlabel('span (m)')
plt.ylabel('t/c')
plt.xlim((-span[-1]/2,span[-1]/2))
plt.show()


# Lift distribution
ellip_lift = 2*totalWeight[-1]/(0.5*span[-1]*np.pi)*np.sqrt(1-(-mesh[-1][1:,1]/(0.5*span[-1]))**2)
plt.plot(mesh[-1][1:,1],lift_dist,'b')
plt.plot(-mesh[-1][1:,1],lift_dist,'b')
plt.plot(mesh[-1][1:,1],ellip_lift,'c--',label='elliptical')
plt.plot(-mesh[-1][1:,1],ellip_lift,'c--')
plt.xlabel('span (m)')
plt.ylabel('lift (N)')
plt.xlim((-span[-1]/2,span[-1]/2))
plt.legend()
plt.show()