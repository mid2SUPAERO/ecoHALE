
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 13:08:36 2019

@author: e.duriez
"""
from openmdao.api import CaseReader
import matplotlib.pyplot as plt
import numpy as np

#define x0
spanRange=np.arange(60,90,10) #3
tcRange=np.arange(0.09,0.18,0.04) #3
skinRange=np.arange(0.0025,0.004,0.0005) #3
sparRange=np.arange(0.0001,0.0004,0.0001) #3


#50
#0.08
#0.001
#0.001
#spanRange=[25]
#tcRange=[0.05]
#skinRange=[0.003]
#sparRange=[0.00030000000000000003]

caseArray=np.zeros((len(skinRange),len(sparRange),len(spanRange),len(tcRange),4),dtype=object)
for i in range(0,len(skinRange),1):
    for j in range(0,len(sparRange),1):
        for k in range(0,len(spanRange),1):
            for l in range(0,len(tcRange),1):
                caseArray[i,j,k,l,]=[skinRange[i],sparRange[j],spanRange[k],tcRange[l]]


cases=np.reshape(caseArray,(len(skinRange)*len(sparRange)*len(spanRange)*len(tcRange),4))
print(cases)


##!!!! 3E MATERIAL


resuWeight=[]
resuTime=[]
resuMrho=[]
resuCO2=[]
resuTaper=[]
resuSpan=[]
resuChord=[]
resuWeightFix=[]
resuTimeFix=[]
resuMrhoFix=[]
resuCO2Fix=[]
resuTaperFix=[]
resuSpanFix=[]
resuChordFix=[]
resuWeightFix2=[]
resuTimeFix2=[]
resuMrhoFix2=[]
resuCO2Fix2=[]
resuTaperFix2=[]
resuSpanFix2=[]
resuChordFix2=[]

#    for mrhof2 in mrhoInit:
for case1 in range(0,len(cases),1):
    cr = CaseReader("aerostructMrhoi505sk"+str(cases[case1][0])+"sr"+str(cases[case1][1])+"sn"+str(cases[case1][2])+"tc"+str(cases[case1][3])+".db")

    driver_cases = cr.list_cases('driver')
    
    iterations=len(driver_cases)
    #$iterations=3
    
    i=iterations-1
    #for i in range(350,380):
    case = cr.get_case(driver_cases[i])
    design_vars = case.get_design_vars()
    objective= case.get_objectives()
    constraints= case.get_constraints()
    resuMrho.append(design_vars['mrho'][0])
    resuWeight.append(case.outputs['wing.structural_mass'][0])
    resuCO2.append(objective['emitted_co2'][0])
#    taper.append(case.inputs['wing.geometry.mesh.taper.taper'][0])
#    span.append(case.inputs['wing.geometry.mesh.stretch.span'][0])
#    chord.append(case.inputs['wing.geometry.mesh.scale_x.chord'][0])
#    chordTip.append(case.inputs['wing.geometry.mesh.scale_x.chord'][-1])
#    surface0.append(case.outputs['AS_point_0.coupled.wing.S_ref'][0])
##    surface1.append(case.outputs['AS_point_1.coupled.wing.S_ref'][0])
#    sparThicknessRoot.append(design_vars['wing.spar_thickness_cp'][-1])
#    sparThicknessTip.append(design_vars['wing.spar_thickness_cp'][0])
#    skinThicknessRoot.append(design_vars['wing.skin_thickness_cp'][-1])
#    skinThicknessTip.append(design_vars['wing.skin_thickness_cp'][0])
##    failure.append(constraints['AS_point_1.wing_perf.failure'][0])
##    power.append(constraints['AS_point_1.enough_power'][0])
##    lift.append(constraints['AS_point_1.L_equals_W'][0])
#    failure.append(constraints['AS_point_1.wing_perf.failure'][0])
#    power.append(constraints['AS_point_0.enough_power'][0])
#    lift.append(constraints['AS_point_0.L_equals_W'][0])
#    tOverC1.append(case.outputs['wing.geometry.t_over_c_cp'])
#    tOverC2.append(case.outputs['wing.t_over_c'][0])
#    buckling.append(constraints['AS_point_1.wing_perf.buckling'][0])
#        
#
#
    cr = CaseReader("aerostructMrhoi560sk"+str(cases[case1][0])+"sr"+str(cases[case1][1])+"sn"+str(cases[case1][2])+"tc"+str(cases[case1][3])+".db")
    driver_cases = cr.list_cases('driver')
    
    iterations=len(driver_cases)
    #$iterations=3
    
    i=iterations-1
    #for i in range(350,380):
    case = cr.get_case(driver_cases[i])
    design_vars = case.get_design_vars()
    objective= case.get_objectives()
    constraints= case.get_constraints()
    resuMrhoFix.append(design_vars['mrho'][0])
    resuWeightFix.append(case.outputs['wing.structural_mass'][0])
    resuCO2Fix.append(objective['emitted_co2'][0])
    
    cr = CaseReader("aerostructMrhoi529sk"+str(cases[case1][0])+"sr"+str(cases[case1][1])+"sn"+str(cases[case1][2])+"tc"+str(cases[case1][3])+".db")
    driver_cases = cr.list_cases('driver')
    
    iterations=len(driver_cases)
    #$iterations=3
    
    i=iterations-1
    #for i in range(350,380):
    case = cr.get_case(driver_cases[i])
    design_vars = case.get_design_vars()
    objective= case.get_objectives()
    constraints= case.get_constraints()
    resuMrhoFix2.append(design_vars['mrho'][0])
    resuWeightFix2.append(case.outputs['wing.structural_mass'][0])
    resuCO2Fix2.append(objective['emitted_co2'][0])