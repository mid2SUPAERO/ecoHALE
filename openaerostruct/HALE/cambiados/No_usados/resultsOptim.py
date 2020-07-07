# -*- coding: utf-8 -*-
"""
Created on Mon May 13 13:08:36 2019

@author: e.duriez
"""
from openmdao.api import Problem, ScipyOptimizeDriver, SqliteRecorder, CaseReader
import matplotlib.pyplot as plt
from fonctionOptim import fctOptim
import os
import numpy as np

#define x0
spanRange=np.arange(40,70,5)
tcRange=np.arange(0.07,0.20,0.04)
skinRange=np.arange(0.001,0.005,0.001)
sparRange=np.arange(0.001,0.005,0.001)


#50
#0.08
#0.001
#0.001
#spanRange=[61]
#tcRange=[0.19]
#skinRange=[0.003]
#sparRange=[0.001]

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
for case1 in range(0,165,1):
    #mrhof=mrhof2/50
    #limbasserho=60
    #limhauterho=200
    limbasserho=55
    limhauterho=8220
#    if case[3]==0.11:
#        case[3]=0.11000000000000001
#    if case[3]==0.15:
#        case[3]=0.15000000000000002
#    if case[3]==0.19:
#        case[3]=0.19000000000000003
    cr = CaseReader("interstr=chord/aerostructMrhoi504sk"+str(cases[case1][0])+"sr"+str(cases[case1][1])+"sn"+str(cases[case1][2])+"tc"+str(cases[case1][3])+".db")

    driver_cases = cr.list_cases('driver')

    iterations=len(driver_cases)
    #$iterations=3
    case = cr.get_case(driver_cases[iterations-1])
    design_vars = case.get_design_vars()
    objective= case.get_objectives()
    constraints= case.get_constraints()
    resuMrho.append(design_vars['mrho'][0])
    resuWeight.append(case.outputs['wing.structural_mass'][0])
    resuCO2.append(objective['emitted_co2'][0])
    resuTaper.append(case.inputs['wing.geometry.mesh.taper.taper'][0])
    resuSpan.append(case.inputs['wing.geometry.mesh.stretch.span'][0])
    resuChord.append(case.inputs['wing.geometry.mesh.scale_x.chord'][0])
#    resuSurf.append(resuSpan[-1]*resuChord[-1]*(1-resuTaper[-1])/2)
#    resuPower.append(constraints['AS_point_0.enough_power'][0])
#    resuFailure.append(constraints['AS_point_1.wing_perf.failure'][0])
#    resuBuckling.append(constraints['AS_point_1.wing_perf.buckling'][0])


    cr = CaseReader("interstr=chord/aerostructMrhoi560sk"+str(cases[case1][0])+"sr"+str(cases[case1][1])+"sn"+str(cases[case1][2])+"tc"+str(cases[case1][3])+".db")

    driver_cases = cr.list_cases('driver')

    iterations=len(driver_cases)
    #$iterations=3
    case = cr.get_case(driver_cases[iterations-1])
    design_vars = case.get_design_vars()
    objective= case.get_objectives()
    constraints= case.get_constraints()
    resuMrhoFix.append(design_vars['mrho'][0])
    resuWeightFix.append(case.outputs['wing.structural_mass'][0])
    resuCO2Fix.append(objective['emitted_co2'][0])
    resuTaperFix.append(case.inputs['wing.geometry.mesh.taper.taper'][0])
    resuSpanFix.append(case.inputs['wing.geometry.mesh.stretch.span'][0])
    resuChordFix.append(case.inputs['wing.geometry.mesh.scale_x.chord'][0])
#    resuSurf.append(resuSpan[-1]*resuChord[-1]*(1-resuTaper[-1])/2)
#    resuPower.append(constraints['AS_point_0.enough_power'][0])
#    resuFailure.append(constraints['AS_point_1.wing_perf.failure'][0])
#    resuBuckling.append(constraints['AS_point_1.wing_perf.buckling'][0])
        
    cr = CaseReader("interstr=chord/aerostructMrhoi529sk"+str(cases[case1][0])+"sr"+str(cases[case1][1])+"sn"+str(cases[case1][2])+"tc"+str(cases[case1][3])+".db")

    driver_cases = cr.list_cases('driver')

    iterations=len(driver_cases)
    #$iterations=3
    case = cr.get_case(driver_cases[iterations-1])
    design_vars = case.get_design_vars()
    objective= case.get_objectives()
    constraints= case.get_constraints()
    resuMrhoFix2.append(design_vars['mrho'][0])
    resuWeightFix2.append(case.outputs['wing.structural_mass'][0])
    resuCO2Fix2.append(objective['emitted_co2'][0])
    resuTaperFix2.append(case.inputs['wing.geometry.mesh.taper.taper'][0])
    resuSpanFix2.append(case.inputs['wing.geometry.mesh.stretch.span'][0])
    resuChordFix2.append(case.inputs['wing.geometry.mesh.scale_x.chord'][0])
#    resuSurf.append(resuSpan[-1]*resuChord[-1]*(1-resuTaper[-1])/2)
#    resuPower.append(constraints['AS_point_0.enough_power'][0])
#    resuFailure.append(constraints['AS_point_1.wing_perf.failure'][0])
#    resuBuckling.append(constraints['AS_point_1.wing_perf.buckling'][0])

    

        


print("rho=504")
weightP=[]
timeP=[]
mrhofP=[]
mrhoiP=[]
co2P=[]
caseP=[]
casesP=[]

for case in range(0,len(cases),1):
    if resuTime[case]!=0:
        weightP.append(resuWeight[case])
        timeP.append(resuTime[case])
        mrhofP.append(resuMrho[case])
        co2P.append(resuCO2[case])
        caseP.append(case)
        casesP.append(cases[case])

#    plt.plot(mrhoInit,resuTimeP[i])
plt.plot(caseP,timeP,'ob')
plt.xlabel('case')
plt.ylabel('time')
#plt.ylim((3.1,3.3))

plt.show()

#    plt.plot(mrhoInit,resuWeightP[i])
plt.plot(caseP,weightP,'ob')
plt.xlabel('case')
plt.ylabel('weight')
plt.ylim((0,60))

plt.show()

#    plt.plot(mrhoInit,resuMrhoP[i])
plt.plot(caseP,mrhofP,'ob')
plt.xlabel('case')
plt.ylabel('mrhof')
#    plt.ylim((1500,5000))

plt.show()

#    plt.plot(mrhoInit,resuCO2P[i])
plt.plot(caseP,co2P,'ob')
plt.xlabel('case')
plt.ylabel('co2')
plt.ylim((0,20000))

plt.show()
    
print("rho=574")
weightPFix=[]
timePFix=[]
mrhofPFix=[]
mrhoiPFix=[]
co2PFix=[]
casePFix=[]
casesPFix=[]

for case in range(0,len(cases),1):
    if resuTimeFix[case]!=0:
        weightPFix.append(resuWeightFix[case])
        timePFix.append(resuTimeFix[case])
        mrhofPFix.append(resuMrhoFix[case])
        co2PFix.append(resuCO2Fix[case])
        casePFix.append(case)
        casesPFix.append(cases[case])


plt.plot(casePFix,weightPFix,'ob')
plt.xlabel('case')
plt.ylabel('weight')
plt.ylim((0,60))

plt.show()

#    plt.plot(mrhoInit,resuTimePFix[i])
plt.plot(casePFix,timePFix,'ob')
plt.xlabel('case')
plt.ylabel('time')
#plt.ylim((3.1,3.3))

plt.show()

#    plt.plot(mrhoInit,resuMrhoPFix[i])
plt.plot(casePFix,mrhofPFix)
plt.xlabel('case')
plt.ylabel('mrhof')
#plt.ylim((0,6000))

plt.show()

#    plt.plot(mrhoInit,resuCO2PFix[i])
plt.plot(casePFix,co2PFix,'ob')
plt.xlabel('case')
plt.ylabel('co2')
plt.ylim((0,20000))

plt.show()

print("rho=529")
weightPFix2=[]
timePFix2=[]
mrhofPFix2=[]
mrhoiPFix2=[]
co2PFix2=[]
casePFix2=[]
casesPFix2=[]

for case in range(0,len(cases),1):
    if resuTimeFix2[case]!=0:
        weightPFix2.append(resuWeightFix2[case])
        timePFix2.append(resuTimeFix2[case])
        mrhofPFix2.append(resuMrhoFix2[case])
        co2PFix2.append(resuCO2Fix2[case])
        casePFix2.append(case)
        casesPFix2.append(cases[case])


plt.plot(casePFix2,weightPFix2,'ob')
plt.xlabel('case')
plt.ylabel('weight')
plt.ylim((0,60))

plt.show()

#    plt.plot(mrhoInit,resuTimePFix[i])
plt.plot(casePFix2,timePFix2,'ob')
plt.xlabel('case')
plt.ylabel('time')
#plt.ylim((3.1,3.3))

plt.show()

#    plt.plot(mrhoInit,resuMrhoPFix[i])
plt.plot(casePFix2,mrhofPFix2)
plt.xlabel('case')
plt.ylabel('mrhof')
#plt.ylim((0,6000))

plt.show()

#    plt.plot(mrhoInit,resuCO2PFix[i])
plt.plot(casePFix2,co2PFix2,'ob')
plt.xlabel('case')
plt.ylabel('co2')
plt.ylim((0,20000))

plt.show()