# -*- coding: utf-8 -*-
"""
Created on Mon May 13 13:08:36 2019

@author: e.duriez and Victor M. Guadano
"""


from openmdao.api import Problem, ScipyOptimizeDriver, SqliteRecorder, CaseReader
import matplotlib.pyplot as plt
import os
import numpy as np


mrhoRange = np.arange(500,601,100)
spanRange=np.arange(25,76,25)
tcRange=np.arange(0.05,0.18,0.06)
skinRange=np.arange(0.002,0.0041,0.001)
sparRange=np.arange(0.001,0.0031,0.001)


caseArray=np.zeros((len(skinRange),len(sparRange),len(spanRange),len(tcRange),len(mrhoRange),5),dtype=object)
for i in range(0,len(skinRange),1):
    for j in range(0,len(sparRange),1):
        for k in range(0,len(spanRange),1):
            for l in range(0,len(tcRange),1):
                for m in range (0,len(mrhoRange),1):
                    caseArray[i,j,k,l,m]=[skinRange[i],sparRange[j],spanRange[k],tcRange[l],mrhoRange[m]]

cases=np.reshape(caseArray,(len(skinRange)*len(sparRange)*len(spanRange)*len(tcRange)*len(mrhoRange),5))
print(cases)

# Results for the cases that converge
resuCases=[]  # Starting point
resuCO2=[]  # Final CO2 emissions [10^4 kg]
resuMrho=[]  # Final material density [g/cm3]
resuWeight=[]  # Final weight [N]

for casenbr in range(0,len(cases),1):

    cr = CaseReader("aerostructMrhoi"+str(cases[casenbr][4])+"sk"+str(cases[casenbr][0])+"sr"+str(cases[casenbr][1])+"sn"+str(cases[casenbr][2])+"tc"+str(cases[casenbr][3])+".db")
    driver_cases = cr.list_cases('driver')
    iterations=len(driver_cases)
    
    case = cr.get_case(driver_cases[-1])
    design_vars = case.get_design_vars()
    objective= case.get_objectives()
    constraints= case.get_constraints()
    
    co2=objective['emitted_co2'][0]
    mrho=design_vars['mrho']
    weight=case.outputs['wing.structural_mass'][0] 
    
    failure=constraints['AS_point_1.wing_perf.failure'][0]
    power=constraints['AS_point_0.enough_power'][0]
    lift=constraints['AS_point_0.L_equals_W'][0]
    buckling=constraints['AS_point_1.wing_perf.buckling'][0]
    acceptableThickness=constraints['acceptableThickness']
    failureOG=constraints['AS_point_2.wing_perf.failure'][0]
    bucklingOG=constraints['AS_point_2.wing_perf.buckling'][0]

    maxconstraint=max(abs(lift),failure,power,max(acceptableThickness),buckling,failureOG,bucklingOG)    
    
    if maxconstraint < 1e-3:  # Check constraints (necessary convergence condition)
        resuCases.append(cases[casenbr])
        resuCO2.append(co2)
        resuMrho.append(mrho)
        resuWeight.append(weight)

# Optimum case (minimum CO2)
optimumCO2=min(resuCO2)  
optimumIndex=np.argmin(resuCO2)
optimumCase=resuCases[optimumIndex]
optimumMrho=resuMrho[optimumIndex]
optimumWeight=resuWeight[optimumIndex]

print(optimumCO2)
print(optimumCase)