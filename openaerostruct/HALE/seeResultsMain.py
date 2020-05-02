# -*- coding: utf-8 -*-
"""
Created on Mon May 13 13:08:36 2019

@author: e.duriez
"""
from openmdao.api import Problem, ScipyOptimizeDriver, SqliteRecorder, CaseReader
from fonctionOptim import fctOptim
import matplotlib.pyplot as plt
import os
import numpy as np

#define x0
#spanRange=np.arange(55,70,5) #3
#tcRange=np.arange(0.11,0.18,0.03) #3
#tcRange=[tcRange[0],tcRange[2]] #2
#skinRange=np.arange(0.0010,0.0017,0.0001) #7
#mrhoRange=[500,1250,2000] #3
#sparRange=[0.0001] #1

spanRange=np.arange(25,70)
tcRange=np.arange(0.05,0.18)
skinRange=np.arange(0.002,0.0045)
sparRange=np.arange(0.0001,0.0004)

##spanRange=np.arange(25,70,20)
##tcRange=np.arange(0.05,0.18,0.04)
##skinRange=np.arange(0.002,0.0045,0.001)
##sparRange=np.arange(0.0001,0.0004,0.0001)
mrhoRange=[505]


caseArray=np.zeros((len(skinRange),len(mrhoRange),len(spanRange),len(tcRange),len(sparRange),5),dtype=object) #CHANGE MAT
for i in range(0,len(skinRange),1):
    for j in range(0,len(mrhoRange),1):
        for k in range(0,len(spanRange),1):
            for l in range(0,len(tcRange),1):
                for m in range(0,len(sparRange),1):   #CHANGE BAT
                    caseArray[i,j,k,l,m,]=[skinRange[i],mrhoRange[j],spanRange[k],tcRange[l],sparRange[m]]  #CHANGE MAT


cases=np.reshape(caseArray,(len(skinRange)*len(mrhoRange)*len(spanRange)*len(tcRange)*len(sparRange),5))  #CHANGE MAT
print(cases)


resuWeight=[]
resuTime=[]
resuMrho=[]
resuCO2=[]
resuCases=[]


for casenbr in range(0,len(cases),1):

#    cr = CaseReader("articleMats/aerostructMrhoi"+str(cases[casenbr][1])+"sk"+str(cases[casenbr][0])+"sr"+str(cases[casenbr][4])+"sn"+str(cases[casenbr][2])+"tc"+str(cases[casenbr][3])+".db")
    cr = CaseReader("aerostructMrhoi"+str(cases[casenbr][1])+"sk"+str(cases[casenbr][0])+"sr"+str(cases[casenbr][4])+"sn"+str(cases[casenbr][2])+"tc"+str(cases[casenbr][3])+".db")
    driver_cases = cr.list_cases('driver')
    iterations=len(driver_cases)
    
    case = cr.get_case(driver_cases[-1])
    design_vars = case.get_design_vars()
    objective= case.get_objectives()
    constraints= case.get_constraints()
    rhorho=design_vars['mrho'][0] 
    weight=case.outputs['wing.structural_mass'][0] 
    co2=objective['emitted_co2'][0] 
    
    failure=constraints['AS_point_1.wing_perf.failure'][0]
    power=constraints['AS_point_0.enough_power'][0]
    lift=constraints['AS_point_0.L_equals_W'][0]
    buckling=constraints['AS_point_1.wing_perf.buckling'][0]
    acceptableThickness=constraints['acceptableThickness']

    maxconstraint=max(abs(lift),failure,power,max(acceptableThickness),buckling)    

    if maxconstraint < 1e-3:
        resuWeight.append(weight)
        resuMrho.append(rhorho)
        resuCO2.append(co2)
        resuCases.append(cases[casenbr])
        
optimumCO2=min(resuCO2)  
optimumIndex=np.argmin(resuCO2)
optimumWeight=resuWeight[optimumIndex]
optimumMrho=resuMrho[optimumIndex]
optimumCase=resuCases[optimumIndex]

print(optimumCO2)
print(optimumCase)


