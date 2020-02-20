# -*- coding: utf-8 -*-
"""
Created on Mon May 13 13:08:36 2019

@author: e.duriez
"""
from openmdao.api import Problem, ScipyOptimizeDriver, SqliteRecorder, CaseReader
from optimFct import optimFct
import matplotlib.pyplot as plt
import os
import numpy as np

#define x0
spanRange=np.arange(55,70,5) #4
tcRange=np.arange(0.11,0.18,0.03) #3
tcRange=[tcRange[0],tcRange[2]]
skinRange=np.arange(0.0010,0.0017,0.0001) #7
mrhoRange=[500,1250,2000] #3
divRange=np.arange(1,1.6,0.1)  #CHANGE MAT




#caseArray=np.zeros((len(skinRange),len(sparRange),len(spanRange),len(tcRange),4),dtype=object)
caseArray=np.zeros((len(skinRange),len(mrhoRange),len(spanRange),len(tcRange),len(divRange),5),dtype=object) #CHANGE MAT
for i in range(0,len(skinRange),1):
    for j in range(0,len(mrhoRange),1):
        for k in range(0,len(spanRange),1):
            for l in range(0,len(tcRange),1):
#                caseArray[i,j,k,l,]=[skinRange[i],sparRange[j],spanRange[k],tcRange[l]]
                for m in range(0,len(divRange),1):   #CHANGE BAT
                    caseArray[i,j,k,l,m,]=[skinRange[i],mrhoRange[j],spanRange[k],tcRange[l],divRange[m]]  #CHANGE MAT


#cases=np.reshape(caseArray,(len(skinRange)*len(sparRange)*len(spanRange)*len(tcRange),4))
cases=np.reshape(caseArray,(len(skinRange)*len(mrhoRange)*len(spanRange)*len(tcRange)*len(divRange),5))  #CHANGE MAT
print(cases)


resuWeight=[[] for i in range(len(divRange))]
resuTime=[[] for i in range(len(divRange))]
resuMrho=[[] for i in range(len(divRange))]
resuCO2=[[] for i in range(len(divRange))]
resuCases=[[] for i in range(len(divRange))]


for casenbr in range(0,len(cases),1):

    cr = CaseReader("aerostructMrhoi"+str(cases[casenbr][1])+"sk"+str(cases[casenbr][0])+"sr"+str(0.0001)+"sn"+str(cases[casenbr][2])+"tc"+str(cases[casenbr][3])+"ed"+str(cases[casenbr][4])+".db")
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
        resuWeight[casenbr % len(divRange)].append(weight)
        resuMrho[casenbr % len(divRange)].append(rhorho)
        resuCO2[casenbr % len(divRange)].append(co2)
        resuCases[casenbr % len(divRange)].append(cases[casenbr])
        
optimumsCO2=[min(resuCO2[i]) for i in np.arange(0,len(divRange))]   
optimumsIndex=[np.argmin(resuCO2[i]) for i in np.arange(0,len(divRange))]
optimumsWeight=[]
optimumsMrho=[]
optimumsCases=[]
for i in range(len(divRange)):
    optimumsWeight.append(resuWeight[i][optimumsIndex[i]])
    optimumsMrho.append(resuMrho[i][optimumsIndex[i]])
    optimumsCases.append(resuCases[i][optimumsIndex[i]])

print(optimumsMrho)

