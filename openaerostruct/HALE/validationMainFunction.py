# -*- coding: utf-8 -*-
"""
Created on Mon May 13 13:08:36 2019

@author: e.duriez
"""
from validationOptimFunction import fctOptim
import matplotlib.pyplot as plt
import os
import numpy as np

#define x0
spanRange=np.arange(25,70,20)
tcRange=np.arange(0.05,0.18,0.04)
skinRange=np.arange(0.002,0.0045,0.001)
sparRange=np.arange(0.0001,0.0004,0.0001)

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


resuWingWeight=[]
resuTime=[]
resuMrho=[]
resuCO2=[]
resuCases=[]

for case in range(0,len(cases),1):

    try:
        resu=fctOptim(505,cases[case][0],cases[case][1],cases[case][2],cases[case][3])  
    ##            resu=fctOptim(mrhof+1,hour,limbasserho,limhauterho) #ED2 
    ##            resu=fctOptim(mrhof+1,1,limbasserho,limhauterho,epmin)  
        wingWeight=resu[0]
        time=resu[1]
        mrho=resu[2]
        co2=resu[3]
        maxconstraint=resu[4]
    except:
        wingWeight=0
        time=0
        mrho=0
        co2=0
        maxconstraint=1
    
    if maxconstraint<1e-3:
        
        resuWingWeight.append(wingWeight)
        resuTime.append(time)
        resuMrho.append(mrho)
        resuCO2.append(co2)
        resuCases.append(cases[case])

optimumCO2=np.amin(resuCO2)      
optimumIndex=np.argmin(resuCO2)
optimumWingWeight=resuWingWeight[optimumIndex]
optimumMrho=resuMrho[optimumIndex]
optimumCases=resuCases[optimumIndex]

print(optimumCO2)
print(optimumMrho)
print(optimumWingWeight)
