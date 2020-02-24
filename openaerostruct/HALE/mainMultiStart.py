# -*- coding: utf-8 -*-
"""
Created on Mon May 13 13:08:36 2019

@author: e.duriez
"""
from optimFct import optimFct
import matplotlib.pyplot as plt
import os
import numpy as np

#define x0
spanRange=np.arange(55,70,5) #4
tcRange=np.arange(0.11,0.18,0.03) #3
tcRange=[tcRange[0],tcRange[2]]
skinRange=np.arange(0.0010,0.0017,0.0001) #7
mrhoRange=[1250,2000] #3


caseArray=np.zeros((len(skinRange),len(mrhoRange),len(spanRange),len(tcRange),4),dtype=object) 
for i in range(0,len(skinRange),1):
    for j in range(0,len(mrhoRange),1):
        for k in range(0,len(spanRange),1):
            for l in range(0,len(tcRange),1):
                caseArray[i,j,k,l,]=[skinRange[i],mrhoRange[j],spanRange[k],tcRange[l]]

cases=np.reshape(caseArray,(len(skinRange)*len(mrhoRange)*len(spanRange)*len(tcRange),4))  

resuWeight=[]
resuTime=[]
resuMrho=[]
resuCO2=[]
resuCases=[]


for case in range(0,len(cases),1):

    try:
        resu=optimFct(cases[case][1],cases[case][0],0.0001,cases[case][2],cases[case][3])   
        weight=resu[0]
        time=resu[1]
        rhorho=resu[2]
        co2=resu[3]
        maxconstraint=resu[4]
    except:
        weight=0
        time=0
        rhorho=0
        co2=0
        maxconstraint=1

    if maxconstraint < 1e-3:
        resuWeight.append(weight)
        resuTime.append(time)
        resuMrho.append(rhorho)
        resuCO2.append(co2)
        resuCases.append(cases[case])

optimumCO2=min(resuCO2)
optimumIndex=np.argmin(resuCO2)
optimumWeight=resuWeight[optimumIndex]
optimumMrho=resuMrho[optimumIndex]
optimumCases=resuCases[optimumIndex]

print(optimumMrho)