# -*- coding: utf-8 -*-
"""
Created on Mon May 13 13:08:36 2019

@author: e.duriez and Victor M. Guadano
"""


from fonctionOptim_CO2minimization import fctOptim
import matplotlib.pyplot as plt
import os
import numpy as np


mrhoRange = np.array([600])
spanRange = np.array([100])
tcRange = np.array([0.13])
skinRange = np.array([0.002])
sparRange = np.array([0.002])


caseArray=np.zeros((len(skinRange),len(sparRange),len(spanRange),len(tcRange),len(mrhoRange),5),dtype=object)
for i in range(0,len(skinRange),1):
    for j in range(0,len(sparRange),1):
        for k in range(0,len(spanRange),1):
            for l in range(0,len(tcRange),1):
                for m in range (0,len(mrhoRange),1):
                    caseArray[i,j,k,l,m]=[skinRange[i],sparRange[j],spanRange[k],tcRange[l],mrhoRange[m]]

cases=np.reshape(caseArray,(len(skinRange)*len(sparRange)*len(spanRange)*len(tcRange)*len(mrhoRange),5))
print(cases)

resuWeight=[]
resuTime=[]
resuMrho=[]
resuCO2=[]

for case in range(0,len(cases),1):
    
    """
    resu=fctOptim(cases[case][4],cases[case][0],cases[case][1],cases[case][2],cases[case][3])   
    #            resu=fctOptim(mrhof+1,hour,limbasserho,limhauterho) #ED2 
    #            resu=fctOptim(mrhof+1,1,limbasserho,limhauterho,epmin)  
    weight=resu[0]
    time=resu[1]
    rhorho=resu[2]
    co2=resu[3]
    """
    
    try:
        resu=fctOptim(cases[case][4],cases[case][0],cases[case][1],cases[case][2],cases[case][3])  
    #            resu=fctOptim(mrhof+1,hour,limbasserho,limhauterho) #ED2 
    #            resu=fctOptim(mrhof+1,1,limbasserho,limhauterho,epmin)  
        weight=resu[0]
        time=resu[1]
        rhorho=resu[2]
        co2=resu[3]
    except:
        weight=0
        time=0
        rhorho=0
        co2=0
    #"""

    resuWeight.append(weight)
    resuTime.append(time)
    resuMrho.append(rhorho)
    resuCO2.append(co2)

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

plt.plot(caseP,timeP,'ob')
plt.xlabel('case')
plt.ylabel('time')
plt.show()

plt.plot(caseP,weightP,'ob')
plt.xlabel('case')
plt.ylabel('weight')
plt.show()

plt.plot(caseP,mrhofP,'ob')
plt.xlabel('case')
plt.ylabel('mrhof')
plt.show()

plt.plot(caseP,co2P,'ob')
plt.xlabel('case')
plt.ylabel('co2')
plt.show()