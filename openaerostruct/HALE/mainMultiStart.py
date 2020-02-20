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

resuWeight=[[] for i in range(len(divRange))]
resuTime=[[] for i in range(len(divRange))]
resuMrho=[[] for i in range(len(divRange))]
resuCO2=[[] for i in range(len(divRange))]
resuCases=[[] for i in range(len(divRange))]


for case in range(0,len(cases),1):

    try:
        resu=optimFct(cases[case][1],cases[case][0],0.0001,cases[case][2],cases[case][3],cases[case][4])   #CHANGE MAT
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
        resuWeight[case % len(divRange)].append(weight)
        resuTime[case % len(divRange)].append(time)
        resuMrho[case % len(divRange)].append(rhorho)
        resuCO2[case % len(divRange)].append(co2)
        resuCases[case % len(divRange)].append(cases[case])

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