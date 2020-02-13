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
spanRange=np.arange(40,80,10) #4
tcRange=np.arange(0.11,0.18,0.03) #3
skinRange=np.arange(0.0012,0.0019,0.0003) #3
mrhoRange=[500,1250,2000] #3
divRange=np.arange(1,1.6,0.1)  #CHANGE MAT

#spanRange=[50] #4
#tcRange=[0.17000000000000004] #3
#skinRange=[0.0012] #3
#mrhoRange=[500,1250,2000] #3
#divRange=[1.1,1.3,1.5,2,5]  #CHANGE MAT




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
        
optimumsCO2=np.amin(resuCO2,axis=1)      
optimumsIndex=np.argmin(resuCO2,axis=1)
for i in range(len(divRange)):
    optimumsWeight=resuWeight[i][optimumsIndex[i]]
    optimumsMrho=resuWeight[i][optimumsIndex[i]]
    optimumsCases=resuWeight[i][optimumsIndex[i]]

print(optimumsCO2)
print(optimumsMrho)

