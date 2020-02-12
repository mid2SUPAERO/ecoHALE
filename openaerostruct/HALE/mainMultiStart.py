# -*- coding: utf-8 -*-
"""
Created on Mon May 13 13:08:36 2019

@author: e.duriez
"""
from fonctionOptim import fctOptim
import matplotlib.pyplot as plt
import os
import numpy as np

#define x0
spanRange=np.arange(40,80,10) #4
tcRange=np.arange(0.11,0.18,0.03) #3
skinRange=np.arange(0.0012,0.0019,0.0003) #3
sparRange=np.arange(0.0001,0.0002,0.0001) #1
divRange=[1.1,1.3,1.5,2,5]  #CHANGE MAT


#50
#0.08
#0.001
#0.001
#spanRange=[25]
#tcRange=[0.05]
#skinRange=[0.003]
#sparRange=[0.00030000000000000003]

#caseArray=np.zeros((len(skinRange),len(sparRange),len(spanRange),len(tcRange),4),dtype=object)
caseArray=np.zeros((len(skinRange),len(sparRange),len(spanRange),len(tcRange),len(divRange),5),dtype=object) #CHANGE MAT
for i in range(0,len(skinRange),1):
    for j in range(0,len(sparRange),1):
        for k in range(0,len(spanRange),1):
            for l in range(0,len(tcRange),1):
#                caseArray[i,j,k,l,]=[skinRange[i],sparRange[j],spanRange[k],tcRange[l]]
                for m in range(0,len(divRange),1):   #CHANGE BAT
                    caseArray[i,j,k,l,m,]=[skinRange[i],sparRange[j],spanRange[k],tcRange[l],divRange[m]]  #CHANGE MAT


#cases=np.reshape(caseArray,(len(skinRange)*len(sparRange)*len(spanRange)*len(tcRange),4))
cases=np.reshape(caseArray,(len(skinRange)*len(sparRange)*len(spanRange)*len(tcRange)*len(divRange),5))  #CHANGE MAT
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
for case in range(0,len(cases),1):
    #mrhof=mrhof2/50
    #limbasserho=60
    #limhauterho=200
    limbasserho=55
    limhauterho=8220

    try:
#        resu=fctOptim(500,cases[case][0],cases[case][1],cases[case][2],cases[case][3])  
        resu=fctOptim(500,cases[case][0],cases[case][1],cases[case][2],cases[case][3],cases[case][4])   #CHANGE MAT
    ##            resu=fctOptim(mrhof+1,hour,limbasserho,limhauterho) #ED2 
    ##            resu=fctOptim(mrhof+1,1,limbasserho,limhauterho,epmin)  
        weight=resu[0]
        time=resu[1]
        rhorho=resu[2]
        co2=resu[3]
    except:
        weight=0
        time=0
        rhorho=0
        co2=0

        weight=0
        time=0
        rhorho=0
        co2=0


    try:
#        resu=fctOptim(2000,cases[case][0],cases[case][1],cases[case][2],cases[case][3])  
        resu=fctOptim(2000,cases[case][0],cases[case][1],cases[case][2],cases[case][3],cases[case][4])   #CHANGE MAT
    #            resu=fctOptim(mrhof,1,mrhof,mrhof,epmin)  
        weightFix=resu[0]
        timeFix=resu[1]
        rhorhoFix=resu[2]
        co2Fix=resu[3]
    except:
        weightFix=0
        timeFix=0
        rhorhoFix=0
        co2Fix=0
##
###        weightFix=0
###        timeFix=0
###        rhorhoFix=0
###        co2Fix=0
##        
    try:
#        resu=fctOptim(1250,cases[case][0],cases[case][1],cases[case][2],cases[case][3])  
        resu=fctOptim(1250,cases[case][0],cases[case][1],cases[case][2],cases[case][3],cases[case][4])   #CHANGE MAT
    #            resu=fctOptim(mrhof,1,mrhof,mrhof,epmin)  
        weightFix2=resu[0]
        timeFix2=resu[1]
        rhorhoFix2=resu[2]
        co2Fix2=resu[3]
    except:
        weightFix2=0
        timeFix2=0
        rhorhoFix2=0
        co2Fix2=0
##
###        weightFix=0
###        timeFix=0
###        rhorhoFix=0
###        co2Fix=0
#
#    
    resuWeight.append(weight)
    resuTime.append(time)
    resuMrho.append(rhorho)
    resuCO2.append(co2)
    resuWeightFix.append(weightFix)
    resuTimeFix.append(timeFix)
    resuMrhoFix.append(rhorhoFix)
    resuCO2Fix.append(co2Fix)
    resuWeightFix2.append(weightFix2)
    resuTimeFix2.append(timeFix2)
    resuMrhoFix2.append(rhorhoFix2)
    resuCO2Fix2.append(co2Fix2)
#        
#
#
##print("rho=504")
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
#    
#print("rho=574")
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
#
#
#plt.plot(casePFix,weightPFix,'ob')
#plt.xlabel('case')
#plt.ylabel('weight')
#plt.ylim((0,60))
#
#plt.show()
#
##    plt.plot(mrhoInit,resuTimePFix[i])
#plt.plot(casePFix,timePFix,'ob')
#plt.xlabel('case')
#plt.ylabel('time')
##plt.ylim((3.1,3.3))
#
#plt.show()
#
##    plt.plot(mrhoInit,resuMrhoPFix[i])
#plt.plot(casePFix,mrhofPFix)
#plt.xlabel('case')
#plt.ylabel('mrhof')
##plt.ylim((0,6000))
#
#plt.show()
#
##    plt.plot(mrhoInit,resuCO2PFix[i])
#plt.plot(casePFix,co2PFix,'ob')
#plt.xlabel('case')
#plt.ylabel('co2')
#plt.ylim((0,20000))
#
#plt.show()
#
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
#
#
#plt.plot(casePFix2,weightPFix2,'ob')
#plt.xlabel('case')
#plt.ylabel('weight')
#plt.ylim((0,60))
#
#plt.show()
#
##    plt.plot(mrhoInit,resuTimePFix[i])
#plt.plot(casePFix2,timePFix2,'ob')
#plt.xlabel('case')
#plt.ylabel('time')
##plt.ylim((3.1,3.3))
#
#plt.show()
#
##    plt.plot(mrhoInit,resuMrhoPFix[i])
#plt.plot(casePFix2,mrhofPFix2)
#plt.xlabel('case')
#plt.ylabel('mrhof')
##plt.ylim((0,6000))
#
#plt.show()
#
##    plt.plot(mrhoInit,resuCO2PFix[i])
#plt.plot(casePFix2,co2PFix2,'ob')
#plt.xlabel('case')
#plt.ylabel('co2')
#plt.ylim((0,20000))
#
#plt.show()