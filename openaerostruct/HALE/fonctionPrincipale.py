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
spanRange=np.arange(20,50,10)
tcRange=np.arange(0.07,0.2,0.04)
skinRange=np.arange(0.005,0.026,0.01)
sparRange=np.arange(0.005,0.026,0.01)


caseArray=np.zeros((len(skinRange),len(sparRange),len(spanRange),len(tcRange),4),dtype=object)
for i in range(0,len(skinRange),1):
    for j in range(0,len(sparRange),1):
        for k in range(0,len(spanRange),1):
            for l in range(0,len(tcRange),1):
                caseArray[i,j,k,l,]=[skinRange[i],sparRange[j],spanRange[k],tcRange[l]]


cases=np.reshape(caseArray,(len(skinRange)*len(sparRange)*len(spanRange)*len(tcRange),4))
print(cases)





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

#    for mrhof2 in mrhoInit:
for case in range(0,len(cases),1):
    #mrhof=mrhof2/50
    #limbasserho=60
    #limhauterho=200
    limbasserho=55
    limhauterho=8220

###Option1 : to do the optimization with the material varying
    try:
        resu=fctOptim(cases[case])  
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

###Option2 : do only the optimization with the material fixed, to see what the function being optimized looks like
#        weight=0
#        time=0
#        rhorho=0
#        co2=0


##Option1 : to see what the function being optimized looks like
    try:
        resu=fctOptim(mrhof,puissanceMM,mrhof,mrhof)  
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

###Option2 : do only the optimization with the material varying, to go faster
#        weightFix=0
#        timeFix=0
#        rhorhoFix=0
#        co2Fix=0

    
    resuWeight.append(weight)
    resuTime.append(time)
    resuMrho.append(rhorho)
    resuCO2.append(co2)
    resuWeightFix.append(weightFix)
    resuTimeFix.append(timeFix)
    resuMrhoFix.append(rhorhoFix)
    resuCO2Fix.append(co2Fix)
        


i=-1 
for puissanceMM in rangeP:
#for hour in rangeH: #ED2
#for epmin in rangeEp:
    i+=1
#Æfor puissanceMM in range(1,6):
    weightP=[]
    timeP=[]
    mrhofP=[]
    mrhoiP=[]
    co2P=[]
    for j in range(len(mrhoInit)):
        if resuTimeP[i][j]!=0:
            weightP.append(resuWeightP[i][j])
            timeP.append(resuTimeP[i][j])
            mrhofP.append(resuMrhoP[i][j])
            co2P.append(resuCO2P[i][j])
            mrhoiP.append(mrhoInit[j])
    
    print(puissanceMM)
#    print(hour) #ED2
    print("var")

#    plt.plot(mrhoInit,resuTimeP[i])
    plt.plot(mrhoiP,timeP,'ob')
    plt.xlabel('mrho')
    plt.ylabel('time')
    #plt.ylim((3.1,3.3))
    
    plt.show()

#    plt.plot(mrhoInit,resuWeightP[i])
    plt.plot(mrhoiP,weightP,'ob')
    plt.xlabel('mrho')
    plt.ylabel('weight')
    plt.ylim((0,20))
    
    plt.show()
    
#    plt.plot(mrhoInit,resuMrhoP[i])
    plt.plot(mrhoiP,mrhofP,'ob')
    plt.xlabel('mrhoi')
    plt.ylabel('mrhof')
#    plt.ylim((1500,5000))
    
    plt.show()

#    plt.plot(mrhoInit,resuCO2P[i])
    plt.plot(mrhoiP,co2P,'ob')
    plt.xlabel('mrhoi')
    plt.ylabel('co2')
    plt.ylim((0,20000))
    
    plt.show()
    
i=-1    
for puissanceMM in rangeP:
#for hour in rangeH: #ED2
#for epmin in rangeEp:
    i+=1
#Æfor puissanceMM in range(1,6):
    weightPFix=[]
    timePFix=[]
    mrhofPFix=[]
    mrhoiPFix=[]
    co2PFix=[]
    for j in range(len(mrhoInit)):
        if resuTimePFix[i][j]!=0:
            weightPFix.append(resuWeightPFix[i][j])
            timePFix.append(resuTimePFix[i][j])
            mrhofPFix.append(resuMrhoPFix[i][j])
            co2PFix.append(resuCO2PFix[i][j])
            mrhoiPFix.append(mrhoInit[j])
    
    print(puissanceMM)
#    print(hour) #ED2
    print("fix")
#    plt.plot(mrhoInit,resuWeightPFix[i])
    plt.plot(mrhoiPFix,weightPFix,'ob')
    plt.xlabel('mrho')
    plt.ylabel('weight')
    plt.ylim((0,60))
    
    plt.show()
    
#    plt.plot(mrhoInit,resuTimePFix[i])
    plt.plot(mrhoiPFix,timePFix,'ob')
    plt.xlabel('mrho')
    plt.ylabel('time')
    #plt.ylim((3.1,3.3))
    
    plt.show()
    
#    plt.plot(mrhoInit,resuMrhoPFix[i])
    plt.plot(mrhoiPFix,mrhofPFix)
    plt.xlabel('mrhoi')
    plt.ylabel('mrhof')
    #plt.ylim((0,6000))
    
    plt.show()
    
#    plt.plot(mrhoInit,resuCO2PFix[i])
    plt.plot(mrhoiPFix,co2PFix,'ob')
    plt.xlabel('mrhoi')
    plt.ylabel('co2')
    plt.ylim((0,20000))
    
    plt.show()