# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 11:13:32 2019

@author: e.duriez
"""
import numpy as np

spanRange=np.arange(20,50,10)
tcRange=np.arange(0.07,0.2,0.04)
skinRange=np.arange(0.005,0.026,0.01)
sparRange=np.arange(0.005,0.026,0.01)


AAA=np.zeros((len(skinRange),len(sparRange),len(spanRange),len(tcRange),4),dtype=object)
for i in range(0,len(skinRange),1):
    for j in range(0,len(sparRange),1):
        for k in range(0,len(spanRange),1):
            for l in range(0,len(tcRange),1):
                AAA[i,j,k,l,]=[skinRange[i],sparRange[j],spanRange[k],tcRange[l]]


BBB=np.reshape(AAA,(len(skinRange)*len(sparRange)*len(spanRange)*len(tcRange),4))
print(BBB)

print(BBB[2])

print(np.append(1521,BBB[2]))


