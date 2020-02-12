# -*- coding: utf-8 -*-
"""
Created on Tue May  7 15:20:22 2019

@author: e.duriez
"""

import numpy as np
import matplotlib.pyplot as plt
from fctMultiMatos import youngMM, material, shearMM, yieldMM, co2MM


           
sandw4=material(504.5,42.5e9,16.3e9,586e6/1.5,44.9,"sandw4")
#    sandw5=material(574.5,42.5e9,16.3e9,586e6/1.5,39.3,"sandw5")
sandw5=material(560.5,42.5e9,16.3e9,586e6/1.5,40.3,"sandw5")
sandw6=material(529,42.5e9,16.3e9,237e6/1.5,42.75,"sandw6")




al7075=material(2.80e3,72.5e9,27e9,444.5e6/1.5,13.15*(1-0.426)+2.61*0.426,"al7075") #from EDUPACK
#    al7075oas=material(2.78e3,73.1e9,73.1e9/2/1.33,444.5e6/1.5,13.15*(1-0.426)+2.61*0.426,"al7075") #from OAS example
qiCFRP=material(1565,54.9e9,21e9,670e6/1.5,48.1,"qiCFRP")
steel=material(7750,200e9,78.5e9,562e6/1.5,4.55*(1-0.374)+1.15*0.374,"steel")
gfrp=material(1860,21.4e9,8.14e9,255e6,6.18,"gfrp")            #epoxy-Eglass,woven,QI
#nomat=material(1370,0.01,0.01,0.01,60,"noMaterial")
nomat=material(50,1e8,1e4,1e5,6000,"noMaterial")    
#    nomat=material(50,1e8,1e4,1e5,60,"noMaterial")    
fakemat=material((2.80e3+7750)/2,(72.5e9+200e9)/2,(27e9+78.5e9)/2,(444.5e6/1.5+562e6/1.5)/2,(13.15*(1-0.426)+2.61*0.426+4.55*(1-0.374)+1.15*0.374)/2,"fakemat")
nomatEnd=material(10000,5e9,2e9,20e6/1.5,60,"nomatEnd")

materials=[al7075, qiCFRP, steel, gfrp]
#materials=[al7075, qiCFRP, steel, gfrp, nomat, fakemat, nomatEnd, sandw4, sandw5, sandw6]



##multimaterial young modulus
#plt.figure(1)    
#for x in materials:
#    plt.plot(x.mrho, x.E, 'ro')
#    
#plt.ylabel("Young's modulus (Pa)")
#plt.xlabel('density (kg/m3)')
#
#for x in range(1566,7750,10):
#    plt.plot(x, youngMM(x,materials,1), 'b.')
#    
#
##multimaterial shear modulus
#plt.figure(2)    
#for x in materials:
#    plt.plot(x.mrho, x.G, 'ro')
#    
#plt.ylabel("shear modulus")
#plt.xlabel('rho')
#for x in range(1566,7750,10):
#    plt.plot(x, shearMM(x,materials,5), 'b.')
#    
##multimaterial yield strength
#plt.figure(3)    
#for x in materials:
#    plt.plot(x.mrho, x.yields, 'ro')
#    
#plt.ylabel("yield strength")
#plt.xlabel('rho')
#
#for x in range(1566,7750,10):
#    plt.plot(x, yieldMM(x,materials,5), 'b.')
    
#multimaterial co2
plt.figure(4)    
for x in materials:
    plt.plot(x.mrho, x.co2, 'ro')
    
plt.ylabel("CO2 emissions (kg/kg)")
plt.xlabel('density (kg/m3)')

for x in range(1566,7750,1):
    plt.plot(x, co2MM(x,materials,5), 'b.')