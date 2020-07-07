# -*- coding: utf-8 -*-
"""
Created on Fri May 10 14:20:10 2019

@author: e.duriez
"""

import numpy as np

class material:
    def __init__(self, density, young, shear, yield_stress, co2, name):
        self.mrho = density        # [kg/m^3] material density
        self.E=young               # [Pa] Young's modulus
        self.G=shear               # [Pa] shear modulus 
        self.yields=yield_stress   # [Pa] allowable yield stress
        self.co2=co2               # [kg/kg] embeded co2
        self.name=name


#multimaterial young modulus
def youngMM(rho,materlist,puissanceMM):
    exponent=puissanceMM
    materialsSorted=sorted(materlist,key=lambda x: x.mrho)
    Emax=0
    materialsSorted2=materialsSorted
#    rho=abs(rho)
    if rho<=0.01:
        raise ValueError("rho must be > 0.01")
    elif rho>materialsSorted2[-1].mrho:
        raise ValueError("rho must be <= "+str(materialsSorted2[-1].mrho))
    else:
        for x in materialsSorted2:
            if x.mrho>=rho:
                mat2=x
                break
            else:
                mat1=x
        if mat2.E<mat1.E:  #case where E decreases with rho
            exponent=1   #ED
        rhop=(rho-mat1.mrho)/(mat2.mrho-mat1.mrho)
        scale=(mat2.E-mat1.E)
        offset=mat1.E
        angular_return=scale*(rhop**exponent)+offset
        # suppress function angle for better 
        if rho>=(mat1.mrho+mat2.mrho)/2: #case closer to mat2 than mat1
            if mat2==materialsSorted2[-1]:     #no angle suppression for last material
                cor_fact=0
            else:
                cor_fact=np.exp(-(rho-mat2.mrho)**2/(0.0005*(mat2.mrho-mat1.mrho)**2))    #gaussian correction factor : =1 on real materials
            closest_mat=mat2.E
        else:         ##case closer to mat1 than mat2
            if mat1==materialsSorted2[0]:     #no angle suppression for first material
                cor_fact=0
            else:
                cor_fact=np.exp(-(rho-mat1.mrho)**2/(0.0005*(mat2.mrho-mat1.mrho)**2))    #gaussian correction factor : =1 on real materials
            closest_mat=mat1.E
#        return cor_fact*closest_mat+(1-cor_fact)*angular_return
        return angular_return
        


    
#multimaterial shear modulus
def shearMM(rho,materlist,puissanceMM):
    exponent=puissanceMM
    materialsSorted=sorted(materlist,key=lambda x: x.mrho)
    shearmax=0
    materialsSorted2=materialsSorted
#    rho=abs(rho)
    if rho<=0.01:
        raise ValueError("rho must be > 0.01")
    elif rho>materialsSorted2[-1].mrho:
        raise ValueError("rho must be <= "+str(materialsSorted2[-1].mrho))
    else:
        for x in materialsSorted2:
            if x.mrho>=rho:
                mat2=x
                break
            else:
                mat1=x
        if mat2.G<mat1.G:  #ED
            exponent=1   #ED
        rhop=(rho-mat1.mrho)/(mat2.mrho-mat1.mrho)
        scale=(mat2.G-mat1.G)
        offset=mat1.G
        angular_return=scale*(rhop**exponent)+offset
        # suppress function angle for better 
        if rho>=(mat1.mrho+mat2.mrho)/2: #case closer to mat2 than mat1
            if mat2==materialsSorted2[-1]:     #no angle suppression for last material
                cor_fact=0
            else:
                cor_fact=np.exp(-(rho-mat2.mrho)**2/(0.0005*(mat2.mrho-mat1.mrho)**2))    #gaussian correction factor : =1 on real materials
            closest_mat=mat2.G
        else:         ##case closer to mat1 than mat2
            if mat1==materialsSorted2[0]:     #no angle suppression for first material
                cor_fact=0
            else:
                cor_fact=np.exp(-(rho-mat1.mrho)**2/(0.0005*(mat2.mrho-mat1.mrho)**2))    #gaussian correction factor : =1 on real materials
            closest_mat=mat1.G
#        return cor_fact*closest_mat+(1-cor_fact)*angular_return
        return angular_return

    
#multimaterial yield strength
def yieldMM(rho,materlist,puissanceMM):
    exponent=puissanceMM
    materialsSorted=sorted(materlist,key=lambda x: x.mrho)
    yieldmax=0
    materialsSorted2=materialsSorted
#    rho=abs(rho)
    if rho<=0.01:
        raise ValueError("rho must be > 0.01")
    elif rho>materialsSorted2[-1].mrho:
        raise ValueError("rho must be <= "+str(materialsSorted2[-1].mrho))
    else:
        for x in materialsSorted2:
            if x.mrho>=rho:
                mat2=x
                break
            else:
                mat1=x
        if mat2.yields<mat1.yields:  #ED
            exponent=1   #ED
        rhop=(rho-mat1.mrho)/(mat2.mrho-mat1.mrho)
        scale=(mat2.yields-mat1.yields)
        offset=mat1.yields
        angular_return=scale*(rhop**exponent)+offset
        # suppress function angle for better 
        if rho>=(mat1.mrho+mat2.mrho)/2: #case closer to mat2 than mat1
            if mat2==materialsSorted2[-1]:     #no angle suppression for last material
                cor_fact=0
            else:
                cor_fact=np.exp(-(rho-mat2.mrho)**2/(0.0005*(mat2.mrho-mat1.mrho)**2))    #gaussian correction factor : =1 on real materials
            closest_mat=mat2.yields
        else:         ##case closer to mat1 than mat2
            if mat1==materialsSorted2[0]:     #no angle suppression for first material
                cor_fact=0
            else:
                cor_fact=np.exp(-(rho-mat1.mrho)**2/(0.0005*(mat2.mrho-mat1.mrho)**2))    #gaussian correction factor : =1 on real materials
            closest_mat=mat1.yields
#        return cor_fact*closest_mat+(1-cor_fact)*angular_return
        return angular_return

    
#multimaterial co2
def co2MM(rho,materlist,puissanceMM):
    exponent=puissanceMM
    materialsSorted=sorted(materlist,key=lambda x: x.mrho)
    co2max=0
    materialsSorted2=materialsSorted
#    rho=abs(rho)
    if rho<=0.01:
        raise ValueError("rho must be > 0.01")
    elif rho>materialsSorted2[-1].mrho:
        raise ValueError("rho must be <= "+str(materialsSorted2[-1].mrho))
    else:
        for x in materialsSorted2:
            if x.mrho>=rho:
                mat2=x
                break
            else:
                mat1=x
        if mat2.co2>mat1.co2:  #ED
            exponent=1   #ED
        rhop=(rho-mat1.mrho)/(mat2.mrho-mat1.mrho)
        scale=(mat2.co2-mat1.co2)
        offset=mat1.co2
        angular_return=scale*(rhop**exponent)+offset
        # suppress function angle for better 
        if rho>=(mat1.mrho+mat2.mrho)/2: #case closer to mat2 than mat1
            if mat2==materialsSorted2[-1]:     #no angle suppression for last material
                cor_fact=0
            else:
                cor_fact=np.exp(-(rho-mat2.mrho)**2/(0.0005*(mat2.mrho-mat1.mrho)**2))    #gaussian correction factor : =1 on real materials
            closest_mat=mat2.co2
        else:         ##case closer to mat1 than mat2
            if mat1==materialsSorted2[0]:     #no angle suppression for first material
                cor_fact=0
            else:
                cor_fact=np.exp(-(rho-mat1.mrho)**2/(0.0005*(mat2.mrho-mat1.mrho)**2))    #gaussian correction factor : =1 on real materials
            closest_mat=mat1.co2
#        return cor_fact*closest_mat+(1-cor_fact)*angular_return
        return angular_return
