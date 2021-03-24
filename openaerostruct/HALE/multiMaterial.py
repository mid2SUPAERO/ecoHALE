# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 10:01:37 2020

@author: Victor M. Guadano
"""


from __future__ import division, print_function
from openmdao.api import ExplicitComponent
##import math
import numpy as np


class material:
    def __init__(self, density, young, shear, yield_stress, co2, name):
        self.mrho = density        # [kg/m^3] material density
        self.E=young               # [Pa] Young's modulus
        self.G=shear               # [Pa] shear modulus 
        self.yields=yield_stress   # [Pa] allowable yield stress
        self.co2=co2               # [kg/kg] embeded co2
        self.name=name
        
        
# multimaterial young modulus
class YoungMM(ExplicitComponent):
    
    def initialize(self):
        self.options.declare('surface', types=dict)

    def setup(self):
        surface = self.options['surface']
        self.ny = surface['mesh'].shape[1]
        
        self.add_input('mrho', val=np.array([1000, 1000]), units='kg/m**3')
        #self.add_input('puissanceMM', val=1)
        #self.add_input('materlist')
        
        self.add_output('young', val=np.array([1e10, 1e10]), units= 'N/m**2')
        
        self.declare_partials('young','mrho')
        #self.declare_partials('young','puissanceMM')
        
    def compute(self, inputs, outputs):
        
        surface = self.options['surface']
        exponent = surface['puissanceMM']
        materlist = surface['materlist']
        rho = inputs['mrho']
        Nmaterial = surface['Nmaterial']
        
        materialsSorted=sorted(materlist,key=lambda x: x.mrho)
        ##Emax=0
        materialsSorted2=materialsSorted
        #    rho=abs(rho)
        angular_return = np.zeros(2)
        
        if Nmaterial == 1:
            if rho[0]<=0.01:
                raise ValueError("rho must be > 0.01")
            elif rho[0]>materialsSorted2[-1].mrho:
                raise ValueError("rho must be <= "+str(materialsSorted2[-1].mrho))
            else:
                for x in materialsSorted2:
                    if x.mrho>=rho[0]:
                        mat2=x
                        break
                    else:
                        mat1=x
                if mat2.E<mat1.E:  #case where E decreases with rho
                    exponent=1   #ED
                rhop=(rho[0]-mat1.mrho)/(mat2.mrho-mat1.mrho)
                scale=(mat2.E-mat1.E)
                offset=mat1.E
                angular_return[:]=scale*(rhop**exponent)+offset
        elif Nmaterial == 2:
            for i in range(2):
                if rho[i]<=0.01:
                    raise ValueError("rho must be > 0.01")
                elif rho[i]>materialsSorted2[-1].mrho:
                    raise ValueError("rho must be <= "+str(materialsSorted2[-1].mrho))
                else:
                    for x in materialsSorted2:
                        if x.mrho>=rho[i]:
                            mat2=x
                            break
                        else:
                            mat1=x
                    if mat2.E<mat1.E:  #case where E decreases with rho
                        exponent=1   #ED
                    rhop=(rho[i]-mat1.mrho)/(mat2.mrho-mat1.mrho)
                    scale=(mat2.E-mat1.E)
                    offset=mat1.E
                    angular_return[i]=scale*(rhop**exponent)+offset
        else:
            raise ValueError("Nmaterial must be equal to 1 or 2")
                    
        outputs['young'] = angular_return

    def compute_partials(self, inputs, partials):
        
        surface = self.options['surface']
        exponent = surface['puissanceMM']
        materlist = surface['materlist']
        rho = inputs['mrho']
        Nmaterial = surface['Nmaterial']
                       
        materialsSorted=sorted(materlist,key=lambda x: x.mrho)
        ##Emax=0
        materialsSorted2=materialsSorted
        #    rho=abs(rho)
        partial_return = np.zeros((2,2))
        
        if Nmaterial == 1:
            if rho[0]<=0.01:
                raise ValueError("rho must be > 0.01")
            elif rho[0]>materialsSorted2[-1].mrho:
                raise ValueError("rho must be <= "+str(materialsSorted2[-1].mrho))
            else:
                for x in materialsSorted2:
                    if x.mrho>=rho[0]:
                        mat2=x
                        break
                    else:
                        mat1=x
                if mat2.E<mat1.E:  #case where E decreases with rho
                    exponent=1   #ED
                rhop=(rho[0]-mat1.mrho)/(mat2.mrho-mat1.mrho)
                scale=(mat2.E-mat1.E)
                partial_return[0,0] = exponent*(scale)*(rhop)**(exponent-1)/(mat2.mrho-mat1.mrho)  
                #partial_return[1,1] = partial_return[0,0]
        elif Nmaterial == 2:
            for i in range(2):
                if rho[i]<=0.01:
                    raise ValueError("rho must be > 0.01")
                elif rho[i]>materialsSorted2[-1].mrho:
                    raise ValueError("rho must be <= "+str(materialsSorted2[-1].mrho))
                else:
                    for x in materialsSorted2:
                        if x.mrho>=rho[i]:
                            mat2=x
                            break
                        else:
                            mat1=x
                    if mat2.E<mat1.E:  #case where E decreases with rho
                        exponent=1   #ED
                    rhop=(rho[i]-mat1.mrho)/(mat2.mrho-mat1.mrho)
                    scale=(mat2.E-mat1.E)
                    ##offset=mat1.E
                    ##angular_return=scale*(rhop**exponent)+offset
                    partial_return[i,i] = exponent*(scale)*(rhop)**(exponent-1)/(mat2.mrho-mat1.mrho)
        else:
            raise ValueError("Nmaterial must be equal to 1 or 2")
            
        partials['young', 'mrho'] = partial_return           


# multimaterial shear modulus
class ShearMM(ExplicitComponent):
    
    def initialize(self):
        self.options.declare('surface', types=dict)

    def setup(self):
        surface = self.options['surface']
        self.ny = surface['mesh'].shape[1]
        
        self.add_input('mrho', val=np.array([1000, 1000]), units='kg/m**3')
        #self.add_input('puissanceMM', val=1)
        #self.add_input('materlist')
        
        self.add_output('shear', val=np.array([1e10, 1e10]), units= 'N/m**2')
        
        self.declare_partials('shear','mrho')
        #self.declare_partials('young','puissanceMM')
        
    def compute(self, inputs, outputs):
        
        surface = self.options['surface']
        exponent = surface['puissanceMM']
        materlist = surface['materlist']
        rho = inputs['mrho']
        Nmaterial = surface['Nmaterial']
        
        materialsSorted=sorted(materlist,key=lambda x: x.mrho)
        ##shearmax=0
        materialsSorted2=materialsSorted
    #    rho=abs(rho)
        angular_return = np.zeros(2)
        
        if Nmaterial == 1:
            if rho[0]<=0.01:
                raise ValueError("rho must be > 0.01")
            elif rho[0]>materialsSorted2[-1].mrho:
                raise ValueError("rho must be <= "+str(materialsSorted2[-1].mrho))
            else:
                for x in materialsSorted2:
                    if x.mrho>=rho[0]:
                        mat2=x
                        break
                    else:
                        mat1=x
                if mat2.G<mat1.G:  #ED
                    exponent=1   #ED
                rhop=(rho[0]-mat1.mrho)/(mat2.mrho-mat1.mrho)
                scale=(mat2.G-mat1.G)
                offset=mat1.G
                angular_return[:]=scale*(rhop**exponent)+offset
        elif Nmaterial == 2:
            for i in range(2):
                if rho[i]<=0.01:
                    raise ValueError("rho must be > 0.01")
                elif rho[i]>materialsSorted2[-1].mrho:
                    raise ValueError("rho must be <= "+str(materialsSorted2[-1].mrho))
                else:
                    for x in materialsSorted2:
                        if x.mrho>=rho[i]:
                            mat2=x
                            break
                        else:
                            mat1=x
                    if mat2.G<mat1.G:  #ED
                        exponent=1   #ED
                    rhop=(rho[i]-mat1.mrho)/(mat2.mrho-mat1.mrho)
                    scale=(mat2.G-mat1.G)
                    offset=mat1.G
                    angular_return[i]=scale*(rhop**exponent)+offset
        else:
            raise ValueError("Nmaterial must be equal to 1 or 2")

        outputs['shear'] = angular_return
            
    def compute_partials(self, inputs, partials):
        
        surface = self.options['surface']
        exponent = surface['puissanceMM']
        materlist = surface['materlist']
        rho = inputs['mrho']
        Nmaterial = surface['Nmaterial']
        
        materialsSorted=sorted(materlist,key=lambda x: x.mrho)
        ##shearmax=0
        materialsSorted2=materialsSorted
    #    rho=abs(rho)
        partial_return = np.zeros((2,2))
        
        if Nmaterial == 1:
            if rho[0]<=0.01:
                raise ValueError("rho must be > 0.01")
            elif rho[0]>materialsSorted2[-1].mrho:
                raise ValueError("rho must be <= "+str(materialsSorted2[-1].mrho))
            else:
                for x in materialsSorted2:
                    if x.mrho>=rho[0]:
                        mat2=x
                        break
                    else:
                        mat1=x
                if mat2.G<mat1.G:  #ED
                    exponent=1   #ED
                rhop=(rho[0]-mat1.mrho)/(mat2.mrho-mat1.mrho)
                scale=(mat2.G-mat1.G)
                partial_return[0,0] = exponent*(scale)*(rhop)**(exponent-1)/(mat2.mrho-mat1.mrho)  
                #partial_return[1,1] = partial_return[0,0]
        elif Nmaterial == 2:    
            for i in range(2):
                if rho[i]<=0.01:
                    raise ValueError("rho must be > 0.01")
                elif rho[i]>materialsSorted2[-1].mrho:
                    raise ValueError("rho must be <= "+str(materialsSorted2[-1].mrho))
                else:
                    for x in materialsSorted2:
                        if x.mrho>=rho[i]:
                            mat2=x
                            break
                        else:
                            mat1=x
                    if mat2.G<mat1.G:  #ED
                        exponent=1   #ED
                    rhop=(rho[i]-mat1.mrho)/(mat2.mrho-mat1.mrho)
                    scale=(mat2.G-mat1.G)
                    ##offset=mat1.G
                    ##angular_return=scale*(rhop**exponent)+offset
                    partial_return[i,i] = exponent*(scale)*(rhop)**(exponent-1)/(mat2.mrho-mat1.mrho)
        else:
            raise ValueError("Nmaterial must be equal to 1 or 2")
            
        partials['shear', 'mrho'] = partial_return
            

# multimaterial yield strength
class YieldMM(ExplicitComponent):
    
    def initialize(self):
        self.options.declare('surface', types=dict)

    def setup(self):
        surface = self.options['surface']
        self.ny = surface['mesh'].shape[1]
        
        self.add_input('mrho', val=np.array([1000, 1000]), units='kg/m**3')
        #self.add_input('puissanceMM', val=1)
        #self.add_input('materlist')
        
        self.add_output('yield', val=np.array([1e8, 1e8]), units= 'N/m**2')
        
        self.declare_partials('yield','mrho')
        #self.declare_partials('young','puissanceMM')
        
    def compute(self, inputs, outputs):
        
        surface = self.options['surface']
        exponent = surface['puissanceMM']
        materlist = surface['materlist']
        rho = inputs['mrho']
        Nmaterial = surface['Nmaterial']
        
        materialsSorted=sorted(materlist,key=lambda x: x.mrho)
        ##yieldmax=0
        materialsSorted2=materialsSorted
    #    rho=abs(rho)
        angular_return = np.zeros(2)
        
        if Nmaterial == 1:
            if rho[0]<=0.01:
                raise ValueError("rho must be > 0.01")
            elif rho[0]>materialsSorted2[-1].mrho:
                raise ValueError("rho must be <= "+str(materialsSorted2[-1].mrho))
            else:
                for x in materialsSorted2:
                    if x.mrho>=rho[0]:
                        mat2=x
                        break
                    else:
                        mat1=x
                if mat2.yields<mat1.yields:  #ED
                    exponent=1   #ED
                rhop=(rho[0]-mat1.mrho)/(mat2.mrho-mat1.mrho)
                scale=(mat2.yields-mat1.yields)
                offset=mat1.yields
                angular_return[:]=scale*(rhop**exponent)+offset
        elif Nmaterial == 2:
            for i in range(2):
                if rho[i]<=0.01:
                    raise ValueError("rho must be > 0.01")
                elif rho[i]>materialsSorted2[-1].mrho:
                    raise ValueError("rho must be <= "+str(materialsSorted2[-1].mrho))
                else:
                    for x in materialsSorted2:
                        if x.mrho>=rho[i]:
                            mat2=x
                            break
                        else:
                            mat1=x
                    if mat2.yields<mat1.yields:  #ED
                        exponent=1   #ED
                    rhop=(rho[i]-mat1.mrho)/(mat2.mrho-mat1.mrho)
                    scale=(mat2.yields-mat1.yields)
                    offset=mat1.yields
                    angular_return[i]=scale*(rhop**exponent)+offset
        else:
            raise ValueError("Nmaterial must be equal to 1 or 2")
          
        outputs['yield'] = angular_return
            
    def compute_partials(self, inputs, partials):
        
        surface = self.options['surface']
        exponent = surface['puissanceMM']
        materlist = surface['materlist']
        rho = inputs['mrho']
        Nmaterial = surface['Nmaterial']
        
        materialsSorted=sorted(materlist,key=lambda x: x.mrho)
        ##yieldmax=0
        materialsSorted2=materialsSorted
    #    rho=abs(rho)
        partial_return = np.zeros((2,2))
        
        if Nmaterial == 1:
            if rho[0]<=0.01:
                raise ValueError("rho must be > 0.01")
            elif rho[0]>materialsSorted2[-1].mrho:
                raise ValueError("rho must be <= "+str(materialsSorted2[-1].mrho))
            else:
                for x in materialsSorted2:
                    if x.mrho>=rho[0]:
                        mat2=x
                        break
                    else:
                        mat1=x
                if mat2.yields<mat1.yields:  #ED
                    exponent=1   #ED
                rhop=(rho[0]-mat1.mrho)/(mat2.mrho-mat1.mrho)
                scale=(mat2.yields-mat1.yields)
                partial_return[0,0] = exponent*(scale)*(rhop)**(exponent-1)/(mat2.mrho-mat1.mrho)  
                #partial_return[1,1] = partial_return[0,0]
        elif Nmaterial == 2:
            for i in range(2):
                if rho[i]<=0.01:
                    raise ValueError("rho must be > 0.01")
                elif rho[i]>materialsSorted2[-1].mrho:
                    raise ValueError("rho must be <= "+str(materialsSorted2[-1].mrho))
                else:
                    for x in materialsSorted2:
                        if x.mrho>=rho[i]:
                            mat2=x
                            break
                        else:
                            mat1=x
                    if mat2.yields<mat1.yields:  #ED
                        exponent=1   #ED
                    rhop=(rho[i]-mat1.mrho)/(mat2.mrho-mat1.mrho)
                    scale=(mat2.yields-mat1.yields)
                    ##offset=mat1.yields
                    ##angular_return=scale*(rhop**exponent)+offset
                    partial_return[i,i] = exponent*(scale)*(rhop)**(exponent-1)/(mat2.mrho-mat1.mrho)
        else:
            raise ValueError("Nmaterial must be equal to 1 or 2")
            
        partials['yield', 'mrho'] = partial_return


# multimaterial co2
class CO2MM(ExplicitComponent):
    
    def initialize(self):
        self.options.declare('surface', types=dict)

    def setup(self):
        surface = self.options['surface']
        self.ny = surface['mesh'].shape[1]
        
        self.add_input('mrho', val=np.array([1000, 1000]), units='kg/m**3')
        #self.add_input('puissanceMM', val=1)
        #self.add_input('materlist')
        
        self.add_output('co2', val=np.array([50, 50]), units= 'kg/kg')
        
        self.declare_partials('co2','mrho')
        #self.declare_partials('young','puissanceMM')
        
    def compute(self, inputs, outputs):
        
        surface = self.options['surface']
        exponent = surface['puissanceMM']
        materlist = surface['materlist']
        rho = inputs['mrho']
        Nmaterial = surface['Nmaterial']
        
        materialsSorted=sorted(materlist,key=lambda x: x.mrho)
        ##co2max=0
        materialsSorted2=materialsSorted
    #    rho=abs(rho)
        angular_return = np.zeros(2)
        
        if Nmaterial == 1:
            if rho[0]<=0.01:
                raise ValueError("rho must be > 0.01")
            elif rho[0]>materialsSorted2[-1].mrho:
                raise ValueError("rho must be <= "+str(materialsSorted2[-1].mrho))
            else:
                for x in materialsSorted2:
                    if x.mrho>=rho[0]:
                        mat2=x
                        break
                    else:
                        mat1=x
                if mat2.co2>mat1.co2:  #ED
                    exponent=1   #ED
                rhop=(rho[0]-mat1.mrho)/(mat2.mrho-mat1.mrho)
                scale=(mat2.co2-mat1.co2)
                offset=mat1.co2
                angular_return[:]=scale*(rhop**exponent)+offset
        elif Nmaterial == 2:
            for i in range(2):
                if rho[i]<=0.01:
                    raise ValueError("rho must be > 0.01")
                elif rho[i]>materialsSorted2[-1].mrho:
                    raise ValueError("rho must be <= "+str(materialsSorted2[-1].mrho))
                else:
                    for x in materialsSorted2:
                        if x.mrho>=rho[i]:
                            mat2=x
                            break
                        else:
                            mat1=x
                    if mat2.co2>mat1.co2:  #ED
                        exponent=1   #ED
                    rhop=(rho[i]-mat1.mrho)/(mat2.mrho-mat1.mrho)
                    scale=(mat2.co2-mat1.co2)
                    offset=mat1.co2
                    angular_return[i]=scale*(rhop**exponent)+offset
        else:
            raise ValueError("Nmaterial must be equal to 1 or 2")

        outputs['co2'] = angular_return
            
    def compute_partials(self, inputs, partials):
        
        surface = self.options['surface']
        exponent = surface['puissanceMM']
        materlist = surface['materlist']
        rho = inputs['mrho']
        Nmaterial = surface['Nmaterial']
        
        materialsSorted=sorted(materlist,key=lambda x: x.mrho)
        ##co2max=0
        materialsSorted2=materialsSorted
    #    rho=abs(rho)
        partial_return = np.zeros((2,2))
        
        if Nmaterial == 1:
            if rho[0]<=0.01:
                raise ValueError("rho must be > 0.01")
            elif rho[0]>materialsSorted2[-1].mrho:
                raise ValueError("rho must be <= "+str(materialsSorted2[-1].mrho))
            else:
                for x in materialsSorted2:
                    if x.mrho>=rho[0]:
                        mat2=x
                        break
                    else:
                        mat1=x
                if mat2.co2>mat1.co2:  #ED
                    exponent=1   #ED
                rhop=(rho[0]-mat1.mrho)/(mat2.mrho-mat1.mrho)
                scale=(mat2.co2-mat1.co2)
                partial_return[0,0] = exponent*(scale)*(rhop)**(exponent-1)/(mat2.mrho-mat1.mrho)
                #partial_return[1,1] = partial_return[0,0]
        elif Nmaterial == 2:
            for i in range(2):
                if rho[i]<=0.01:
                    raise ValueError("rho must be > 0.01")
                elif rho[i]>materialsSorted2[-1].mrho:
                    raise ValueError("rho must be <= "+str(materialsSorted2[-1].mrho))
                else:
                    for x in materialsSorted2:
                        if x.mrho>=rho[i]:
                            mat2=x
                            break
                        else:
                            mat1=x
                    if mat2.co2>mat1.co2:  #ED
                        exponent=1   #ED
                    rhop=(rho[i]-mat1.mrho)/(mat2.mrho-mat1.mrho)
                    scale=(mat2.co2-mat1.co2)
                    ##offset=mat1.co2
                    ##angular_return=scale*(rhop**exponent)+offset
                    partial_return[i,i] = exponent*(scale)*(rhop)**(exponent-1)/(mat2.mrho-mat1.mrho)
        else:
            raise ValueError("Nmaterial must be equal to 1 or 2")
            
        partials['co2', 'mrho'] = partial_return