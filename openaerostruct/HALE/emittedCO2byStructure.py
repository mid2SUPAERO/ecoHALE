# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 17:28:06 2019

@author: e.duriez
"""

from __future__ import print_function
import numpy as np
import math
from openmdao.api import ExplicitComponent

##from fctMultiMatos import*



class structureCO2(ExplicitComponent):

    def initialize(self):
        self.options.declare('surfaces', types=list)

    def setup(self):

        ##self.add_input('mrho', val=1000, units='kg/m**3') #ED
        self.add_input('mass', val=100, units='kg')#ED
        self.add_input('PV_mass', val=100, units='kg')#ED
        
        self.add_input('co2', val=np.array([50, 50]), units='kg/kg')  #VMGM
        self.add_input('spars_mass', val=100, units='kg')  #VMGM
        
        #code added for debug
#        self.ny=len(surface['t_over_c_cp']) #TODELETE
#        self.add_input('twist', val=np.zeros((self.ny))) #TODELETE
#        self.add_input('spar_thickness', val=np.zeros((self.ny))) #TODELETE
#        self.add_input('skin_thickness', val=np.zeros((self.ny))) #TODELETE
#        self.add_input('span', val=5) #TODELETE
#        self.add_input('chord', val=2) #TODELETE
#        self.add_input('taper', val=0.1) #TODELETE
#        self.add_input('t_over_c', val=np.zeros((self.ny))) #TODELETE
#        self.add_input('alpha_maneuver', val=1) #TODELETE
#        self.add_input('mrhoVar', val=1700) #TODELETE

        self.add_output('emitted_co2', val=20000, units='kg')#ED

        self.declare_partials('emitted_co2', 'mass')
        self.declare_partials('emitted_co2', 'PV_mass')
#        self.declare_partials('emitted_co2', 'mrho', method='fd', step=0.1, step_calc='abs')
        ##self.declare_partials('emitted_co2', 'mrho', method='cs')
        
        self.declare_partials('emitted_co2', 'co2')  #VMGM 
        self.declare_partials('emitted_co2', 'spars_mass')  #VMGM        
        
    def compute(self, inputs, outputs):
        surfaces = self.options['surfaces']
        for surface in surfaces:
            ##puissanceMM = surface['puissanceMM']
            ##materlist=surface['materlist']
            PVco2=surface['co2PV']

#        print('structCO2') #ED
#        print(inputs['mrho'])  #ED
#        print(inputs['mass'])
#        if math.floor(abs(inputs['mass']))==31116:  #TODELETE
#            print('there we are')  #TODELETE
        

        ##co2 = co2MM(inputs['mrho'],materlist,puissanceMM)  #ED
        
        co2 = inputs['co2'] #VMGM
        
        outputs['emitted_co2']=co2[0]*inputs['spars_mass'] + co2[1]*(inputs['mass']-inputs['spars_mass']) + PVco2*inputs['PV_mass']

    def compute_partials(self, inputs, partials):
        surfaces = self.options['surfaces']
        for surface in surfaces:
            ##puissanceMM = surface['puissanceMM']
            ##materlist=surface['materlist']
            PVco2=surface['co2PV']

        ##co2 = co2MM(inputs['mrho'],materlist,puissanceMM)  #ED  
        
        co2 = inputs['co2']  #VMGM
        mass = inputs['mass']  #VMGM
        spars_mass = inputs['spars_mass']  #VMGM

        partials['emitted_co2', 'mass']=co2[1]
        partials['emitted_co2', 'PV_mass']=PVco2
        partials['emitted_co2', 'co2']=np.array([spars_mass, mass-spars_mass])  #VMGM
        partials['emitted_co2', 'spars_mass']=co2[0] - co2[1]  #VMGM