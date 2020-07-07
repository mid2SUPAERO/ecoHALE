# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 17:28:06 2019

@author: e.duriez
"""

from __future__ import print_function
import numpy as np
import math
from openmdao.api import ExplicitComponent




class checkThickness(ExplicitComponent):

    def initialize(self):
        self.options.declare('surface', types=dict)

    def setup(self):
        surface = self.options['surface']

        self.ny=len(surface['t_over_c_cp'])

        self.add_input('t_over_c', val=np.zeros((self.ny)))#ED
        self.add_input('chordroot', val=1, units='m')#ED
        self.add_input('skinThickness', val=np.zeros((self.ny)), units='m')#ED
        self.add_input('taper', val=0.1)#ED

        self.add_output('acceptableThickness', val=np.zeros((self.ny)) )#ED

        self.declare_partials('acceptableThickness', 'skinThickness')
        self.declare_partials('acceptableThickness', 't_over_c')
        self.declare_partials('acceptableThickness', 'chordroot')
        self.declare_partials('acceptableThickness', 'taper')

#        self.declare_partials('emitted_co2', 'mrho', method='cs')

    def compute(self, inputs, outputs):
        surface = self.options['surface']
        
        t_over_c = inputs['t_over_c']
        taper = inputs['taper'][0]
        chordroot = inputs['chordroot'][0]
        skinThickness = inputs['skinThickness']
        
        minWingThickness=min(surface['data_y_upper'].real-surface['data_y_lower'].real)
        maxWingThickness=max(surface['data_y_upper'].real-surface['data_y_lower'].real)
        minToverC_ratio=minWingThickness/maxWingThickness

        num0=len(skinThickness)
        wing_min_thickness = [] #local minimum thickness of the wing for each point
        for i in range(num0) :
            wing_min_thickness.append(((i)/num0*chordroot+(num0-i)/num0*chordroot*taper)*t_over_c[i]*minToverC_ratio)

        acceptableThickness=2*skinThickness-wing_min_thickness
        if any(x>0 for x in acceptableThickness):
            print('skin problem')

        outputs['acceptableThickness']=2*skinThickness-wing_min_thickness


    def compute_partials(self, inputs, partials):
        surface = self.options['surface']
        t_over_c = inputs['t_over_c']
        taper = inputs['taper']
        chordroot = inputs['chordroot']
        skinThickness = inputs['skinThickness']
        
        minWingThickness=min(surface['data_y_upper'].real-surface['data_y_lower'].real)
        maxWingThickness=max(surface['data_y_upper'].real-surface['data_y_lower'].real)
        minToverC_ratio=minWingThickness/maxWingThickness

        num0=len(skinThickness)
        local_min_thickness = []
        diffLocal_min_thicknessChordroot = []
        diffLocal_min_thicknessTaper = []
        diffLocal_min_thicknessT_over_c = []
        for i in range(num0) :
            local_min_thickness.append(((i)/num0*chordroot+(num0-i)/num0*chordroot*taper)*t_over_c[i]*minToverC_ratio)
            diffLocal_min_thicknessChordroot.append(-((i)/num0+(num0-i)/num0*taper)*t_over_c[i]*minToverC_ratio)
            diffLocal_min_thicknessTaper.append(-((num0-i)/num0*chordroot)*t_over_c[i]*minToverC_ratio)
            diffLocal_min_thicknessT_over_c.append(-((i)/num0*chordroot+(num0-i)/num0*chordroot*taper)*minToverC_ratio)

        

        partials['acceptableThickness', 'skinThickness'] = [[2,0,0,0],[0,2,0,0],[0,0,2,0],[0,0,0,2]]
        partials['acceptableThickness', 't_over_c'] = np.diag(np.transpose(np.array(diffLocal_min_thicknessT_over_c))[0])
        partials['acceptableThickness', 'chordroot'] = diffLocal_min_thicknessChordroot
        partials['acceptableThickness', 'taper'] = diffLocal_min_thicknessTaper
