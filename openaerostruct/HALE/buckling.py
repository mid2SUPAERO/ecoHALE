# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 17:08:36 2019

@author: e.duriez
"""

from __future__ import division, print_function
from openmdao.api import ExplicitComponent
import math
import numpy as np
##from openaerostruct.HALE.fctMultiMatos import*


class BucklingKS(ExplicitComponent):
    """
    computes the buckling optimisation constraint. 
    If it is positive at least one wing panel buckles; if it is negative, no buckling appears for the panels.
    
    Parameters
    ----------
    buckling_coef : float
        buckling coefficient depending on ribs ans stringer spacing
    inter_stringer : float
        distance between two consecutive stringers
    E : float
        young modulus of material
    G : float
        shear modulus of material
    top_bending_stress : float
        stress in upper skin


    Returns
    -------
    buckling : float
        Equality constraint for stress in skin = buckling limit stress. buckling < 0 for the
        constraint to be satisfied. KS aggregation quantity obtained by combining the buckling criteria
        for each FEM node. Used to simplify the optimization problem by
        reducing the number of constraints.
        
    """

    def initialize(self):
        self.options.declare('surface', types=dict)
        self.options.declare('rho', types=float, default=100.)

    def setup(self):
        
        surface = self.options['surface']        
        self.ny = surface['mesh'].shape[1]
        
        self.add_input('top_bending_stress', val=np.zeros(self.ny-1), units='N/m**2')
        self.add_input('skin_thickness', val=np.zeros((self.ny - 1)), units='m')
        ##self.add_input('mrho', val=1000, units='kg/m**3')
        self.add_input('chord', val=1, units='m')
        self.add_input('taper', val=1)
        
        self.add_input('young', val=1e10, units= 'N/m**2')  #VMGM
        self.add_input('shear', val=1e10, units= 'N/m**2')  #VMGM
        
        self.add_output('buckling', val=1.)

        self.declare_partials('buckling','top_bending_stress')
        self.declare_partials('buckling','skin_thickness')
        ##self.declare_partials('buckling', 'mrho', method='cs')
        self.declare_partials('buckling','chord')
        self.declare_partials('buckling','taper')
        
        self.declare_partials('buckling','young')   #VMGM
        self.declare_partials('buckling','shear')   #VMGM

    def compute(self, inputs, outputs):
        surface = self.options['surface']
        rho=self.options['rho']
        tbc=-inputs['top_bending_stress'] #"-" to turn compression into positive value
        skin=inputs['skin_thickness']
        kc=surface['buckling_coef']

        rootchord=inputs['chord']/2 # half the chord between two spars
        taper=inputs['taper']
        chords=rootchord*(1-(1-taper)*(1-(np.arange(len(skin))+0.5)/(len(skin))))

        #b=surface['inter_stringer']
        ##mrho = inputs['mrho']
        ##G = shearMM(mrho,surface['materlist'],surface['puissanceMM'])
        ##E = youngMM(mrho,surface['materlist'],surface['puissanceMM'])

        E = inputs['young'] #VMGM
        G = inputs['shear'] #VMGM        
        
        sigmaBuc=kc*math.pi**2*skin**2/chords**2*G**2/(3*(4*G-E))
        fmax = np.max(tbc/sigmaBuc - 1)

        nlog, nsum, nexp = np.log, np.sum, np.exp
        ks = 1 / rho * nlog(nsum(nexp(rho * (tbc/sigmaBuc - 1 - fmax))))
        outputs['buckling'] = fmax + ks       

        
#        if math.floor(outputs['buckling'])==-586244:  #TODELETE
#            print('there we are')  #TODELETE        
        
    def compute_partials(self, inputs, partials):
        tbc=inputs['top_bending_stress']

        surface = self.options['surface']
        rho=self.options['rho']
        tbc=-inputs['top_bending_stress'] # "-" to turn compression into positive value
        skin=inputs['skin_thickness']
        kc=surface['buckling_coef']
        
        rootchord=inputs['chord']/2
        taper=inputs['taper']
        chords=rootchord*(1-(1-taper)*(1-(np.arange(len(skin))+0.5)/(len(skin))))
        #b=surface['inter_stringer']
        ##mrho = inputs['mrho']
        ##G = shearMM(mrho,surface['materlist'],surface['puissanceMM'])
        ##E = youngMM(mrho,surface['materlist'],surface['puissanceMM'])  
        
        E = inputs['young'] #VMGM
        G = inputs['shear'] #VMGM  
        
        sigmaBuc=kc*math.pi**2*skin**2/chords**2*G**2/(3*(4*G-E))

        rho=self.options['rho']

        # Find the location of the max stress constraint
        fmax = np.max(tbc / sigmaBuc - 1)
        i = np.where((tbc / sigmaBuc - 1)==fmax)
        i = i[0]

        tempb0 = 1 / (rho * np.sum(np.exp(rho * (tbc/sigmaBuc - fmax - 1))))
        tempb = np.exp(rho*(tbc/sigmaBuc-fmax-1))*rho*tempb0
        fmaxb = 1 - np.sum(tempb)

        # Populate the entries
        derivs = tempb / sigmaBuc
        derivs[i] += fmaxb / sigmaBuc[i]

        # Reshape and save them to the jac dict
        partials['buckling', 'top_bending_stress'] = -derivs.reshape(-1)  #  "-" because tbc = - input...   again in next line
        partials['buckling', 'skin_thickness'] = -partials['buckling', 'top_bending_stress']*(-tbc/sigmaBuc)*2*(sigmaBuc/skin)
        derivChord=-partials['buckling', 'top_bending_stress']*(-tbc/sigmaBuc)*(-2)*(sigmaBuc/chords)*(1-(1-taper)*(1-(np.arange(len(skin))+0.5)/(len(skin))))/2
        partials['buckling', 'chord'] = np.sum(derivChord[0])
        derivTaper = -partials['buckling', 'top_bending_stress']*(-tbc/sigmaBuc)*(-2)*(sigmaBuc/chords)*rootchord*(1-(np.arange(len(skin))+0.5)/(len(skin)))
        partials['buckling', 'taper'] = np.sum(derivTaper[0])
        
        derivYoung = -partials['buckling', 'top_bending_stress']*(-tbc/sigmaBuc)*(sigmaBuc/(4*G-E)) #VMGM
        partials['buckling', 'young'] = np.sum(derivYoung[0])    #VMGM
        derivShear = -partials['buckling', 'top_bending_stress']*(-tbc/sigmaBuc)*(2*sigmaBuc/G-12*sigmaBuc/(3*(4*G-E))) #VMGM
        partials['buckling', 'shear'] = np.sum(derivShear[0])    #VMGM