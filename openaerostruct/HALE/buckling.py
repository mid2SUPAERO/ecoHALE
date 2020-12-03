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
        
        self.add_input('young', val=np.array([1e10, 1e10]), units= 'N/m**2')  #VMGM
        self.add_input('shear', val=np.array([1e10, 1e10]), units= 'N/m**2')  #VMGM
        
        self.add_input('t_over_c', val=np.zeros((self.ny - 1)))  #VMGM
        self.add_input('horizontal_shear', val=np.zeros(self.ny-1), units='N/m**2')  #VMGM
        
        self.add_output('buckling', val=1.)
        
        self.declare_partials('*', '*', method='cs')

        """
        self.declare_partials('buckling','top_bending_stress')
        self.declare_partials('buckling','skin_thickness')
        ##self.declare_partials('buckling', 'mrho', method='cs')
        self.declare_partials('buckling','chord')
        self.declare_partials('buckling','taper')
        
        self.declare_partials('buckling','young')   #VMGM
        self.declare_partials('buckling','shear')   #VMGM
        
        self.declare_partials('buckling','t_over_c')   #VMGM """

    def compute(self, inputs, outputs):
        surface = self.options['surface']
        rho=self.options['rho']
        tbc=-inputs['top_bending_stress'] #"-" to turn compression into positive value
        skin=inputs['skin_thickness']

        rootchord=inputs['chord']/2 # half the chord between two spars
        taper=inputs['taper']
        chords=rootchord*(1-(1-taper)*(1-(np.arange(len(skin))+0.5)/(len(skin))))
        
        t_over_c = inputs['t_over_c']  #VMGM
        hs = inputs['horizontal_shear']  #VMGM

        #b=surface['inter_stringer']
        ##mrho = inputs['mrho']
        ##G = shearMM(mrho,surface['materlist'],surface['puissanceMM'])
        ##E = youngMM(mrho,surface['materlist'],surface['puissanceMM'])

        E = inputs['young'][1] #VMGM
        G = inputs['shear'][1] #VMGM
        v = E/(2*G) - 1  #VMGM
        
        x = chords
        y = t_over_c

        p00 =     -0.8372  
        p10 =      -7.556  
        p01 =       18.11  
        p20 =   1.949e-09  
        p11 =       62.61  
        p02 =      -144.8 
        p30 =  -4.989e-09  
        p21 =   1.451e-09  
        p12 =      -241.8  
        p03 =       539.9  
        p40 =   5.621e-09  
        p31 =   -2.28e-09  
        p22 =   6.385e-10  
        p13 =       437.4  
        p04 =      -947.6  
        p50 =  -2.262e-09 
        p41 =   1.001e-09  
        p32 =   1.128e-10  
        p23 =  -7.291e-10  
        p14 =      -299.5  
        p05 =       631.7 
        r = p00 + p10*x + p01*y + p20*x**2 + p11*x*y + p02*y**2 + p30*x**3 + p21*x**2*y + p12*x*y**2 + p03*y**3 + p40*x**4 + p31*x**3*y + p22*x**2*y**2 + p13*x*y**3 + p04*y**4 + p50*x**5 + p41*x**4*y + p32*x**3*y**2 + p23*x**2*y**3 + p14*x*y**4 + p05*y**5
        
        Z = chords**2*(1 - v**2)**(1/2)/(-r*skin)
        
        """
        k = []
        ks = []
                
        for i, item in enumerate(Z):
            
            if Z[i] < 6.986480987:
                ks.append(5.5)
                k.append(4)
            elif Z[i] >= 6.986480987 and Z[i] < 21.40391211:
                ks.append(2.22368421*Z[i]**0.4658402443)
                k.append(4)
            else:
                ks.append(2.22368421*Z[i]**0.4658402443)
                k.append(Z[i]**0.8889373/3.80772)
        
        ks = np.array(ks, dtype=np.float32)
        k = np.array(k, dtype=np.float32)
        """
        
        k = np.zeros(Z.shape)
        ks = np.zeros(Z.shape)
        
        """ 
        # Considering only straight lines for the k coefficients
        for i in range(Z.shape[0]):
            if Z[i] < 6.986480987:
                ks[i] = 5.5
                k[i] = 4
            elif Z[i] >= 6.986480987 and Z[i] < 21.40391211:
                ks[i] = 2.22368421*Z[i]**0.4658402443
                k[i] = 4
            else:
                ks[i] = 2.22368421*Z[i]**0.4658402443
                k[i] = Z[i]**0.8889373/3.80772
        """
        
        # With a curve bettwen the lines to have a smooth transition
        for i in range(Z.shape[0]):
            if Z[i] < 2:
                ks[i] = 5.5
                k[i] = 4
            elif Z[i] >= 2 and Z[i] < 6:
                ks[i] = 10**(3.1929 - np.sqrt(2.4526**2 - (np.log10(Z[i]) - 0.3010)**2))
                k[i] = 4
            elif Z[i] >= 6 and Z[i] < 22.4051:
                ks[i] = 10**(3.1929 - np.sqrt(2.4526**2 - (np.log10(Z[i]) - 0.3010)**2))
                k[i] = 10**(2.0548 - np.sqrt(1.4527**2 - (np.log10(Z[i]) - 0.7782)**2))
            elif Z[i] >= 22.4051 and Z[i] < 59.6362:
                ks[i] = 2.22368421*Z[i]**0.4658402443
                k[i] = 10**(2.0548 - np.sqrt(1.4527**2 - (np.log10(Z[i]) - 0.7782)**2))
            else:
                ks[i] = 2.22368421*Z[i]**0.4658402443
                k[i] = Z[i]**0.8889373/3.80772
        
        sigmaBuc=k*math.pi**2*skin**2/chords**2*G**2/(3*(4*G-E))
        tauBuc = ks*math.pi**2*skin**2/chords**2*G**2/(3*(4*G-E))
            
        fmax = np.max((hs/tauBuc)**2 + tbc/sigmaBuc - 1)

        nlog, nsum, nexp = np.log, np.sum, np.exp
        k_s = 1 / rho * nlog(nsum(nexp(rho * ((hs/tauBuc)**2 + tbc/sigmaBuc - 1 - fmax))))
        outputs['buckling'] = fmax + k_s 
        
        
#        if math.floor(outputs['buckling'])==-586244:  #TODELETE
#            print('there we are')  #TODELETE        

    """        
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
        partials['buckling', 'shear'] = np.sum(derivShear[0])    #VMGM """
        
    """
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
        
        t_over_c = inputs['t_over_c']  #VMGM
        
        #b=surface['inter_stringer']
        ##mrho = inputs['mrho']
        ##G = shearMM(mrho,surface['materlist'],surface['puissanceMM'])
        ##E = youngMM(mrho,surface['materlist'],surface['puissanceMM'])  
        
        E = inputs['young'] #VMGM
        G = inputs['shear'] #VMGM 
        v = E/(2*G) - 1  #VMGM
        
        x = chords
        y = t_over_c

        p00 =     -0.8372  
        p10 =      -7.556  
        p01 =       18.11  
        p20 =   1.949e-09  
        p11 =       62.61  
        p02 =      -144.8 
        p30 =  -4.989e-09  
        p21 =   1.451e-09  
        p12 =      -241.8  
        p03 =       539.9  
        p40 =   5.621e-09  
        p31 =   -2.28e-09  
        p22 =   6.385e-10  
        p13 =       437.4  
        p04 =      -947.6  
        p50 =  -2.262e-09 
        p41 =   1.001e-09  
        p32 =   1.128e-10  
        p23 =  -7.291e-10  
        p14 =      -299.5  
        p05 =       631.7 
        r = p00 + p10*x + p01*y + p20*x**2 + p11*x*y + p02*y**2 + p30*x**3 + p21*x**2*y + p12*x*y**2 + p03*y**3 + p40*x**4 + p31*x**3*y + p22*x**2*y**2 + p13*x*y**3 + p04*y**4 + p50*x**5 + p41*x**4*y + p32*x**3*y**2 + p23*x**2*y**3 + p14*x*y**4 + p05*y**5
        
        Z = chords**2*(1 - v**2)**(1/2)/(-r*skin)
        k = []
                
        for i, item in enumerate(Z):
        
            if Z[i] < 21.40391211:
                k.append(kc)
            else:
                k.append(Z[i]**0.8889373/3.80772)
        
        k = np.array(k, dtype=np.float32)
        
        sigmaBuc=k*math.pi**2*skin**2/chords**2*G**2/(3*(4*G-E))

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
        
        derivrchords = p10 + 2*p20*x + p11*y + 3*p30*x**2 + 2*p21*x*y + p12*y**2 + 4*p40*x**3 + 3*p31*x**2*y + 2*p22*x*y**2 + p13*y**3 + 5*p50*x**4 + 4*p41*x**3*y + 3*p32*x**2*y**2 + 2*p23*x*y**3 + p14*y**4
        derivrtoverc = p01 + p11*x + 2*p02*y + p21*x**2 + 2*p12*x*y + 3*p03*y**2 + p31*x**3 + 2*p22*x**2*y + 3*p13*x*y**2 + 4*p04*y**3 + p41*x**4 + 2*p32*x**3*y + 3*p23*x**2*y**2 + 4*p14*x*y**3 + 5*p05*y**4

        # Reshape and save them to the jac dict
        partials['buckling', 'top_bending_stress'] = -derivs.reshape(-1)  #  "-" because tbc = - input...   again in next line
        #partials['buckling', 'skin_thickness'] = -partials['buckling', 'top_bending_stress']*(-tbc/sigmaBuc)*(2*(sigmaBuc/skin) - Z/skin*0.8889373/3.80772*Z**(0.8889373-1)*math.pi**2*skin**2/chords**2*G**2/(3*(4*G-E)))
        partials['buckling', 'skin_thickness'] = -partials['buckling', 'top_bending_stress']*(-tbc/sigmaBuc)*(-(0.8889373/3.80772)*Z**(0.8889373)/skin*sigmaBuc/k + 2*sigmaBuc/skin)
        partials['buckling', 't_over_c'] = -partials['buckling', 'top_bending_stress']*(-tbc/sigmaBuc)*Z/(-r*skin)*(derivrtoverc*skin)*0.8889373/3.80772*Z**(0.8889373-1)*math.pi**2*skin**2/chords**2*G**2/(3*(4*G-E))
        derivChord=-partials['buckling', 'top_bending_stress']*(-tbc/sigmaBuc)*((-2)*(sigmaBuc/chords)*(1-(1-taper)*(1-(np.arange(len(skin))+0.5)/(len(skin))))/2 + (((2*chords*(1-v**2)**0.5*(-r*skin)-chords**2*(1-v**2)**0.5*(-derivrchords*skin))/(-r*skin)**2)*(1-(1-taper)*(1-(np.arange(len(skin))+0.5)/(len(skin))))/2*0.8889373/3.80772*Z**(0.8889373-1)*math.pi**2*skin**2/chords**2*G**2/(3*(4*G-E))))
        partials['buckling', 'chord'] = np.sum(derivChord[0])
        derivTaper = derivChord*rootchord*(1-(np.arange(len(skin))+0.5)/(len(skin)))
        partials['buckling', 'taper'] = np.sum(derivTaper[0])
        
        derivYoung = -partials['buckling', 'top_bending_stress']*(-tbc/sigmaBuc)*((sigmaBuc/(4*G-E)) - 1/(2*G)*Z*v/(1-v**2)**0.5*0.8889373/3.80772*Z**(0.8889373-1)*math.pi**2*skin**2/chords**2*G**2/(3*(4*G-E))) #VMGM
        partials['buckling', 'young'] = np.sum(derivYoung[0])    #VMGM
        derivShear = -partials['buckling', 'top_bending_stress']*(-tbc/sigmaBuc)*((2*sigmaBuc/G-12*sigmaBuc/(3*(4*G-E))) - 1/(2*G)*Z*v/(1-v**2)**0.5*0.8889373/3.80772*Z**(0.8889373-1)*math.pi**2*skin**2/chords**2*G**2/(3*(4*G-E))) #VMGM
        partials['buckling', 'shear'] = np.sum(derivShear[0])    #VMGM """