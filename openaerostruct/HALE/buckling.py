# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 17:08:36 2019

@author: e.duriez and Victor M. Guadano
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


    def compute(self, inputs, outputs):
        surface = self.options['surface']
        rho=self.options['rho']
        tbc=-inputs['top_bending_stress'] #"-" to turn compression into positive value
        skin=inputs['skin_thickness']

        # rootchord=inputs['chord']/2 # half the chord between two spars
        rootchord=inputs['chord']/4 # distance between stringers
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
        
        x = 2 * chords # half the chord between two spars
        y = t_over_c
        
        Rcurv = surface['airfoil_radius_curvature']

        p00 = Rcurv[0] 
        p11 = Rcurv[1] 
        p12 = Rcurv[2]  
        p20 = Rcurv[3] 
        p21 = Rcurv[4] 
        p32 = Rcurv[5] 
        p31 = Rcurv[6] 
        p03 = Rcurv[7] 
        p10 = Rcurv[8]   
        p01 = Rcurv[9]      
        p02 = Rcurv[10] 
        p30 = Rcurv[11]       
        p40 = Rcurv[12]  
        p22 = Rcurv[13]  
        p13 = Rcurv[14]  
        p04 = Rcurv[15]  
        p50 = Rcurv[16] 
        p41 = Rcurv[17]    
        p23 = Rcurv[18] 
        p14 = Rcurv[19] 
        p05 = Rcurv[20]
        
        r = p00 + p10*x + p01*y + p20*x**2 + p11*x*y + p02*y**2 + p30*x**3 + p21*x**2*y + p12*x*y**2 + p03*y**3 + p40*x**4 + p31*x**3*y + p22*x**2*y**2 + p13*x*y**3 + p04*y**4 + p50*x**5 + p41*x**4*y + p32*x**3*y**2 + p23*x**2*y**3 + p14*x*y**4 + p05*y**5
        
        Z = chords**2*(1 - v**2)**(1/2)/(-r*skin)
        
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