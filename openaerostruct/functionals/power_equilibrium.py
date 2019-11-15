# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 17:08:36 2019

@author: e.duriez
"""

from __future__ import division, print_function
from openmdao.api import ExplicitComponent
from openaerostruct.utils.constants import grav_constant
import math


class PowerEquilibrium(ExplicitComponent):
    """
    Computes enough_power, which is a normalized measure of the power needed be the electric
    aircraft minus the total surface. So if enough_power is positive,
    the aircraft needs more power than the amount of power its wing's surface can produce.

    Parameters
    ----------
    total_weight : float
        Total weight of the entire aircraft, including W0, all structural
        weights (and weight of PV and baterry : TODO).
    speed_of_sound : float
        The Mach speed, speed of sound, at the specified flight condition.
    Mach_number : float
        The Mach number of the aircraft at the specified flight condition.
    load_factor : float
        Multiplicative factor on gravity. 1.0 is normal flight; 2.5 would be
        for a 2.5g manuever.
    CL : float
        Total coefficient of lift (CL) for the entire aircraft.
    CD : float
        Total coefficient of drag (CD) for the entire aircraft.
    S_ref_total : float
        Total surface area of the aircraft based on the sum of individual
        surface areas.

    Returns
    -------
    enough_pwer : float
        Equality constraint for power needed = power produced. enough_power < 0 for the
        constraint to be satisfied.
    PV_surface : float
        Surface that is needed to be covered in photovoltaic cells
    """

    def initialize(self):
        self.options.declare('surfaces', types=list)

    def setup(self):

        self.add_input('CL', val=1)
        self.add_input('S_ref_total', val=15., units='m**2')
        self.add_input('CD', val=0.02)
        self.add_input('speed_of_sound', val=100., units='m/s')
        self.add_input('Mach_number', val=0.3)
        self.add_input('total_weight', val=1., units='N')

        self.add_output('enough_power', val=1.)
        self.add_output('PV_surface', val=30., units='m**2')

        self.declare_partials('enough_power','CL')
        self.declare_partials('enough_power','CD')
        self.declare_partials('enough_power','speed_of_sound')
        self.declare_partials('enough_power','Mach_number')
        self.declare_partials('enough_power','S_ref_total')
        self.declare_partials('enough_power','total_weight')
        self.declare_partials('PV_surface','CL')
        self.declare_partials('PV_surface','CD')
        self.declare_partials('PV_surface','speed_of_sound')
        self.declare_partials('PV_surface','Mach_number')
        self.declare_partials('PV_surface','S_ref_total')
        self.declare_partials('PV_surface','total_weight')

    def compute(self, inputs, outputs):

        CLrel=inputs['CL']
        if CLrel>=0:
            CLabs=CLrel
        else:
            CLabs=-CLrel
        CDrel = inputs['CD']
        if CDrel>=0:
            CDabs=CDrel
        else:
            CDabs=-CDrel
        speed_of_sound = inputs['speed_of_sound']
        Mach_number = inputs['Mach_number']
        S_ref_total = inputs['S_ref_total']
        total_weight = inputs['total_weight']

        #get the efficiency of the solar cells (only works if all surfaces have the same solar cells)
        for surface in self.options['surfaces']:
            productivityPV = surface['productivityPV']
            payloadPower= surface['payload_power']
            mp_efficiency = surface['motor_propeller_efficiency']
        
        speed=Mach_number*speed_of_sound
        thrust=total_weight*CDabs/CLabs
        needed_power=speed*thrust/mp_efficiency + payloadPower
        
        possibly_available_power=productivityPV*S_ref_total
        
        #get the surface of solar cells needed (this surface can be higher than the actual available surface)
        PVsurf=needed_power/productivityPV
#        if possibly_available_power > needed_power:
#            PVsurf=needed_power/productivityPV
#        else:
#            PVsurf=S_ref_total

        outputs['enough_power'] = 1 - possibly_available_power / needed_power
        outputs['PV_surface'] = PVsurf
        
#        if math.floor(10000*outputs['enough_power'])==-27428:  #TODELETE
#            print('there we are')  #TODELETE        
        
    def compute_partials(self, inputs, partials):
        CLrel=inputs['CL']
        if CLrel>=0:
            CLabs=CLrel
        else:
            CLabs=-CLrel
        CDrel = inputs['CD']
        if CDrel>=0:
            CDabs=CDrel
        else:
            CDabs=-CDrel

        speed_of_sound = inputs['speed_of_sound']
        Mach_number = inputs['Mach_number']
        S_ref_total = inputs['S_ref_total']
        total_weight = inputs['total_weight']

        #get the efficiency of the solar cells (only works if all surfaces have the same solar cells)
        for surface in self.options['surfaces']:
            productivityPV = surface['productivityPV']
            payloadPower= surface['payload_power']
            mp_efficiency = surface['motor_propeller_efficiency']

        speed=Mach_number*speed_of_sound
        thrust=total_weight*CDabs/CLabs
        needed_power=speed*thrust/mp_efficiency + payloadPower
        
        possibly_available_power=productivityPV*S_ref_total
#        PVsurf=Mach_number*speed_of_sound*total_weight*CD/CL/productivityPV
#        enoughPower=1-productivityPV*S_ref_total/(Mach_number*speed_of_sound*total_weight*CD/CL+payloadPower)

        
        #here, we compute the partial derivatives of needed_power
        if CLrel>=0:
            d_neededpower_d_CL=-speed/mp_efficiency*total_weight*CDabs/CLabs**2
        else:
            d_neededpower_d_CL=speed/mp_efficiency*total_weight*CDabs/CLabs**2
        if CDrel>=0:
            d_neededpower_d_CD=speed/mp_efficiency*total_weight/CLabs
        else:
            d_neededpower_d_CD=-speed/mp_efficiency*total_weight/CLabs
        d_neededpower_d_speedsound=Mach_number*thrust/mp_efficiency
        d_neededpower_d_mach=speed_of_sound*thrust/mp_efficiency
        d_neededpower_d_totweight=speed/mp_efficiency*CDabs/CLabs

        partials['enough_power', 'CL'] = possibly_available_power*d_neededpower_d_CL/needed_power**2
        partials['enough_power', 'CD'] = possibly_available_power*d_neededpower_d_CD/needed_power**2
        partials['enough_power', 'speed_of_sound'] = possibly_available_power*d_neededpower_d_speedsound/needed_power**2
        partials['enough_power', 'Mach_number'] = possibly_available_power*d_neededpower_d_mach/needed_power**2
        partials['enough_power', 'S_ref_total'] = -productivityPV/needed_power
        partials['enough_power', 'total_weight'] = possibly_available_power*d_neededpower_d_totweight/needed_power**2
        
        partials['PV_surface', 'CL'] = d_neededpower_d_CL/productivityPV
        partials['PV_surface', 'CD'] = d_neededpower_d_CD/productivityPV
        partials['PV_surface', 'speed_of_sound'] = d_neededpower_d_speedsound/productivityPV
        partials['PV_surface', 'Mach_number'] = d_neededpower_d_mach/productivityPV
        partials['PV_surface', 'total_weight'] = d_neededpower_d_totweight/productivityPV

        
