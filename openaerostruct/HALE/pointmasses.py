# -*- coding: utf-8 -*-
"""
Created on Fri May 15 14:42:36 2020

@author: Victor M. Guadano
"""

from __future__ import division, print_function
from openmdao.api import ExplicitComponent
import numpy as np

class PointMassLocations(ExplicitComponent):

    def initialize(self):
        self.options.declare('surface', types=dict)

    def setup(self):
        surface = self.options['surface']    
        self.ny = surface['mesh'].shape[1]
        self.n_point_masses = surface['n_point_masses']     
        
        self.add_input('span', val=50, units='m')
        self.add_input('engine_location', val=-0.2)
        self.add_input('nodes', val=np.zeros((self.ny, 3)), units='m')
        
        self.add_output('point_mass_locations', val=np.zeros((self.n_point_masses, 3)), units='m')

        self.declare_partials('point_mass_locations', 'span', method='cs')
        self.declare_partials('point_mass_locations', 'engine_location', method='cs')
        self.declare_partials('point_mass_locations', 'nodes', method='cs')

    def compute(self, inputs, outputs):
        span = inputs['span']
        engine_location = inputs['engine_location']
        nodes = inputs['nodes']
        point_mass_locations = outputs['point_mass_locations'] 
        
        point_mass_locations[0][1] = engine_location*span/2
        point_mass_locations[0][0] = (point_mass_locations[0][1]-nodes[-1][1])/(nodes[0][1]-nodes[-1][1])*(nodes[0][0]-nodes[-1][0]) + nodes[-1][0]
        point_mass_locations[0][2] = (point_mass_locations[0][1]-nodes[-1][1])/(nodes[0][1]-nodes[-1][1])*(nodes[0][2]-nodes[-1][2]) + nodes[-1][2]  

    ##def compute_partials(self, inputs, partials):
        ##span = inputs['span']
        ##engines_location = inputs['engine_location']
        
        ##partials['point_mass_locations', 'span'] = [0,engines_location,0]
        ##partials['point_mass_locations', 'engine_location'] = [0,span,0]
    
        
class PointMasses(ExplicitComponent):

    def initialize(self):
        self.options.declare('surface', types=dict)

    def setup(self):
        surface = self.options['surface']        
        self.n_point_masses = surface['n_point_masses']
        
        self.add_input('PV_surface', val=30., units='m**2')       
        
        self.add_output('point_masses', val=np.zeros((self.n_point_masses)), units='kg')

        self.declare_partials('point_masses', 'PV_surface')

    def compute(self, inputs, outputs):
        surface = self.options['surface']
        PVsurf = inputs['PV_surface']
        payload_power = surface['payload_power']
        productivityPV = surface['productivityPV']
        prop_density = surface['prop_density']
        n_point_masses = surface['n_point_masses']
        produced_power = PVsurf*productivityPV
        prop_power = produced_power-payload_power
        prop_mass = prop_power*prop_density
        outputs['point_masses'] = prop_mass/(2*n_point_masses)
        
    def compute_partials(self, inputs, partials):
        surface = self.options['surface']
        productivityPV = surface['productivityPV']
        prop_density = surface['prop_density']
        n_point_masses = surface['n_point_masses']
        
        partials['point_masses', 'PV_surface'] = productivityPV*prop_density/(2*n_point_masses)