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
        self.n_point_masses = surface['n_point_masses']     
        
        self.add_input('span', val=50, units='m')
        self.add_input('point_mass_locations_span', val=np.zeros((self.n_point_masses, 3)))
        self.add_output('nodes', val=np.zeros((ny, 3)), units='m')
        
        self.add_output('point_mass_locations', val=np.zeros((self.n_point_masses, 3)), units='m')

        self.declare_partials('point_mass_locations', 'span')
        self.declare_partials('point_mass_locations', 'point_mass_locations_span')
        self.declare_partials('point_mass_locations', 'nodes')

    def compute(self, inputs, outputs):
        span = inputs['span']
        point_mass_locations_span = inputs['point_mass_locations_span']
        nodes = inputs['nodes']
        outputs['point_mass_locations'] = point_mass_locations
        
        point_mass_locations[0] = (point_mass_locations[1]-nodes[1][0])/(nodes[1][-1]-nodes[1][0])*(nodes[0][-1]-nodes[0][0]) + nodes[0][0]
        point_mass_locations[2] = (point_mass_locations[1]-nodes[1][0])/(nodes[1][-1]-nodes[1][0])*(nodes[2][-1]-nodes[2][0]) + nodes[2][0]

        outputs['point_mass_locations'] = point_mass_locations_span*span    
        
    def compute_partials(self, inputs, partials):
        span = inputs['span']
        point_mass_locations_span = inputs['point_mass_locations_span']
        
        partials['point_mass_locations', 'span'] = point_mass_locations_span
        partials['point_mass_locations', 'point_mass_locations_span'] = span
        
        
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