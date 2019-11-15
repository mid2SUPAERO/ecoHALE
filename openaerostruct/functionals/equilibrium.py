from __future__ import division, print_function
from openmdao.api import ExplicitComponent
from openaerostruct.utils.constants import grav_constant
import math


class Equilibrium(ExplicitComponent):
    """
    Computes L_equals_W, which is a normalized measure of the weight of the
    aircraft minus the total generated lift. So if L_equals_W is positive,
    the aircraft weighs more than the amount of lift it's producing.

    Generally we use this measure to ensure that lift equals weight for a cruise
    fligth condition.
    Note that we add information from each lifting surface.

    Parameters
    ----------
    structural_mass : float
        Total weight of the structural spar for a given surface.

    fuelburn : float
        Computed fuel burn in kg based on the Breguet range equation.
    W0 : float
        The operating empty weight of the aircraft, without fuel or structural
        mass. Supplied in kg despite being a 'weight' due to convention.
    load_factor : float
        Multiplicative factor on gravity. 1.0 is normal flight; 2.5 would be
        for a 2.5g manuever.
    CL : float
        Total coefficient of lift (CL) for the entire aircraft.
    S_ref_total : float
        Total surface area of the aircraft based on the sum of individual
        surface areas.
    v : float
        Freestream air velocity in m/s.
    rho : float
        Air density in kg/m^3.

    Returns
    -------
    L_equals_W : float
        Equality constraint for lift = total weight. L_equals_W = 0 for the
        constraint to be satisfied.
    total_weight : float
        Total weight of the entire aircraft, including W0, all structural
        weights, and fuel.
    """

    def initialize(self):
        self.options.declare('surfaces', types=list)

    def setup(self):
        for surface in self.options['surfaces']:
            name = surface['name']
            self.add_input(name + '_structural_mass', val=1., units='kg')
            self.declare_partials('L_equals_W', name+'_structural_mass')
            self.declare_partials('total_weight', name+'_structural_mass')

#        self.add_input('fuelburn', val=123., units='kg')
        self.add_input('W0', val=10., units='kg')
        self.add_input('load_factor', val=1.05)

        self.add_input('CL', val=1)

        self.add_input('S_ref_total', val=15., units='m**2')
        self.add_input('v', val=100., units='m/s')
        self.add_input('rho', val=1.2, units='kg/m**3')
		
        self.add_input('PV_surface', val=30., units='m**2')

        self.add_output('L_equals_W', val=1.)
        self.add_output('total_weight', val=1., units='N')
        self.add_output('PV_mass', val=1., units='kg')

        self.declare_partials('L_equals_W','CL')
        self.declare_partials('L_equals_W','S_ref_total')
        self.declare_partials('L_equals_W','W0')
#        self.declare_partials('L_equals_W','fuelburn')
        self.declare_partials('L_equals_W','load_factor')
        self.declare_partials('L_equals_W','rho')
        self.declare_partials('L_equals_W','v')
        self.declare_partials('L_equals_W','PV_surface')
        self.declare_partials('PV_mass','PV_surface')
        self.declare_partials('total_weight','W0')
#        self.declare_partials('total_weight','fuelburn')
        self.declare_partials('total_weight','load_factor')
        self.declare_partials('total_weight','PV_surface')

    def compute(self, inputs, outputs):

        g = grav_constant * inputs['load_factor']
        W0 = inputs['W0']
        rho = inputs['rho']
        v = inputs['v']

        PVsurf = inputs['PV_surface']

        structural_mass = 0.
        PVdensity = 0.
        if len(self.options['surfaces']) > 1:
            print('MODIFY CODE TO TAKE INTO ACCOUNT MULTIPLE SURFACE')
        for surface in self.options['surfaces']:
            name = surface['name']
            structural_mass += inputs[name + '_structural_mass']
            PVdensity = surface['densityPV']  #works only if there is only one surface... otherwise, some sort of mean depending on each surface's surface could be made. Same modification has to be made in partials.
            prop_density = surface['prop_density']
            mppt_density = surface['mppt_density']
            payload_power = surface['payload_power']
            productivityPV = surface['productivityPV']

        PVmass=PVsurf*PVdensity
        S_ref_tot = inputs['S_ref_total']
        produced_power = PVsurf*productivityPV
        mppt_mass = produced_power*mppt_density
        prop_power = produced_power-payload_power
        prop_mass = prop_power*prop_density
		

#        tot_weight = (structural_mass + inputs['fuelburn'] + W0) * g
        tot_weight = (structural_mass + W0 + PVmass + mppt_mass + prop_mass) * g
        if math.floor(100000*structural_mass)==74729:  #TODELETE
            print('there we are')  #TODELETE      
        outputs['PV_mass'] = PVmass
        outputs['total_weight'] = tot_weight
        outputs['L_equals_W'] = 1 - (0.5 * rho * v**2 * S_ref_tot) * inputs['CL'] / tot_weight
        


    def compute_partials(self, inputs, partials):
        g = grav_constant * inputs['load_factor']
        W0 = inputs['W0']
        rho = inputs['rho']
        v = inputs['v']
        PVsurf = inputs['PV_surface']

        structural_mass = 0.
        PVdensity=0.
        for surface in self.options['surfaces']:
            name = surface['name']
            structural_mass += inputs[name + '_structural_mass']
            PVdensity = surface['densityPV']
            prop_density = surface['prop_density']
            mppt_density = surface['mppt_density']
            payload_power = surface['payload_power']
            productivityPV = surface['productivityPV']

        partials['PV_mass','PV_surface'] = PVdensity
        S_ref_tot = inputs['S_ref_total']
        produced_power = PVsurf*productivityPV
        mppt_mass = produced_power*mppt_density
        prop_power = produced_power-payload_power
        prop_mass = prop_power*prop_density
        PVmass = PVsurf*PVdensity

#        tot_weight = (structural_mass + inputs['fuelburn'] + W0) * g
        tot_weight = (structural_mass + W0 + PVsurf*PVdensity + mppt_mass + prop_mass) * g


        L = inputs['CL'] * (0.5 * rho * v**2 * S_ref_tot)

#        partials['total_weight', 'fuelburn'] = g
        partials['total_weight', 'W0'] = g
#        partials['total_weight', 'load_factor'] = (inputs['fuelburn'] + \
#            inputs['W0'] + structural_mass) * grav_constant
        partials['total_weight', 'load_factor'] = ( \
            inputs['W0'] + structural_mass + PVmass + mppt_mass + prop_mass) * grav_constant

        partials['total_weight', 'PV_surface'] = g*(PVdensity + mppt_density*productivityPV + productivityPV*prop_density)

#        partials['L_equals_W', 'fuelburn'] = L / tot_weight**2 * g
        partials['L_equals_W', 'W0'] = L / tot_weight**2 * g
        partials['L_equals_W', 'PV_surface'] = L / tot_weight**2 * g * (PVdensity + mppt_density*productivityPV + productivityPV*prop_density)
#        partials['L_equals_W', 'load_factor'] = L / tot_weight**2 * (inputs['fuelburn'] + \
#            inputs['W0'] + structural_mass) * grav_constant
        partials['L_equals_W', 'load_factor'] = L / tot_weight**2 * ( \
            inputs['W0'] + structural_mass + PVsurf*PVdensity + mppt_mass + prop_mass) * grav_constant
        partials['L_equals_W', 'rho'] = -.5 * S_ref_tot * v**2 * inputs['CL'] / tot_weight
        partials['L_equals_W', 'v'] = - rho * S_ref_tot * v * inputs['CL'] / tot_weight
        partials['L_equals_W', 'CL'] = - .5 * rho * v**2 * S_ref_tot / tot_weight
        partials['L_equals_W', 'S_ref_total'] = - .5 * rho * v**2 * inputs['CL'] / tot_weight

        for surface in self.options['surfaces']:
            name = surface['name']
            partials['total_weight', name + '_structural_mass'] = 1.0 * g
            partials['L_equals_W', name + '_structural_mass'] = L / tot_weight**2 * g
