from __future__ import division, print_function
import numpy as np

from openmdao.api import ExplicitComponent


class BreguetRange(ExplicitComponent):
    """
    Computes the fuel burn using the Breguet range equation using
    the computed CL, CD, weight, and provided specific fuel consumption, speed of sound,
    Mach number, initial weight, and range.

    Note that we add information from each lifting surface.

    Parameters
    ----------
    CL : float
        Total coefficient of lift (CL) for the lifting surface.
    CD : float
        Total coefficient of drag (CD) for the lifting surface.
    weight : float
        Total weight of the structural spar.

    Returns
    -------
    fuelburn : float
        Computed fuel burn in kg based on the Breguet range equation.

    """

    def initialize(self):
        self.options.declare('surfaces', types=list)

    def setup(self):
        for surface in self.options['surfaces']:
            name = surface['name']
            self.add_input(name + '_structural_weight', val=1., units='N')

        self.add_input('CT', val=0.25, units='1/s')
        self.add_input('CL', val=0.7)
        self.add_input('CD', val=0.02)
        self.add_input('a', val=100., units='m/s')
        self.add_input('R', val=3000., units='m')
        self.add_input('M', val=1.2)
        self.add_input('W0', val=200., units='kg')
        self.add_input('load_factor', val=1.05)

        self.add_output('fuelburn', val=1., units='kg')

        self.declare_partials('*', '*')
        self.set_check_partial_options(wrt='*', method='cs', step=1e-30)

    def compute(self, inputs, outputs):
        print('BR load fac', inputs['load_factor'])

        g = 9.80665 * inputs['load_factor']
        CT = inputs['CT']
        a = inputs['a']
        R = inputs['R']
        M = inputs['M']
        W0 = inputs['W0'] * g

        # Loop through the surfaces and add up the structural weights
        # to get the total structural weight.
        Ws = 0.
        for surface in self.options['surfaces']:
            name = surface['name']
            Ws += inputs[name + '_structural_weight']

        CL = inputs['CL']
        CD = inputs['CD']

        fuelburn = (W0 + Ws * inputs['load_factor']) * (np.exp(R * CT / a / M * CD / CL) - 1)

        # Convert fuelburn from N to kg
        outputs['fuelburn'] = fuelburn / 9.80665

    def compute_partials(self, inputs, partials):

        g = 9.80665 * inputs['load_factor']
        CT = inputs['CT']
        a = inputs['a']
        R = inputs['R']
        M = inputs['M']
        W0 = inputs['W0'] * g

        Ws = 0.
        for surface in self.options['surfaces']:
            name = surface['name']
            Ws += inputs[name + '_structural_weight']

        CL = inputs['CL']
        CD = inputs['CD']

        dfb_dCL = -(W0 + Ws * inputs['load_factor']) * np.exp(R * CT / a / M * CD / CL) \
            * R * CT / a / M * CD / CL ** 2
        dfb_dCD = (W0 + Ws * inputs['load_factor']) * np.exp(R * CT / a / M * CD / CL) \
            * R * CT / a / M / CL
        dfb_dCT = (W0 + Ws * inputs['load_factor']) * np.exp(R * CT / a / M * CD / CL) \
            * R / a / M / CL * CD
        dfb_dR = (W0 + Ws * inputs['load_factor']) * np.exp(R * CT / a / M * CD / CL) \
            / a / M / CL * CD * CT
        dfb_da = -(W0 + Ws * inputs['load_factor']) * np.exp(R * CT / a / M * CD / CL) \
            * R * CT / a**2 / M * CD / CL
        dfb_dM = -(W0 + Ws * inputs['load_factor']) * np.exp(R * CT / a / M * CD / CL) \
            * R * CT / a / M**2 * CD / CL

        dfb_dW = np.exp(R * CT / a / M * CD / CL) - 1

        partials['fuelburn', 'CL'] = dfb_dCL / 9.80665
        partials['fuelburn', 'CD'] = dfb_dCD / 9.80665
        partials['fuelburn', 'CT'] = dfb_dCT / 9.80665
        partials['fuelburn', 'a'] = dfb_da / 9.80665
        partials['fuelburn', 'R'] = dfb_dR / 9.80665
        partials['fuelburn', 'M'] = dfb_dM / 9.80665
        partials['fuelburn', 'W0'] = dfb_dW * inputs['load_factor']
        partials['fuelburn', 'load_factor'] = (W0 / inputs['load_factor'] + Ws) * dfb_dW / 9.80665

        for surface in self.options['surfaces']:
            name = surface['name']
            inp_name = name + '_structural_weight'
            partials['fuelburn', inp_name] = dfb_dW / 9.80665 * inputs['load_factor']
