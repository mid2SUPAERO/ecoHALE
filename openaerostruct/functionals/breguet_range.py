from __future__ import division, print_function
import numpy as np

from openmdao.api import ExplicitComponent

try:
    import OAS_API
    fortran_flag = True
    data_type = float
except:
    fortran_flag = False
    data_type = complex


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
        self.metadata.declare('surfaces', type_=list, required=True)
        self.metadata.declare('prob_dict', type_=dict, required=True)

    def initialize_variables(self):
        for surface in self.metadata['surfaces']:
            name = surface['name']
            self.add_input(name + 'structural_weight', val=1.)

        self.add_input('CL', val=10.)
        self.add_input('CD', val=10.)

        self.add_output('fuelburn', val=1.)
        self.add_output('weighted_obj', val=1.)

    def compute(self, inputs, outputs):
        prob_dict = self.metadata['prob_dict']

        g = prob_dict['g']
        CT = prob_dict['CT']
        a = prob_dict['a']
        R = prob_dict['R']
        M = prob_dict['M']
        W0 = prob_dict['W0'] * g
        beta = prob_dict['beta']

        # Loop through the surfaces and add up the structural weights
        # to get the total structural weight.
        Ws = 0.
        for surface in self.metadata['surfaces']:
            name = surface['name']
            Ws += inputs[name + 'structural_weight']

        CL = inputs['CL']
        CD = inputs['CD']

        fuelburn = (W0 + Ws) * (np.exp(R * CT / a / M * CD / CL) - 1)

        # Convert fuelburn from N to kg
        outputs['fuelburn'] = fuelburn / g

        # This lines makes the 'weight' the total aircraft weight
        outputs['weighted_obj'] = (beta * fuelburn + (1 - beta) * (W0 + Ws + fuelburn)) / g

    def compute_partial_derivs(self, inputs, outputs, partials):
        prob_dict = self.metadata['prob_dict']

        g = prob_dict['g']
        CT = prob_dict['CT']
        a = prob_dict['a']
        R = prob_dict['R']
        M = prob_dict['M']
        W0 = prob_dict['W0'] * g
        beta = prob_dict['beta']

        Ws = 0.
        for surface in self.metadata['surfaces']:
            name = surface['name']
            Ws += inputs[name + 'structural_weight']

        CL = inputs['CL']
        CD = inputs['CD']

        dfb_dCL = -(W0 + Ws) * np.exp(R * CT / a / M * CD / CL) \
            * R * CT / a / M * CD / CL ** 2
        dfb_dCD = (W0 + Ws) * np.exp(R * CT / a / M * CD / CL) \
            * R * CT / a / M / CL
        dfb_dWs = np.exp(R * CT / a / M * CD / CL) - 1

        partials['fuelburn', 'CL'] = dfb_dCL / g
        partials['fuelburn', 'CD'] = dfb_dCD / g
        partials['weighted_obj', 'CL'] = beta * dfb_dCL / g + (1 - beta) * dfb_dCL / g
        partials['weighted_obj', 'CD'] = beta * dfb_dCD / g + (1 - beta) * dfb_dCD / g

        for surface in self.metadata['surfaces']:
            name = surface['name']
            inp_name = name + 'structural_weight'
            partials['fuelburn', inp_name] = dfb_dWs / g
            partials['weighted_obj', inp_name] = beta * dfb_dWs / g + (1 - beta) * dfb_dWs / g \
                + (1 - beta) / g
