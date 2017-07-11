from __future__ import division, print_function
import numpy as np

from openmdao.api import ExplicitComponent

try:
    from openaerostruct.fortran import OAS_API
    fortran_flag = True
    data_type = float
except:
    fortran_flag = False
    data_type = complex


class Equilibrium(ExplicitComponent):
    """
    Lift = weight constraint.
    Note that we add information from each lifting surface.
    Parameters
    ----------
    L : float
        Total lift for the lifting surface.
    structural_weight : float
        Total weight of the structural spar.
    fuelburn : float
        Computed fuel burn in kg based on the Breguet range equation.
    Returns
    -------
    L_equals_W : float
        Equality constraint for lift = total weight. L_equals_W = 0 for the constraint to be satisfied.
    total_weight : float
        Total weight of the entire aircraft, including W0, all structural weights,
        and fuel.

    """

    def initialize(self):
        self.metadata.declare('surfaces', type_=list, required=True)
        self.metadata.declare('prob_dict', type_=dict, required=True)

    def setup(self):
        for surface in self.metadata['surfaces']:
            name = surface['name']
            self.add_input(name + 'L', val=1.)
            self.add_input(name + 'structural_weight', val=1.)

        self.add_input('fuelburn', val=1.)
        self.add_input('W0', val=1.)

        self.add_output('L_equals_W', val=1.)
        self.add_output('total_weight', val=1.)

    def compute(self, inputs, outputs):
        prob_dict = self.metadata['prob_dict']

        g = prob_dict['g']
        W0 = inputs['W0'] * g

        structural_weight = 0.
        L = 0.
        for surface in self.metadata['surfaces']:
            name = surface['name']
            structural_weight += inputs[name + 'structural_weight']
            L += inputs[name + 'L']

        tot_weight = structural_weight + inputs['fuelburn'] * g + W0

        outputs['total_weight'] = tot_weight
        outputs['L_equals_W'] = 1 - L / tot_weight


        for surface in self.metadata['surfaces']:
            name = surface['name']
            self.declare_partials('total_weight', name + 'L', dependent=False)
            self.declare_partials('L_equals_W', name + 'structural_weight', dependent=False)

    def compute_partials(self, inputs, outputs, partials):
        prob_dict = self.metadata['prob_dict']

        g = prob_dict['g']
        W0 = inputs['W0'] * g

        structural_weight = 0.
        L = 0.
        for surface in self.metadata['surfaces']:
            name = surface['name']
            structural_weight += inputs[name + 'structural_weight']
            L += inputs[name + 'L']

        tot_weight = structural_weight + inputs['fuelburn'] * g + W0

        partials['total_weight', 'fuelburn'] = g
        partials['L_equals_W', 'fuelburn'] = L / tot_weight ** 2 * g
        for surface in self.metadata['surfaces']:
            name = surface['name']
            partials['total_weight', name + 'structural_weight'] = 1.0
            partials['L_equals_W', name + 'L'] = -1.0 / tot_weight
            partials['L_equals_W', name + 'structural_weight'] = L / tot_weight**2
