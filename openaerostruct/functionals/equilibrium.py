from __future__ import division, print_function
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
        self.options.declare('surfaces', types=list)

    def setup(self):
        for surface in self.options['surfaces']:
            name = surface['name']
            self.add_input(name + '_structural_weight', val=1., units='N')

        self.add_input('fuelburn', val=1., units='kg')
        self.add_input('W0', val=1., units='kg')
        self.add_input('load_factor', val=1.)

        self.add_input('CL', val=1.)

        self.add_input('S_ref_total', val=1., units='m**2')
        self.add_input('v', val=1., units='m/s')
        self.add_input('rho', val=1., units='kg/m**3')

        self.add_output('L_equals_W', val=1.)
        self.add_output('total_weight', val=1., units='N')

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):

        g = 9.80665 * inputs['load_factor']
        W0 = inputs['W0'] * g
        rho = inputs['rho']
        v = inputs['v']

        structural_weight = 0.
        symmetry = False
        for surface in self.options['surfaces']:
            name = surface['name']
            structural_weight += inputs[name + '_structural_weight']
            if surface['symmetry']:
                symmetry = True

        S_ref_tot = inputs['S_ref_total']

        tot_weight = structural_weight + inputs['fuelburn'] * g + W0

        outputs['total_weight'] = tot_weight
        outputs['L_equals_W'] = 1 - (0.5 * rho * v**2 * S_ref_tot) * inputs['CL'] / tot_weight

    def compute_partials(self, inputs, partials):
        g = 9.80665 * inputs['load_factor']
        W0 = inputs['W0'] * g
        rho = inputs['rho']
        v = inputs['v']

        structural_weight = 0.
        for surface in self.options['surfaces']:
            name = surface['name']
            structural_weight += inputs[name + '_structural_weight']

        S_ref_tot = inputs['S_ref_total']

        tot_weight = structural_weight + inputs['fuelburn'] * g + W0

        L = inputs['CL'] * (0.5 * rho * v**2 * S_ref_tot)

        partials['total_weight', 'fuelburn'] = g
        partials['total_weight', 'W0'] = g
        partials['total_weight', 'load_factor'] = (inputs['fuelburn'] + inputs['W0']) * 9.80665

        partials['L_equals_W', 'fuelburn'] = L / tot_weight**2 * g
        partials['L_equals_W', 'W0'] = L / tot_weight**2 * g
        partials['L_equals_W', 'load_factor'] = L / tot_weight**2 * (inputs['fuelburn'] * 9.80665 + inputs['W0'] * 9.80665)
        partials['L_equals_W', 'rho'] = -.5 * S_ref_tot * v**2 * inputs['CL'] / tot_weight
        partials['L_equals_W', 'v'] = - rho * S_ref_tot * v * inputs['CL'] / tot_weight
        partials['L_equals_W', 'CL'] = - .5 * rho * v**2 * S_ref_tot / tot_weight
        partials['L_equals_W', 'S_ref_total'] = - .5 * rho * v**2 * inputs['CL'] / tot_weight

        for surface in self.options['surfaces']:
            name = surface['name']
            partials['total_weight', name + '_structural_weight'] = 1.0
            partials['L_equals_W', name + '_structural_weight'] = L / tot_weight**2
