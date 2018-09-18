from __future__ import division, print_function
import numpy as np

from openmdao.api import ExplicitComponent


class CenterOfGravity(ExplicitComponent):
    """
    Compute the center of gravity of the entire aircraft based on the inputted W0
    and its corresponding cg and the weighted sum of each surface's structural
    weight and location.

    Note that we add information from each lifting surface.

    Parameters
    ----------
    structural_weight : float
        Total weight of the structural spar for a given surface.
    cg_location[3] : numpy array
        Location of the structural spar's cg for a given surface.

    total_weight : float
        Total weight of the entire aircraft, including W0, all structural weights,
        and fuel.
    fuelburn : float
        Computed fuel burn in kg based on the Breguet range equation.

    Returns
    -------
    cg[3] : numpy array
        The x, y, z coordinates of the center of gravity for the entire aircraft.

    """

    def initialize(self):
        self.options.declare('surfaces', types=list)

    def setup(self):

        arange = np.arange(3)
        for surface in self.options['surfaces']:
            name = surface['name']
            self.add_input(name + '_structural_weight', val=1., units='N')
            self.add_input(name + '_cg_location', val=np.ones(3), units='m')

            self.declare_partials('cg', name + '_cg_location', rows=arange, cols=arange)
            self.declare_partials('cg', name + '_structural_weight')

        self.add_input('total_weight', val=1000., units='N')
        self.add_input('fuelburn', val=1.5, units='kg')
        self.add_input('W0', val=123., units='kg')
        self.add_input('load_factor', val=1.05)
        self.add_input('empty_cg', val=np.ones((3)), units='m')
        self.add_output('cg', val=np.ones(3), units='m')

        self.declare_partials('cg', 'total_weight')
        self.declare_partials('cg', 'W0')
        self.declare_partials('cg', 'fuelburn')
        self.declare_partials('cg', 'load_factor')
        self.declare_partials('cg', 'empty_cg', rows=arange, cols=arange)

    def compute(self, inputs, outputs):

        g = 9.80665 * inputs['load_factor']
        W0 = inputs['W0']
        cg = inputs['empty_cg']
        W0_cg = W0 * cg * g

        spar_cg = np.zeros(3)

        # Loop through the surfaces and compute the weighted cg location
        # of all structural spars
        for surface in self.options['surfaces']:
            name = surface['name']
            spar_cg = spar_cg + inputs[name + '_cg_location'] * inputs[name + '_structural_weight']

        # Compute the total cg of the aircraft based on the empty weight cg and
        # the structures cg. Here we assume the fuel weight is at the cg.
        outputs['cg'] = (W0_cg + spar_cg * inputs['load_factor']) / (inputs['total_weight'] - inputs['fuelburn'] * g)

    def compute_partials(self, inputs, partials):

        g = 9.80665 * inputs['load_factor']
        W0 = inputs['W0']
        cg = inputs['empty_cg']
        fb = inputs['fuelburn']
        tw = inputs['total_weight']
        W0_cg = W0 * cg * g

        spar_cg = np.zeros(3)

        # Loop through the surfaces and compute the weighted cg location
        # of all structural spars
        for surface in self.options['surfaces']:
            name = surface['name']
            spar_cg = spar_cg + inputs[name + '_cg_location'] * inputs[name + '_structural_weight']

        partials['cg', 'total_weight'] = -(W0_cg + spar_cg * inputs['load_factor']) / (tw - fb * g) ** 2
        partials['cg', 'fuelburn'] = g * (W0_cg + spar_cg* inputs['load_factor']) / (tw - fb * g) ** 2
        partials['cg', 'load_factor'] =  (9.80665 * W0 * cg + spar_cg) / (tw - fb * g) + 9.80665 * (W0 * cg * g + spar_cg * inputs['load_factor']) / (tw - fb * g)**2 * (fb)
        partials['cg', 'empty_cg'] = W0 * g / (tw - fb * g)

        partials['cg', 'W0'] = cg * g  / (inputs['total_weight'] - inputs['fuelburn'] * g)

        for surface in self.options['surfaces']:
            name = surface['name']
            partials['cg', name + '_cg_location'] = inputs[name + '_structural_weight'] \
                / (tw - fb * g) * inputs['load_factor']
            partials['cg', name + '_structural_weight'] = inputs[name + '_cg_location'] \
                / (tw - fb * g) * inputs['load_factor']
