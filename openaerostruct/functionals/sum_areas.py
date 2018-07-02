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


class SumAreas(ExplicitComponent):
    """
    Compute the coefficients of lift (CL) and drag (CD) for the entire aircraft.

    Parameters
    ----------
    S_ref : float
        Surface area for one lifting surface.

    Returns
    -------

    """

    def initialize(self):
        self.options.declare('surfaces', types=list)

    def setup(self):
        for surface in self.options['surfaces']:
            name = surface['name']
            self.add_input(name + '_S_ref', val=1., units='m**2')

        self.add_output('S_ref_total', val=0., units='m**2')

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        outputs['S_ref_total'] = 0.
        for surface in self.options['surfaces']:
            name = surface['name']
            S_ref = inputs[name + '_S_ref']
            outputs['S_ref_total'] += S_ref

    def compute_partials(self, inputs, partials):
        for surface in self.options['surfaces']:
            name = surface['name']
            partials['S_ref_total', name + '_S_ref'] = 1.
