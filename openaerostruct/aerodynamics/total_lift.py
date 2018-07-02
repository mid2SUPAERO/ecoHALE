from __future__ import print_function, division

from openmdao.api import ExplicitComponent

class TotalLift(ExplicitComponent):
    """
    Calculate total lift in force units.

    Parameters
    ----------
    CL1 : float
        Induced coefficient of lift (CL) for the lifting surface.

    Returns
    -------
    CL : float
        Total coefficient of lift (CL) for the lifting surface.
    """


    def initialize(self):
        self.options.declare('surface', types=dict)

    def setup(self):
        surface = self.options['surface']

        self.add_input('CL1', val=1.)

        self.add_output('CL', val=1.)

        self.CL0 = surface['CL0']

        self.declare_partials('CL', 'CL1', val=1.)

    def compute(self, inputs, outputs):
        outputs['CL'] = inputs['CL1'] + self.CL0
