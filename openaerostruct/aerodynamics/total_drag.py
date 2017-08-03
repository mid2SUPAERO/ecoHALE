from __future__ import print_function, division

from openmdao.api import ExplicitComponent

class TotalDrag(ExplicitComponent):
    """ Calculate total drag in force units.

    inputeters
    ----------
    CDi : float
        Induced coefficient of drag (CD) for the lifting surface.
    CDv : float
        Calculated coefficient of viscous drag for the lifting surface.

    Returns
    -------
    CD : float
        Total coefficient of drag (CD) for the lifting surface.
    """

    def initialize(self):
        self.metadata.declare('surface', type_=dict)

    def setup(self):
        surface = self.metadata['surface']

        self.add_input('CDi', val=1.)
        self.add_input('CDv', val=1.)
        
        self.add_output('CD', val=1.)

        self.CD0 = surface['CD0']

        self.declare_partials('CD', 'CDi', val=1.)
        self.declare_partials('CD', 'CDv', val=1.)

    def compute(self, inputs, outputs):
        outputs['CD'] = inputs['CDi'] + inputs['CDv'] + self.CD0
