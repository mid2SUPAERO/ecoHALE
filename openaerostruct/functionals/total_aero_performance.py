from __future__ import division, print_function
import numpy as np

from openmdao.api import Group

from openaerostruct.functionals.moment_coefficient import MomentCoefficient
from openaerostruct.functionals.total_lift_drag import TotalLiftDrag

class TotalAeroPerformance(Group):
    """
    Group to contain the total aerodynamic performance components.
    """

    def initialize(self):
        self.metadata.declare('surfaces', type_=list, required=True)
        self.metadata.declare('prob_dict', type_=dict, required=True)

    def setup(self):
        prob_dict = self.metadata['prob_dict']
        surfaces = self.metadata['surfaces']

        self.add_subsystem('CL_CD',
                 TotalLiftDrag(surfaces=surfaces, prob_dict=prob_dict),
                 promotes_inputs=['*CL', '*CD', '*S_ref'],
                 promotes_outputs=['CL', 'CD'])

        self.add_subsystem('moment',
                 MomentCoefficient(surfaces=surfaces, prob_dict=prob_dict),
                 promotes_inputs=['v', 'alpha', 'cg', '*b_pts', '*widths', '*chords', '*sec_forces'],
                 promotes_outputs=['CM'])
