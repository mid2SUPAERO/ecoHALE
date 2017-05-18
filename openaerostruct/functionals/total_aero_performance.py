from __future__ import division, print_function
import numpy as np

from openmdao.api import Group

from openaerostruct.functionals.moment_coefficient import MomentCoefficient
from openaerostruct.functionals.total_lift_drag import TotalLiftDrag

class TotalAeroPerformance(Group):
    """
    Group to contain the total aerodynamic performance components.
    """

    def __init__(self, surfaces, prob_dict):
        super(TotalAeroPerformance, self).__init__()

        self.add_subsystem('moment',
                 MomentCoefficient(surfaces=surfaces, prob_dict=prob_dict),
                 promotes=['*'])
        self.add_subsystem('CL_CD',
                 TotalLiftDrag(surfaces=surfaces, prob_dict=prob_dict),
                 promotes=['*'])
