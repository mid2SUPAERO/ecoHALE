from __future__ import division, print_function
import numpy as np

from openmdao.api import Group

from openaerostruct.functionals.breguet_range import BreguetRange
from openaerostruct.functionals.equilibrium import Equilibrium
from openaerostruct.functionals.center_of_gravity import CenterOfGravity
from openaerostruct.functionals.moment_coefficient import MomentCoefficient
from openaerostruct.functionals.total_lift_drag import TotalLiftDrag


class TotalPerformance(Group):
    """
    Group to contain the total aerostructural performance components.
    """

    def __init__(self, surfaces, prob_dict):
        super(TotalPerformance, self).__init__()

        self.add_subsystem('fuelburn',
                 BreguetRange(surfaces=surfaces, prob_dict=prob_dict),
                 promotes=['*'])
        self.add_subsystem('L_equals_W',
                 Equilibrium(surfaces=surfaces, prob_dict=prob_dict),
                 promotes=['*'])
        self.add_subsystem('CG',
                 CenterOfGravity(surfaces=surfaces, prob_dict=prob_dict),
                 promotes=['*'])
        self.add_subsystem('moment',
                 MomentCoefficient(surfaces=surfaces, prob_dict=prob_dict),
                 promotes=['*'])
        self.add_subsystem('CL_CD',
                 TotalLiftDrag(surfaces=surfaces, prob_dict=prob_dict),
                 promotes=['*'])
