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

    def initialize(self):
        self.metadata.declare('surfaces', type_=list, required=True)
        self.metadata.declare('prob_dict', type_=dict, required=True)

    def initialize_subsystems(self):
        prob_dict = self.metadata['prob_dict']
        with_viscous = prob_dict['with_viscous']
        surfaces = self.metadata['surfaces']

        self.add_subsystem('CL_CD',
             TotalLiftDrag(surfaces=surfaces, prob_dict=prob_dict),
             promotes=['*'])
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
