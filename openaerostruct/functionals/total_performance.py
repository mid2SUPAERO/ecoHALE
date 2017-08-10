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

    def setup(self):
        surfaces = self.metadata['surfaces']

        self.add_subsystem('CL_CD',
             TotalLiftDrag(surfaces=surfaces),
             promotes_inputs=['*CL', '*CD', '*S_ref'],
             promotes_outputs=['CL', 'CD'])

        self.add_subsystem('fuelburn',
             BreguetRange(surfaces=surfaces),
             promotes_inputs=['*structural_weight', 'CL', 'CD', 'CT', 'a', 'R', 'M', 'W0', 'load_factor'],
             promotes_outputs=['fuelburn'])

        self.add_subsystem('L_equals_W',
             Equilibrium(surfaces=surfaces),
             promotes_inputs=['*L', '*D', '*structural_weight', 'fuelburn', 'W0', 'load_factor', 'alpha'],
             promotes_outputs=['L_equals_W', 'total_weight'])

        self.add_subsystem('CG',
             CenterOfGravity(surfaces=surfaces),
             promotes_inputs=['*structural_weight', '*cg_location', 'fuelburn', 'total_weight', 'W0', 'empty_cg', 'load_factor'],
             promotes_outputs=['cg'])

        self.add_subsystem('moment',
             MomentCoefficient(surfaces=surfaces),
             promotes_inputs=['v', 'rho', 'cg', '*S_ref', '*b_pts', '*widths', '*chords', '*sec_forces'],
             promotes_outputs=['CM'])
