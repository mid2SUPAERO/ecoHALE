from __future__ import division, print_function
import numpy as np

from openmdao.api import Group

from openaerostruct.functionals.breguet_range import BreguetRange
from openaerostruct.functionals.equilibrium import Equilibrium
from openaerostruct.functionals.center_of_gravity import CenterOfGravity
from openaerostruct.functionals.moment_coefficient import MomentCoefficient
from openaerostruct.functionals.total_lift_drag import TotalLiftDrag
from openaerostruct.functionals.sum_areas import SumAreas


class TotalPerformance(Group):
    """
    Group to contain the total aerostructural performance components.
    """

    def initialize(self):
        self.options.declare('surfaces', types=list)

    def setup(self):
        surfaces = self.options['surfaces']

        self.add_subsystem('sum_areas',
             SumAreas(surfaces=surfaces),
             promotes_inputs=['*S_ref'],
             promotes_outputs=['S_ref_total'])

        self.add_subsystem('CL_CD',
             TotalLiftDrag(surfaces=surfaces),
             promotes_inputs=['*CL', '*CD', '*S_ref', 'S_ref_total'],
             promotes_outputs=['CL', 'CD'])

        self.add_subsystem('fuelburn',
             BreguetRange(surfaces=surfaces),
             promotes_inputs=['*structural_weight', 'CL', 'CD', 'CT', 'a', 'R', 'M', 'W0', 'load_factor'],
             promotes_outputs=['fuelburn'])

        self.add_subsystem('L_equals_W',
             Equilibrium(surfaces=surfaces),
             promotes_inputs=['*L', '*structural_weight', 'S_ref_total', 'fuelburn', 'W0', 'load_factor', 'rho', 'v'],
             promotes_outputs=['L_equals_W', 'total_weight'])

        self.add_subsystem('CG',
             CenterOfGravity(surfaces=surfaces),
             promotes_inputs=['*structural_weight', '*cg_location', 'fuelburn', 'total_weight', 'W0', 'empty_cg', 'load_factor'],
             promotes_outputs=['cg'])

        self.add_subsystem('moment',
             MomentCoefficient(surfaces=surfaces),
             promotes_inputs=['v', 'rho', 'cg', 'S_ref_total', '*b_pts', '*widths', '*chords', '*sec_forces', '*S_ref'],
             promotes_outputs=['CM'])
