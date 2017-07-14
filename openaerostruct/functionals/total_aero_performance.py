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

    def setup(self):
        surfaces = self.metadata['surfaces']

        self.add_subsystem('CL_CD',
                 TotalLiftDrag(surfaces=surfaces),
                 promotes_inputs=['*CL', '*CD', '*S_ref', 'S_ref_total'],
                 promotes_outputs=['CL', 'CD'])

        self.add_subsystem('moment',
                 MomentCoefficient(surfaces=surfaces),
                 promotes_inputs=['v', 'cg', 'rho', '*S_ref', '*b_pts', '*widths', '*chords', '*sec_forces', 'S_ref_total'],
                 promotes_outputs=['CM'])
