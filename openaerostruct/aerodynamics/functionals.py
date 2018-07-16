from openmdao.api import Group
from openaerostruct.aerodynamics.lift_drag import LiftDrag
from openaerostruct.aerodynamics.coeffs import Coeffs
from openaerostruct.aerodynamics.total_lift import TotalLift
from openaerostruct.aerodynamics.total_drag import TotalDrag
from openaerostruct.aerodynamics.viscous_drag import ViscousDrag
from openaerostruct.aerodynamics.lift_coeff_2D import LiftCoeff2D


class VLMFunctionals(Group):
    """
    Group that contains the aerodynamic functionals used to evaluate
    performance.
    """

    def initialize(self):
        self.options.declare('surface', types=dict)

    def setup(self):
        surface = self.options['surface']

        self.add_subsystem('viscousdrag',
            ViscousDrag(surface=surface),
            promotes_inputs=['M', 're', 'widths', 'cos_sweep', 'lengths', 'S_ref'], promotes_outputs=['CDv'])

        # This component is generally not needed unless you want the sectional CLs
        self.add_subsystem('liftcoeff',
             LiftCoeff2D(surface=surface),
             promotes_inputs=['v', 'alpha', 'rho', 'widths', 'chords', 'sec_forces'],
             promotes_outputs=['Cl'])

        self.add_subsystem('liftdrag',
            LiftDrag(surface=surface),
            promotes_inputs=['alpha', 'sec_forces'],
            promotes_outputs=['L', 'D'])

        self.add_subsystem('coeffs',
            Coeffs(surface=surface),
            promotes_inputs=['v', 'rho', 'S_ref', 'L', 'D'],
            promotes_outputs=['CL1', 'CDi'])

        self.add_subsystem('CD',
            TotalDrag(surface=surface),
            promotes_inputs=['CDv', 'CDi'],
            promotes_outputs=['CD'])

        self.add_subsystem('CL',
            TotalLift(surface=surface),
            promotes_inputs=['CL1'],
            promotes_outputs=['CL'])
