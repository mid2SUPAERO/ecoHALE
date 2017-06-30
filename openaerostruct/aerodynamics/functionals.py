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
        self.metadata.declare('surface', type_=dict, required=True)
        self.metadata.declare('prob_dict', type_=dict, required=True)

    def setup(self):
        with_viscous = self.metadata['prob_dict']['with_viscous']
        surface = self.metadata['surface']

        self.add_subsystem('viscousdrag',
            ViscousDrag(surface=surface, with_viscous=with_viscous),
            promotes=['*'])
        self.add_subsystem('liftcoeff',
             LiftCoeff2D(surface=surface),
             promotes=['*'])
        self.add_subsystem('liftdrag',
            LiftDrag(surface=surface),
            promotes=['*'])
        self.add_subsystem('coeffs',
            Coeffs(surface=surface),
            promotes=['*'])
        self.add_subsystem('CD',
            TotalDrag(surface=surface),
            promotes=['*'])
        self.add_subsystem('CL',
            TotalLift(surface=surface),
            promotes=['*'])
