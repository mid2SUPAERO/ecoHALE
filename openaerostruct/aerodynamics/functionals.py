from openmdao.api import Group
from openaerostruct.aerodynamics.lift_drag import LiftDrag
from openaerostruct.aerodynamics.coeffs import Coeffs
from openaerostruct.aerodynamics.toatl_lift import TotalLift
from openaerostruct.aerodynamics.toatl_drag import TotalDrag
from openaerostruct.aerodynamics.viscous_drag import ViscousDrag

class VLMFunctionals(Group):
    """
    Group that contains the aerodynamic functionals used to evaluate
    performance.
    """

    def __init__(self, surface, prob_dict):
        super(VLMFunctionals, self).__init__()

        with_viscous = prob_dict['with_viscous']

        self.add_subsystem('liftdrag',
                 LiftDrag(surface=surface),
                 promotes=['*'])
        self.add_subsystem('coeffs',
                 Coeffs(surface=surface),
                 promotes=['*'])
        self.add_subsystem('CL',
                 TotalLift(surface=surface),
                 promotes=['*'])
        self.add_subsystem('CD',
                 TotalDrag(surface=surface),
                 promotes=['*'])
        self.add_subsystem('viscousdrag',
                 ViscousDrag(surface=surface, with_viscous=with_viscous),
                 promotes=['*'])
