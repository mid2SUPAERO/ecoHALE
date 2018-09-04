from openmdao.api import Group, ExplicitComponent, BsplinesComp
from openaerostruct.transfer.compute_ref_curve import ComputeRefCurve
from openaerostruct.transfer.displacement_transfer import DisplacementTransfer


from openmdao.api import IndepVarComp, Group


class DisplacementTransferGroup(Group):
    """ Group that contains everything needed for a structural-only problem. """

    def initialize(self):
        self.options.declare('surface', types=dict)

    def setup(self):
        surface = self.options['surface']

        self.add_subsystem('compute_ref_curve',
                 ComputeRefCurve(surface=surface),
                 promotes=['*'])

        self.add_subsystem('displacement_transfer',
                 DisplacementTransfer(surface=surface),
                 promotes=['*'])
