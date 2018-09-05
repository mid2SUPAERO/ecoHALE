from openmdao.api import IndepVarComp, Group
from openaerostruct.transfer.displacement_transfer import DisplacementTransfer
from openaerostruct.transfer.compute_transformation_matrix import ComputeTransformationMatrix


class DisplacementTransferGroup(Group):
    """ Group that contains everything needed for a structural-only problem. """

    def initialize(self):
        self.options.declare('surface', types=dict)

    def setup(self):
        surface = self.options['surface']

        self.add_subsystem('compute_transformation_matrix',
                 ComputeTransformationMatrix(surface=surface),
                 promotes=['*'])

        self.add_subsystem('displacement_transfer',
                 DisplacementTransfer(surface=surface),
                 promotes=['*'])
