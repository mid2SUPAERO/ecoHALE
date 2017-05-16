from openmdao.api import Group
from openaerostruct.structures.compute_nodes import ComputeNodes
from openaerostruct.structures.assemble_k import AssembleK


class SpatialBeamSetup(Group):
    """ Group that sets up the spatial beam components and assembles the
        stiffness matrix."""

    def __init__(self, surface):
        super(SpatialBeamSetup, self).__init__()

        self.add_subsystem('nodes',
                 ComputeNodes(surface=surface),
                 promotes=['*'])
        self.add_subsystem('assembly',
                 AssembleK(surface=surface),
                 promotes=['*'])
