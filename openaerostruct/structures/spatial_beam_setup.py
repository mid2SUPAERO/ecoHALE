from openmdao.api import Group
from openaerostruct.structures.compute_nodes import ComputeNodes
from openaerostruct.structures.assemble_k import AssembleK


class SpatialBeamSetup(Group):
    """ Group that sets up the spatial beam components and assembles the
        stiffness matrix."""

    def initialize(self):
        self.metadata.declare('surface', type_=dict, required=True)

    def setup(self):
        surface = self.metadata['surface']

        self.add_subsystem('nodes',
                 ComputeNodes(surface=surface),
                 promotes_inputs=['mesh'], promotes_outputs=['nodes'])

        self.add_subsystem('assembly',
                 AssembleK(surface=surface),
                 promotes_inputs=['A', 'Iy', 'Iz', 'J', 'nodes'], promotes_outputs=['K'])
