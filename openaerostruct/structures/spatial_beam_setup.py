from openmdao.api import Group
from openaerostruct.structures.compute_nodes import ComputeNodes
from openaerostruct.structures.assemble_k import AssembleK
from openaerostruct.structures.weight import Weight



class SpatialBeamSetup(Group):
    """ Group that sets up the spatial beam components and assembles the
        stiffness matrix."""

    def initialize(self):
        self.metadata.declare('surface', type_=dict)

    def setup(self):
        surface = self.metadata['surface']

        self.add_subsystem('nodes',
                 ComputeNodes(surface=surface),
                 promotes_inputs=['mesh'], promotes_outputs=['nodes'])

        self.add_subsystem('assembly',
                 AssembleK(surface=surface),
                 promotes_inputs=['A', 'Iy', 'Iz', 'J', 'nodes'], promotes_outputs=['K'])

        self.add_subsystem('structural_weight',
                 Weight(surface=surface),
                 promotes_inputs=['A', 'nodes', 'load_factor'],
                 promotes_outputs=['structural_weight', 'cg_location', 'element_weights'])
