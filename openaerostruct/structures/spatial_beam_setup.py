from openmdao.api import Group
from openaerostruct.structures.compute_nodes import ComputeNodes
from openaerostruct.structures.assemble_k import AssembleK
from openaerostruct.structures.weight import Weight
from openaerostruct.structures.structural_cg import StructuralCG


class SpatialBeamSetup(Group):
    """ Group that sets up the spatial beam components and assembles the
        stiffness matrix."""

    def initialize(self):
        self.options.declare('surface', types=dict)

    def setup(self):
        surface = self.options['surface']

        self.add_subsystem('nodes',
                 ComputeNodes(surface=surface),
                 promotes_inputs=['mesh'], promotes_outputs=['nodes'])

        self.add_subsystem('assembly',
                 AssembleK(surface=surface),
                 promotes_inputs=['A', 'Iy', 'Iz', 'J', 'nodes'], promotes_outputs=['K'])

        self.add_subsystem('structural_weight',
                 Weight(surface=surface),
                 promotes_inputs=['A', 'nodes', 'load_factor'],
                 promotes_outputs=['structural_weight', 'element_weights'])

        self.add_subsystem('structural_cg',
            StructuralCG(surface=surface),
            promotes_inputs=['nodes', 'structural_weight', 'element_weights'],
            promotes_outputs=['cg_location'])
