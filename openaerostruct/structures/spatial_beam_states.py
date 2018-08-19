from openmdao.api import Group
from openaerostruct.structures.create_rhs import CreateRHS
from openaerostruct.structures.fem import FEM
from openaerostruct.structures.disp import Disp
from openaerostruct.structures.wing_weight_loads import StructureWeightLoads
from openaerostruct.structures.total_loads import TotalLoads

class SpatialBeamStates(Group):
    """ Group that contains the spatial beam states. """

    def initialize(self):
        self.options.declare('surface', types=dict)

    def setup(self):
        surface = self.options['surface']

        size = int(6 * surface['num_y'] + 6)

        self.add_subsystem('struct_weight_loads',
                 StructureWeightLoads(surface=surface),
                 promotes_inputs=['element_weights', 'nodes'], promotes_outputs=['struct_weight_loads'])
        self.add_subsystem('total_loads',
                 TotalLoads(surface=surface),
                 promotes_inputs=['loads', 'struct_weight_loads'], promotes_outputs=['total_loads'])
        self.add_subsystem('create_rhs',
                 CreateRHS(surface=surface),
                 promotes_inputs=['total_loads', 'element_weights'], promotes_outputs=['forces'])
        self.add_subsystem('fem',
                 FEM(size=size),
                 promotes_inputs=['*'], promotes_outputs=['*'])
        self.add_subsystem('disp',
                 Disp(surface=surface),
                 promotes_inputs=['*'], promotes_outputs=['*'])
