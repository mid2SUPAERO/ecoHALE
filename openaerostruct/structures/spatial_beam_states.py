from openmdao.api import Group
from openaerostruct.structures.create_rhs import CreateRHS
from openaerostruct.structures.fem import FEM
from openaerostruct.structures.disp import Disp

class SpatialBeamStates(Group):
    """ Group that contains the spatial beam states. """

    def initialize(self):
        self.options.declare('surface', types=dict)

    def setup(self):
        surface = self.options['surface']

        size = int(6 * surface['num_y'] + 6)

        self.add_subsystem('create_rhs',
                 CreateRHS(surface=surface),
                 promotes_inputs=['loads', 'element_weights'], promotes_outputs=['forces'])
        self.add_subsystem('fem',
                 FEM(size=size),
                 promotes_inputs=['*'], promotes_outputs=['*'])
        self.add_subsystem('disp',
                 Disp(surface=surface),
                 promotes_inputs=['*'], promotes_outputs=['*'])
