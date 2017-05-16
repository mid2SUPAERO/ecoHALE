from openmdao.api import Group
from openaerostruct.structures.create_rhs import CreateRHS
from openaerostruct.structures.fem import FEM
from openaerostruct.structures.disp import Disp

class SpatialBeamStates(Group):
    """ Group that contains the spatial beam states. """

    def __init__(self, surface):
        super(SpatialBeamStates, self).__init__()

        size = 6 * surface['num_y'] + 6

        self.add_subsystem('create_rhs',
                 CreateRHS(surface=surface),
                 promotes=['*'])
        self.add_subsystem('fem',
                 FEM(size=size),
                 promotes=['*'])
        self.add_subsystem('disp',
                 Disp(surface=surface),
                 promotes=['*'])
