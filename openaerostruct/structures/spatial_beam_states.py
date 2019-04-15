from openmdao.api import Group
from openaerostruct.structures.create_rhs import CreateRHS
from openaerostruct.structures.fem import FEM
from openaerostruct.structures.disp import Disp
from openaerostruct.structures.wing_weight_loads import StructureWeightLoads
from openaerostruct.structures.fuel_loads import FuelLoads
from openaerostruct.structures.total_loads import TotalLoads
from openaerostruct.structures.add_point_masses import AddPointMasses

class SpatialBeamStates(Group):
    """ Group that contains the spatial beam states. """

    def initialize(self):
        self.options.declare('surface', types=dict)

    def setup(self):
        surface = self.options['surface']

        promotes = []
        if surface['struct_weight_relief']:
            self.add_subsystem('struct_weight_loads',
                     StructureWeightLoads(surface=surface),
                     promotes_inputs=['element_mass', 'nodes', 'load_factor'],
                     promotes_outputs=['struct_weight_loads'])
            promotes.append('struct_weight_loads')

        if surface['distributed_fuel_weight']:
            self.add_subsystem('fuel_loads',
                     FuelLoads(surface=surface),
                     promotes_inputs=['nodes', 'load_factor', 'fuel_vols', 'fuel_mass'],
                     promotes_outputs=['fuel_weight_loads'])
            promotes.append('fuel_weight_loads')

        if 'n_point_masses' in surface.keys():
            self.add_subsystem('point_masses',
                     AddPointMasses(surface=surface),
                     promotes_inputs=['point_mass_locations', 'point_masses', 'nodes', 'load_factor'],
                     promotes_outputs=['loads_from_point_masses'])
            promotes.append('loads_from_point_masses')

        self.add_subsystem('total_loads',
                 TotalLoads(surface=surface),
                 promotes_inputs=['loads'] + promotes,
                 promotes_outputs=['total_loads'])

        self.add_subsystem('create_rhs',
                 CreateRHS(surface=surface),
                 promotes_inputs=['total_loads'], promotes_outputs=['forces'])

        self.add_subsystem('fem',
                 FEM(surface=surface),
                 promotes_inputs=['*'], promotes_outputs=['*'])

        self.add_subsystem('disp',
                 Disp(surface=surface),
                 promotes_inputs=['*'], promotes_outputs=['*'])
