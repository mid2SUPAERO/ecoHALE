from openmdao.api import Group, ExplicitComponent
from openaerostruct.geometry.geometry_mesh import GeometryMesh
from openaerostruct.geometry.geometry_group import Geometry
from openaerostruct.structures.spatial_beam_states import SpatialBeamStates
from openaerostruct.structures.spatial_beam_functionals import SpatialBeamFunctionals
from openaerostruct.structures.spatial_beam_setup import SpatialBeamSetup
from openaerostruct.structures.tube_group import TubeGroup


class SpatialBeamAlone(Group):
    """ Group that contains everything needed for a structural-only problem. """

    def initialize(self):
        self.options.declare('surface', types=dict)

    def setup(self):
        surface = self.options['surface']

        tube_promotes = []

        if 'thickness_cp' in surface.keys():
            tube_promotes.append('thickness_cp')

        self.add_subsystem('geometry',
            Geometry(surface=surface),
            promotes_inputs=[],
            promotes_outputs=['mesh'])

        if surface['fem_model_type'] == 'tube':
            self.add_subsystem('tube_group',
                TubeGroup(surface=surface),
                promotes_inputs=['mesh'],
                promotes_outputs=['A', 'Iy', 'Iz', 'J', 'radius', 'thickness'] + tube_promotes)
        elif surface['fem_model_type'] == 'wingbox':
            self.add_subsystem('wingbox_group',
                WingboxGroup(surface=surface),
                promotes_inputs=['mesh'],
                promotes_outputs=['A', 'Iy', 'Iz', 'J', 'radius', 'thickness'] + tube_promotes)
        else:
            raise NameError('Please select a valid `fem_model_type` from either `tube` or `wingbox`.')

        self.add_subsystem('struct_setup',
            SpatialBeamSetup(surface=surface),
            promotes_inputs=['mesh', 'A', 'Iy', 'Iz', 'J', 'load_factor'],
            promotes_outputs=['nodes', 'K', 'structural_weight', 'cg_location', 'element_weights'])

        self.add_subsystem('struct_states',
            SpatialBeamStates(surface=surface),
            promotes_inputs=['K', 'forces', 'loads', 'element_weights'],
            promotes_outputs=['disp'])

        self.add_subsystem('struct_funcs',
            SpatialBeamFunctionals(surface=surface),
            promotes_inputs=['thickness', 'radius', 'nodes', 'disp'],
            promotes_outputs=['thickness_intersects', 'vonmises', 'failure'])
