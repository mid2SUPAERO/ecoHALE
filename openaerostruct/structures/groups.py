from openmdao.api import Group, ExplicitComponent
from openaerostruct.geometry.new_geometry_mesh import GeometryMesh
from openaerostruct.geometry.geometry_group import Geometry
from openaerostruct.structures.spatial_beam_states import SpatialBeamStates
from openaerostruct.structures.spatial_beam_functionals import SpatialBeamFunctionals
from openaerostruct.structures.spatial_beam_setup import SpatialBeamSetup
from openaerostruct.structures.materials_tube import MaterialsTube


class SpatialBeamAlone(Group):
    """ Group that contains everything needed for a structural-only problem. """

    def initialize(self):
        self.metadata.declare('surface', type_=dict, required=True)

    def setup(self):
        surface = self.metadata['surface']

        geom_promotes = []

        if 'thickness_cp' in surface.keys():
            geom_promotes.append('thickness_cp')

        self.add_subsystem('geometry',
            Geometry(surface=surface),
            # promotes=['*'])
            # TODO: Check out this bug here; promotes=['*'] works but promotes_inputs=['*'] doesn't work
            promotes_inputs=['*'],
            promotes_outputs=['mesh', 'radius', 'thickness'])

        self.add_subsystem('tube',
            MaterialsTube(surface=surface),
            promotes_inputs=['thickness', 'radius'],
            promotes_outputs=['A', 'Iy', 'Iz', 'J'])

        self.add_subsystem('struct_setup',
            SpatialBeamSetup(surface=surface),
            promotes_inputs=['mesh', 'A', 'Iy', 'Iz', 'J'],
            promotes_outputs=['nodes', 'K'])

        self.add_subsystem('struct_states',
            SpatialBeamStates(surface=surface),
            promotes_inputs=['K', 'forces', 'loads'],
            promotes_outputs=['disp'])

        self.add_subsystem('struct_funcs',
            SpatialBeamFunctionals(surface=surface),
            promotes_inputs=['thickness', 'radius', 'A', 'nodes', 'disp'],
            promotes_outputs=['thickness_intersects', 'structural_weight', 'cg_location', 'vonmises', 'failure'])
