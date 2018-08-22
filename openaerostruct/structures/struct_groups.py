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
            promotes_outputs=['mesh', 't_over_c'])

        if surface['fem_model_type'] == 'tube':
            self.add_subsystem('tube_group',
                TubeGroup(surface=surface),
                promotes_inputs=['mesh', 't_over_c'],
                promotes_outputs=['A', 'Iy', 'Iz', 'J', 'radius', 'thickness'] + tube_promotes)
        elif surface['fem_model_type'] == 'wingbox':
            wingbox_promotes = []
            if 'skin_thickness_cp' in surface.keys() and 'spar_thickness_cp' in surface.keys():
                wingbox_promotes.append('skin_thickness_cp')
                wingbox_promotes.append('spar_thickness_cp')
                wingbox_promotes.append('skin_thickness')
                wingbox_promotes.append('spar_thickness')
            elif 'skin_thickness_cp' in surface.keys() or 'spar_thickness_cp' in surface.keys():
                raise NameError('Please have both skin and spar thickness as design variables, not one or the other.')

            self.add_subsystem('wingbox_group',
                WingboxGroup(surface=surface),
                promotes_inputs=['mesh', 't_over_c'],
                promotes_outputs=['A', 'Iy', 'Iz', 'J', 'Qz', 'A_enc', 'htop', 'hbottom', 'hfront', 'hrear'] + wingbox_promotes)
        else:
            raise NameError('Please select a valid `fem_model_type` from either `tube` or `wingbox`.')

        self.add_subsystem('struct_setup',
            SpatialBeamSetup(surface=surface),
            promotes_inputs=['mesh', 'A', 'Iy', 'Iz', 'J', 'load_factor'],
            promotes_outputs=['nodes', 'K', 'structural_weight', 'cg_location', 'element_weights'])

        promotes = []
        if surface['struct_weight_relief']:
            promotes = ['nodes']
        if surface['distributed_fuel_weight']:
            promotes = ['nodes', 'load_factor']

        self.add_subsystem('struct_states',
            SpatialBeamStates(surface=surface),
            promotes_inputs=['K', 'forces', 'loads', 'element_weights'] + promotes,
            promotes_outputs=['disp'])

        self.add_subsystem('struct_funcs',
            SpatialBeamFunctionals(surface=surface),
            promotes_inputs=['thickness', 'radius', 'nodes', 'disp'],
            promotes_outputs=['thickness_intersects', 'vonmises', 'failure'])
