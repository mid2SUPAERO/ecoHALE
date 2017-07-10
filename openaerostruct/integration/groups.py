from openmdao.api import Group, ExplicitComponent, LinearRunOnce
from openaerostruct.geometry.new_geometry_mesh import GeometryMesh
from openaerostruct.aerodynamics.geometry import VLMGeometry
from openaerostruct.geometry.bsplines import Bsplines
from openaerostruct.transfer.displacement_transfer import DisplacementTransfer
from openaerostruct.structures.materials_tube import MaterialsTube
from openaerostruct.structures.spatial_beam_setup import SpatialBeamSetup
from openaerostruct.transfer.displacement_transfer import DisplacementTransfer
from openaerostruct.structures.spatial_beam_states import SpatialBeamStates
from openaerostruct.aerodynamics.functionals import VLMFunctionals
from openaerostruct.structures.spatial_beam_functionals import SpatialBeamFunctionals


class Aerostruct(Group):
    """ Group that contains everything needed for a structural-only problem. """

    def initialize(self):
        self.metadata.declare('surface', type_=dict, required=True)
        self.metadata.declare('indep_var_comp', type_=ExplicitComponent, required=True)

    def setup(self):
        surface = self.metadata['surface']
        indep_var_comp = self.metadata['indep_var_comp']
        ny = surface['mesh'].shape[1]
        num_thickness_cp = 3

        # Add components to include in the surface's group
        self.add_subsystem('indep_vars',
            indep_var_comp,
            promotes=['*'])

        # Add bspline components for active bspline geometric variables.
        # We only add the component if the corresponding variable is a desvar,
        # a special parameter (radius), or if the user or geometry provided
        # an initial distribution.
        self.add_subsystem('twist_bsp', Bsplines(
            in_name='twist_cp', out_name='twist',
            num_cp=int(surface['num_twist_cp']), num_pt=int(ny)),
            promotes=['*'])
        self.add_subsystem('thickness_bsp', Bsplines(
            in_name='thickness_cp', out_name='thickness',
            num_cp=int(surface['num_thickness_cp']), num_pt=int(ny-1)),
            promotes=['*'])

        self.add_subsystem('mesh',
            GeometryMesh(surface=surface),
            promotes=['*'])
        self.add_subsystem('tube',
            MaterialsTube(surface=surface),
            promotes=['*'])

        self.add_subsystem('struct_setup',
            SpatialBeamSetup(surface=surface),
            promotes=['*'])

class CoupledAS(Group):
    """ Group that contains everything needed for a structural-only problem. """

    def initialize(self):
        self.metadata.declare('surface', type_=dict, required=True)

    def setup(self):
        surface = self.metadata['surface']

        self.add_subsystem('struct_states',
            SpatialBeamStates(surface=surface),
            promotes=['*'])
        self.add_subsystem('def_mesh',
            DisplacementTransfer(surface=surface),
            promotes=['*'])
        self.add_subsystem('aero_geom',
            VLMGeometry(surface=surface),
            promotes=['*'])

        self.linear_solver = LinearRunOnce()

class CoupledPerformance(Group):
    """ Group that contains everything needed for a structural-only problem. """

    def initialize(self):
        self.metadata.declare('surface', type_=dict, required=True)
        self.metadata.declare('prob_dict', type_=dict, required=True)

    def setup(self):
        surface = self.metadata['surface']
        prob_dict = self.metadata['prob_dict']

        self.add_subsystem('aero_funcs',
            VLMFunctionals(surface=surface, prob_dict=prob_dict),
            promotes=['*'])
        self.add_subsystem('struct_funcs',
            SpatialBeamFunctionals(surface=surface),
            promotes=['*'])
