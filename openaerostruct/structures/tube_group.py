from openmdao.api import Group, ExplicitComponent
from openaerostruct.geometry.geometry_mesh import GeometryMesh
from openaerostruct.geometry.geometry_group import Geometry
from openaerostruct.structures.spatial_beam_states import SpatialBeamStates
from openaerostruct.structures.spatial_beam_functionals import SpatialBeamFunctionals
from openaerostruct.structures.spatial_beam_setup import SpatialBeamSetup
from openaerostruct.structures.materials_tube import MaterialsTube
from openaerostruct.geometry.bsplines import Bsplines
from openaerostruct.geometry.radius_comp import RadiusComp

from openmdao.api import IndepVarComp, Group


class TubeGroup(Group):
    """ Group that contains everything needed for a structural-only problem. """

    def initialize(self):
        self.options.declare('surface', types=dict)

    def setup(self):
        surface = self.options['surface']
        ny = surface['num_y']

        # Add independent variables that do not belong to a specific component
        indep_var_comp = IndepVarComp()

        # Add structural components to the surface-specific group
        self.add_subsystem('indep_vars',
                 indep_var_comp,
                 promotes=['*'])

        if 'struct' in surface['type']:
            self.add_subsystem('radius_comp',
                RadiusComp(surface=surface),
                promotes_inputs=['mesh'],
                promotes_outputs=['radius'])

        if 'thickness_cp' in surface.keys():
            # Add bspline components for active bspline geometric variables.
            self.add_subsystem('thickness_bsp', Bsplines(
                in_name='thickness_cp', out_name='thickness',
                num_cp=len(surface['thickness_cp']), num_pt=int(ny-1)),
                promotes_inputs=['thickness_cp'], promotes_outputs=['thickness'])
            indep_var_comp.add_output('thickness_cp', val=surface['thickness_cp'], units='m')

        self.add_subsystem('tube',
            MaterialsTube(surface=surface),
            promotes_inputs=['thickness', 'radius'],
            promotes_outputs=['A', 'Iy', 'Iz', 'J'])
