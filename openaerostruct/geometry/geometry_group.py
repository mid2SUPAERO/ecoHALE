import numpy as np

from openaerostruct.geometry.new_geometry_mesh import GeometryMesh
from openaerostruct.aerodynamics.geometry import VLMGeometry
from openaerostruct.geometry.bsplines import Bsplines
from openaerostruct.transfer.displacement_transfer import DisplacementTransfer
from openaerostruct.structures.materials_tube import MaterialsTube
from openaerostruct.structures.spatial_beam_setup import SpatialBeamSetup
from openaerostruct.structures.spatial_beam_states import SpatialBeamStates
from openaerostruct.aerodynamics.functionals import VLMFunctionals
from openaerostruct.structures.spatial_beam_functionals import SpatialBeamFunctionals
from openaerostruct.functionals.total_performance import TotalPerformance
from openaerostruct.transfer.load_transfer import LoadTransfer
from openaerostruct.aerodynamics.states import VLMStates

from openmdao.api import IndepVarComp, Problem, Group, NewtonSolver, ScipyIterativeSolver, LinearBlockGS, NonlinearBlockGS, DirectSolver, DenseJacobian, LinearRunOnce, ExplicitComponent

class Geometry(Group):
    """ Group that contains everything needed for a structural-only problem. """

    def initialize(self):
        self.metadata.declare('surface', type_=dict, required=True)

    def setup(self):
        surface = self.metadata['surface']

        # Get the surface name and create a group to contain components
        # only for this surface
        ny = surface['mesh'].shape[1]

        # Add independent variables that do not belong to a specific component
        indep_var_comp = IndepVarComp()
        indep_var_comp.add_output('disp', val=np.zeros((ny, 6)))

        # Add structural components to the surface-specific group
        self.add_subsystem('indep_vars',
                 indep_var_comp,
                 promotes=['*'])

        bsp_inputs = []

        if 'twist_cp' in surface.keys():
            # Add bspline components for active bspline geometric variables.
            self.add_subsystem('twist_bsp', Bsplines(
                in_name='twist_cp', out_name='twist',
                num_cp=len(surface['twist_cp']), num_pt=int(ny)),
                promotes_inputs=['twist_cp'], promotes_outputs=['twist'])
            bsp_inputs.append('twist')
            indep_var_comp.add_output('twist_cp', val=surface['twist_cp'])

        if 'thickness_cp' in surface.keys():
            # Add bspline components for active bspline geometric variables.
            self.add_subsystem('thickness_bsp', Bsplines(
                in_name='thickness_cp', out_name='thickness',
                num_cp=len(surface['thickness_cp']), num_pt=int(ny-1)),
                promotes_inputs=['thickness_cp'], promotes_outputs=['thickness'])
            indep_var_comp.add_output('thickness_cp', val=surface['thickness_cp'])

        self.add_subsystem('mesh',
            GeometryMesh(surface=surface),
            promotes_inputs=bsp_inputs,
            promotes_outputs=['mesh', 'radius'])

        if surface['type'] == 'aero':
            self.add_subsystem('def_mesh',
                DisplacementTransfer(surface=surface),
                promotes_inputs=['disp', 'mesh'],
                promotes_outputs=['def_mesh'])
