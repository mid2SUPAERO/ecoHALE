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

        if 'chord_cp' in surface.keys():
            # Add bspline components for active bspline geometric variables.
            self.add_subsystem('chord_bsp', Bsplines(
                in_name='chord_cp', out_name='chord',
                num_cp=len(surface['chord_cp']), num_pt=int(ny)),
                promotes_inputs=['chord_cp'], promotes_outputs=['chord'])
            indep_var_comp.add_output('chord_cp', val=surface['chord_cp'])

        if 'xshear_cp' in surface.keys():
            # Add bspline components for active bspline geometric variables.
            self.add_subsystem('xshear_bsp', Bsplines(
                in_name='xshear_cp', out_name='xshear',
                num_cp=len(surface['xshear_cp']), num_pt=int(ny)),
                promotes_inputs=['xshear_cp'], promotes_outputs=['xshear'])
            indep_var_comp.add_output('xshear_cp', val=surface['xshear_cp'])

        if 'yshear_cp' in surface.keys():
            # Add bspline components for active bspline geometric variables.
            self.add_subsystem('yshear_bsp', Bsplines(
                in_name='yshear_cp', out_name='yshear',
                num_cp=len(surface['yshear_cp']), num_pt=int(ny)),
                promotes_inputs=['yshear_cp'], promotes_outputs=['yshear'])
            indep_var_comp.add_output('yshear_cp', val=surface['yshear_cp'])

        if 'zshear_cp' in surface.keys():
            # Add bspline components for active bspline geometric variables.
            self.add_subsystem('zshear_bsp', Bsplines(
                in_name='zshear_cp', out_name='zshear',
                num_cp=len(surface['zshear_cp']), num_pt=int(ny)),
                promotes_inputs=['zshear_cp'], promotes_outputs=['zshear'])
            indep_var_comp.add_output('zshear_cp', val=surface['zshear_cp'])

        if 'thickness_cp' in surface.keys():
            # Add bspline components for active bspline geometric variables.
            self.add_subsystem('thickness_bsp', Bsplines(
                in_name='thickness_cp', out_name='thickness',
                num_cp=len(surface['thickness_cp']), num_pt=int(ny-1)),
                promotes_inputs=['thickness_cp'], promotes_outputs=['thickness'])
            indep_var_comp.add_output('thickness_cp', val=surface['thickness_cp'])

        mesh_promotes = ['mesh']

        if 'struct' in surface['type']:
            mesh_promotes.append('radius')

        self.add_subsystem('mesh',
            GeometryMesh(surface=surface),
            promotes_inputs=bsp_inputs,
            promotes_outputs=mesh_promotes)

        if surface['type'] == 'aero':
            indep_var_comp.add_output('disp', val=np.zeros((ny, 6)))
            self.add_subsystem('def_mesh',
                DisplacementTransfer(surface=surface),
                promotes_inputs=['disp', 'mesh'],
                promotes_outputs=['def_mesh'])
