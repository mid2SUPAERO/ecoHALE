import numpy as np

from openaerostruct.geometry.bsplines import Bsplines
from openaerostruct.geometry.radius_comp import RadiusComp

from openmdao.api import IndepVarComp, Group

class Geometry(Group):
    """ Group that contains everything needed for a structural-only problem. """

    def initialize(self):
        self.metadata.declare('surface', type_=dict)
        # self.metadata.declare('DVGeo')

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


        if 0:#self.metadata['DVGeo']:
            from openaerostruct.geometry.ffd_component import GeometryMesh
            indep_var_comp.add_output('shape', val=np.zeros((surface['mx'], surface['my'])), units='m')

            self.add_subsystem('mesh',
                GeometryMesh(surface=surface, DVGeo=self.metadata['DVGeo']),
                promotes_inputs=['shape'],
                promotes_outputs=['mesh'])

        else:
            from openaerostruct.geometry.geometry_mesh import GeometryMesh

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
                bsp_inputs.append('chord')
                indep_var_comp.add_output('chord_cp', val=surface['chord_cp'], units='m')

            if 'xshear_cp' in surface.keys():
                # Add bspline components for active bspline geometric variables.
                self.add_subsystem('xshear_bsp', Bsplines(
                    in_name='xshear_cp', out_name='xshear',
                    num_cp=len(surface['xshear_cp']), num_pt=int(ny)),
                    promotes_inputs=['xshear_cp'], promotes_outputs=['xshear'])
                bsp_inputs.append('xshear')
                indep_var_comp.add_output('xshear_cp', val=surface['xshear_cp'], units='m')

            if 'yshear_cp' in surface.keys():
                # Add bspline components for active bspline geometric variables.
                self.add_subsystem('yshear_bsp', Bsplines(
                    in_name='yshear_cp', out_name='yshear',
                    num_cp=len(surface['yshear_cp']), num_pt=int(ny)),
                    promotes_inputs=['yshear_cp'], promotes_outputs=['yshear'])
                bsp_inputs.append('yshear')
                indep_var_comp.add_output('yshear_cp', val=surface['yshear_cp'], units='m')

            if 'zshear_cp' in surface.keys():
                # Add bspline components for active bspline geometric variables.
                self.add_subsystem('zshear_bsp', Bsplines(
                    in_name='zshear_cp', out_name='zshear',
                    num_cp=len(surface['zshear_cp']), num_pt=int(ny)),
                    promotes_inputs=['zshear_cp'], promotes_outputs=['zshear'])
                bsp_inputs.append('zshear')
                indep_var_comp.add_output('zshear_cp', val=surface['zshear_cp'], units='m')

            self.add_subsystem('mesh',
                GeometryMesh(surface=surface),
                promotes_inputs=bsp_inputs,
                promotes_outputs=['mesh'])

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
