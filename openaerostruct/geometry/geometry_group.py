import numpy as np

from openaerostruct.geometry.bsplines import Bsplines
from openaerostruct.geometry.radius_comp import RadiusComp

from openmdao.api import IndepVarComp, Group

class Geometry(Group):
    """ Group that contains everything needed for a structural-only problem. """

    def initialize(self):
        self.options.declare('surface', types=dict)
        self.options.declare('DVGeo', default=None)

    def setup(self):
        surface = self.options['surface']

        # Get the surface name and create a group to contain components
        # only for this surface
        ny = surface['mesh'].shape[1]

        if 'twist_cp' in surface.keys() or 'chord_cp' in surface.keys() or 'xshear_cp' in surface.keys() or 'yshear_cp' in surface.keys() or 'zshear_cp' in surface.keys() or self.options['DVGeo']:
            # Add independent variables that do not belong to a specific component
            indep_var_comp = IndepVarComp()

            # Add structural components to the surface-specific group
            self.add_subsystem('indep_vars',
                     indep_var_comp,
                     promotes=['*'])

        if self.options['DVGeo']:
            from openaerostruct.geometry.ffd_component import GeometryMesh
            indep_var_comp.add_output('shape', val=np.zeros((surface['mx'], surface['my'])), units='m')

            self.add_subsystem('mesh',
                GeometryMesh(surface=surface, DVGeo=self.options['DVGeo']),
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

            if 'toverc_cp' in surface.keys():
                # Add bspline components for active bspline geometric variables.
                self.add_subsystem('toverc_bsp', Bsplines(
                    in_name='toverc_cp', out_name='toverc',
                    num_cp=len(surface['toverc_cp']), num_pt=int(ny-1)),
                    promotes_inputs=['toverc_cp'], promotes_outputs=['toverc'])
                indep_var_comp.add_output('toverc_cp', val=surface['toverc_cp'], units='m')

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
