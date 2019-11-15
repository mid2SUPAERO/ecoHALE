import numpy as np

from openmdao.api import IndepVarComp, Group, BsplinesComp


class Geometry(Group):
    """
    Group that contains all components needed for any type of OAS problem.

    Because we use this general group, there's some logic to figure out which
    components to add and which connections to make.
    This is especially true for all of the geometric manipulation types, such
    as twist, sweep, etc., in that we handle the creation of these parameters
    differently if the user wants to have them vary in the optimization problem.    
    """
    def initialize(self):
        self.options.declare('surface', types=dict)
        self.options.declare('DVGeo', default=None)

    def setup(self):
        surface = self.options['surface']

        # Get the surface name and create a group to contain components
        # only for this surface
        ny = surface['mesh'].shape[1]

        # Check if any control points were added to the surface dict
        dv_keys = set(['twist_cp', 'chord_cp', 'xshear_cp', 'yshear_cp', 'zshear_cp', 'sweep', 'span', 'taper', 'dihedral', 't_over_c_cp'])
        active_dv_keys = dv_keys.intersection(set(surface.keys()))
        # Make sure that at least one of them is an independent variable
        make_ivc = False
        for key in active_dv_keys:
            if surface.get(key + '_dv', True):
                make_ivc = True
                break

        if make_ivc or self.options['DVGeo']:
            # Add independent variables that do not belong to a specific component
            indep_var_comp = IndepVarComp()

            # Add structural components to the surface-specific group
            self.add_subsystem('indep_vars',
                     indep_var_comp,
                     promotes=['*'])

        if self.options['DVGeo']:
            from openaerostruct.geometry.ffd_component import GeometryMesh
            indep_var_comp.add_output('shape', val=np.zeros((surface['mx'], surface['my'])), units='m')

            if 't_over_c_cp' in surface.keys():
                n_cp = len(surface['t_over_c_cp'])
                # Add bspline components for active bspline geometric variables.
                self.add_subsystem('t_over_c_bsp', BsplinesComp(
                    in_name='t_over_c_cp', out_name='t_over_c',
                    num_control_points=n_cp, num_points=int(ny-1),
                    bspline_order=min(n_cp, 4), distribution='uniform'),
                    promotes_inputs=['t_over_c_cp'], promotes_outputs=['t_over_c'])
                if surface.get('t_over_c_cp_dv', True):
                    indep_var_comp.add_output('t_over_c_cp', val=surface['t_over_c_cp'])

            self.add_subsystem('mesh',
                GeometryMesh(surface=surface, DVGeo=self.options['DVGeo']),
                promotes_inputs=['shape'],
                promotes_outputs=['mesh'])

        else:
            from openaerostruct.geometry.geometry_mesh import GeometryMesh

            bsp_inputs = []

            if 'twist_cp' in surface.keys():
                n_cp = len(surface['twist_cp'])
                # Add bspline components for active bspline geometric variables.
                self.add_subsystem('twist_bsp', BsplinesComp(
                    in_name='twist_cp', out_name='twist', units='deg',
                    num_control_points=n_cp, num_points=int(ny),
                    bspline_order=min(n_cp, 4), distribution='uniform'),
                    promotes_inputs=['twist_cp'], promotes_outputs=['twist'])
                bsp_inputs.append('twist')

                # Since default assumption is that we want tail rotation as a design variable, add this to allow for trimmed drag polar where the tail rotation should not be a design variable
                if surface.get('twist_cp_dv', True):
                    indep_var_comp.add_output('twist_cp', val=surface['twist_cp'], units = 'deg')

            if 'chord_cp' in surface.keys():
                n_cp = len(surface['chord_cp'])
                # Add bspline components for active bspline geometric variables.
                self.add_subsystem('chord_bsp', BsplinesComp(
                    in_name='chord_cp', out_name='chord', units='m',
                    num_control_points=n_cp, num_points=int(ny),
                    bspline_order=min(n_cp, 4), distribution='uniform'),
                    promotes_inputs=['chord_cp'], promotes_outputs=['chord'])
                bsp_inputs.append('chord')
                if surface.get('chord_cp_dv', True):
                    indep_var_comp.add_output('chord_cp', val=surface['chord_cp'], units='m')

            if 't_over_c_cp' in surface.keys():
                n_cp = len(surface['t_over_c_cp'])
                # Add bspline components for active bspline geometric variables.
                self.add_subsystem('t_over_c_bsp', BsplinesComp(
                    in_name='t_over_c_cp', out_name='t_over_c',
                    num_control_points=n_cp, num_points=int(ny-1),
                    bspline_order=min(n_cp, 4), distribution='uniform'),
                    promotes_inputs=['t_over_c_cp'], promotes_outputs=['t_over_c'])
                if surface.get('t_over_c_cp_dv', True):
                    indep_var_comp.add_output('t_over_c_cp', val=surface['t_over_c_cp'])

            if 'xshear_cp' in surface.keys():
                n_cp = len(surface['xshear_cp'])
                # Add bspline components for active bspline geometric variables.
                self.add_subsystem('xshear_bsp', BsplinesComp(
                    in_name='xshear_cp', out_name='xshear', units='m',
                    num_control_points=n_cp, num_points=int(ny),
                    bspline_order=min(n_cp, 4), distribution='uniform'),
                    promotes_inputs=['xshear_cp'], promotes_outputs=['xshear'])
                bsp_inputs.append('xshear')
                if surface.get('xshear_cp_dv', True):
                    indep_var_comp.add_output('xshear_cp', val=surface['xshear_cp'], units='m')

            if 'yshear_cp' in surface.keys():
                n_cp = len(surface['yshear_cp'])
                # Add bspline components for active bspline geometric variables.
                self.add_subsystem('yshear_bsp', BsplinesComp(
                    in_name='yshear_cp', out_name='yshear', units='m',
                    num_control_points=n_cp, num_points=int(ny),
                    bspline_order=min(n_cp, 4), distribution='uniform'),
                    promotes_inputs=['yshear_cp'], promotes_outputs=['yshear'])
                bsp_inputs.append('yshear')
                if surface.get('yshear_cp_dv', True):
                    indep_var_comp.add_output('yshear_cp', val=surface['yshear_cp'], units='m')

            if 'zshear_cp' in surface.keys():
                n_cp = len(surface['zshear_cp'])
                # Add bspline components for active bspline geometric variables.
                self.add_subsystem('zshear_bsp', BsplinesComp(
                    in_name='zshear_cp', out_name='zshear', units='m',
                    num_control_points=n_cp, num_points=int(ny),
                    bspline_order=min(n_cp, 4), distribution='uniform'),
                    promotes_inputs=['zshear_cp'], promotes_outputs=['zshear'])
                bsp_inputs.append('zshear')
                if surface.get('zshear_cp_dv', True):
                    indep_var_comp.add_output('zshear_cp', val=surface['zshear_cp'], units='m')

            if 'sweep' in surface.keys():
                bsp_inputs.append('sweep')
                if surface.get('sweep_dv', True):
                    indep_var_comp.add_output('sweep', val=surface['sweep'], units='deg')

            if 'span' in surface.keys():
                bsp_inputs.append('span')
                if surface.get('span_dv', True):
                    indep_var_comp.add_output('span', val=surface['span'], units='m')


            if 'dihedral' in surface.keys():
                bsp_inputs.append('dihedral')
                if surface.get('dihedral_dv', True):
                    indep_var_comp.add_output('dihedral', val=surface['dihedral'], units='deg')

            if 'taper' in surface.keys():
                bsp_inputs.append('taper')
                if surface.get('taper_dv', True):
                    indep_var_comp.add_output('taper', val=surface['taper'])

            self.add_subsystem('mesh',
                GeometryMesh(surface=surface),
                promotes_inputs=bsp_inputs,
                promotes_outputs=['mesh'])