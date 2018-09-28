from openmdao.api import Group, BsplinesComp
from openaerostruct.structures.section_properties_tube import SectionPropertiesTube
from openaerostruct.geometry.radius_comp import RadiusComp

from openmdao.api import IndepVarComp, Group


class TubeGroup(Group):
    """ Group that contains everything needed for a structural-only problem. """

    def initialize(self):
        self.options.declare('surface', types=dict)

    def setup(self):
        surface = self.options['surface']
        mesh = surface['mesh']
        ny = mesh.shape[1]

        # Add independent variables that do not belong to a specific component
        indep_var_comp = IndepVarComp()

        # Add structural components to the surface-specific group
        self.add_subsystem('indep_vars',
                 indep_var_comp,
                 promotes=['*'])

        if 'thickness_cp' in surface.keys():
            n_cp = len(surface['thickness_cp'])
            # Add bspline components for active bspline geometric variables.
            self.add_subsystem('thickness_bsp', BsplinesComp(
                in_name='thickness_cp', out_name='thickness', units='m',
                num_control_points=n_cp, num_points=int(ny-1),
                bspline_order=min(n_cp, 4), distribution='uniform'),
                promotes_inputs=['thickness_cp'], promotes_outputs=['thickness'])
            indep_var_comp.add_output('thickness_cp', val=surface['thickness_cp'], units='m')

        if 'radius_cp' in surface.keys():
            n_cp = len(surface['radius_cp'])
            # Add bspline components for active bspline geometric variables.
            self.add_subsystem('radius_bsp', BsplinesComp(
                in_name='radius_cp', out_name='radius', units='m',
                num_control_points=n_cp, num_points=int(ny-1),
                bspline_order=min(n_cp, 4), distribution='uniform'),
                promotes_inputs=['radius_cp'], promotes_outputs=['radius'])
            indep_var_comp.add_output('radius_cp', val=surface['radius_cp'], units='m')
        else:
            self.add_subsystem('radius_comp',
                RadiusComp(surface=surface),
                promotes_inputs=['mesh', 't_over_c'],
                promotes_outputs=['radius'])

        self.add_subsystem('tube',
            SectionPropertiesTube(surface=surface),
            promotes_inputs=['thickness', 'radius'],
            promotes_outputs=['A', 'Iy', 'Iz', 'J'])
