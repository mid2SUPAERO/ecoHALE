from __future__ import print_function
import numpy as np

from openmdao.api import Group, IndepVarComp

from openaerostruct_v2.common.bspline_comp import BsplinesComp


class FEABsplineGroup(Group):

    def initialize(self):
        self.metadata.declare('lifting_surfaces', type_=list)

    def setup(self):
        lifting_surfaces = self.metadata['lifting_surfaces']

        default_bspline = (2, 2)

        comp = IndepVarComp()
        comp.add_output('fea_dummy_var')
        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            thickness = lifting_surface_data.get('thickness', None)
            radius = lifting_surface_data.get('radius', None)

            thickness_ncp, thickness_order = lifting_surface_data.get('thickness_bspline', default_bspline)
            radius_ncp, radius_order = lifting_surface_data.get('radius_bspline', default_bspline)

            name = '{}_tube_{}_cp'.format(lifting_surface_name, 'thickness')
            comp.add_output(name, val=thickness, shape=2 * thickness_ncp - 1)

            name = '{}_tube_{}_cp'.format(lifting_surface_name, 'radius')
            comp.add_output(name, val=radius, shape=2 * radius_ncp - 1)

        self.add_subsystem('indep_var_comp', comp, promotes=['*'])

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 2

            for name in ['thickness', 'radius']:
                ncp, bspline_order = lifting_surface_data.get(
                    '{}_bspline'.format(name), default_bspline)
                num_control_points = 2 * ncp - 1

                in_name = '{}_tube_{}_cp'.format(lifting_surface_name, name)
                out_name = '{}_tube_{}'.format(lifting_surface_name, name)
                comp = BsplinesComp(
                    num_control_points=num_control_points,
                    num_points=num_points_z,
                    bspline_order=bspline_order,
                    in_name=in_name,
                    out_name=out_name,
                    distribution='sine',
                )
                self.add_subsystem('{}_{}_bspline_comp'.format(lifting_surface_name, name), comp,
                    promotes=['*'])
