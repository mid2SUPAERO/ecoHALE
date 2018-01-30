from __future__ import print_function
import numpy as np

from openmdao.api import Group, IndepVarComp

from openaerostruct.common.bspline_comp import BsplineComp
from openaerostruct.common.array_expansion_comp import ArrayExpansionComp


class FEABsplineGroup(Group):

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)
        self.metadata.declare('lifting_surfaces', types=list)

    def setup(self):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']

        default_bspline = (2, 2)

        initial_vals = {}
        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            thickness = lifting_surface_data.get('thickness', None)
            radius = lifting_surface_data.get('radius', None)

            thickness_ncp, thickness_order = lifting_surface_data.get('thickness_bspline', default_bspline)
            radius_ncp, radius_order = lifting_surface_data.get('radius_bspline', default_bspline)

            if thickness is not None:
                initial_vals[lifting_surface_name, 'thickness'] = thickness * np.ones(2 * thickness_ncp - 1)
            else:
                initial_vals[lifting_surface_name, 'thickness'] = None

            if radius is not None:
                initial_vals[lifting_surface_name, 'radius'] = radius * np.ones(2 * radius_ncp - 1)
            else:
                initial_vals[lifting_surface_name, 'radius'] = None

        comp = IndepVarComp()
        comp.add_output('fea_dummy_var')
        for lifting_surface_name, lifting_surface_data in lifting_surfaces:

            for name in ['thickness', 'radius']:
                val = initial_vals[lifting_surface_name, name]
                if val is not None:
                    name = '{}_tube_{}_dv'.format(lifting_surface_name, name)
                    comp.add_output(name, val=val)

        self.add_subsystem('indep_var_comp', comp, promotes=['*'])

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            for name in ['thickness', 'radius']:
                ncp, bspline_order = lifting_surface_data.get(
                    '{}_bspline'.format(name), default_bspline)
                num_control_points = 2 * ncp - 1

                in_name = '{}_tube_{}_dv'.format(lifting_surface_name, name)
                out_name = '{}_tube_{}_cp'.format(lifting_surface_name, name)
                comp = ArrayExpansionComp(
                    shape=(num_nodes, num_control_points),
                    expand_indices=[0],
                    in_name=in_name,
                    out_name=out_name,
                )

                val = initial_vals[lifting_surface_name, name]
                if val is not None:
                    self.add_subsystem('{}_{}_expand_cp_comp'.format(lifting_surface_name, name), comp,
                        promotes=['*'])

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            distribution = lifting_surface_data['distribution']

            for name in ['thickness', 'radius']:
                ncp, bspline_order = lifting_surface_data.get(
                    '{}_bspline'.format(name), default_bspline)
                num_control_points = 2 * ncp - 1

                in_name = '{}_tube_{}_cp'.format(lifting_surface_name, name)
                out_name = '{}_tube_{}'.format(lifting_surface_name, name)
                comp = BsplineComp(
                    num_nodes=num_nodes,
                    num_control_points=num_control_points,
                    num_points=num_points_z - 1,
                    bspline_order=bspline_order,
                    in_name=in_name,
                    out_name=out_name,
                    distribution=distribution,
                )
                self.add_subsystem('{}_{}_bspline_comp'.format(lifting_surface_name, name), comp,
                    promotes=['*'])
