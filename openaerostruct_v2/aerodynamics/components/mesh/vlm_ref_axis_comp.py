from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent

from openaerostruct_v2.utils.misc_utils import get_array_indices, get_airfoils


class VLMRefAxisComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('lifting_surfaces', type_=list)

    def setup(self):
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            x_name = '{}_{}'.format(lifting_surface_name, 'sec_x')
            y_name = '{}_{}'.format(lifting_surface_name, 'sec_y')
            z_name = '{}_{}'.format(lifting_surface_name, 'sec_z')
            out_name = '{}_ref_axis'.format(lifting_surface_name)

            self.add_input(x_name, shape=num_points_z)
            self.add_input(y_name, shape=num_points_z)
            self.add_input(z_name, shape=num_points_z)
            self.add_output(out_name, shape=(num_points_z, 3))

            out_indices = get_array_indices(num_points_z, 3)
            in_indices = np.arange(num_points_z)
            self.declare_partials(out_name, x_name, val=1., rows=out_indices[:, 0], cols=in_indices)
            self.declare_partials(out_name, y_name, val=1., rows=out_indices[:, 1], cols=in_indices)
            self.declare_partials(out_name, z_name, val=1., rows=out_indices[:, 2], cols=in_indices)

    def compute(self, inputs, outputs):
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            x_name = '{}_{}'.format(lifting_surface_name, 'sec_x')
            y_name = '{}_{}'.format(lifting_surface_name, 'sec_y')
            z_name = '{}_{}'.format(lifting_surface_name, 'sec_z')
            out_name = '{}_ref_axis'.format(lifting_surface_name)

            outputs[out_name][:, 0] = inputs[x_name]
            outputs[out_name][:, 1] = inputs[y_name]
            outputs[out_name][:, 2] = inputs[z_name]
