from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent


class FEADispComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('lifting_surfaces', type_=list)

    def setup(self):
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            size = 6 * num_points_z + 6

            states_name = '{}_states'.format(lifting_surface_name)
            disp_name = '{}_disp'.format(lifting_surface_name)

            self.add_input(states_name, shape=size)
            self.add_output(disp_name, shape=(num_points_z, 6))

            arange = np.arange(6 * num_points_z)
            self.declare_partials(disp_name, states_name, val=1., rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            size = 6 * num_points_z + 6

            states_name = '{}_states'.format(lifting_surface_name)
            disp_name = '{}_disp'.format(lifting_surface_name)

            outputs[disp_name] = inputs[states_name][:6 * num_points_z].reshape((num_points_z, 6))
