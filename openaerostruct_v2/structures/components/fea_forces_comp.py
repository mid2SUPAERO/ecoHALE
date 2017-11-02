from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent


class FEAForcesComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('lifting_surfaces', type_=list)

    def setup(self):
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            size = 6 * num_points_z + 6

            loads_name = '{}_loads'.format(lifting_surface_name)
            forces_name = '{}_forces'.format(lifting_surface_name)

            self.add_input(loads_name, shape=(num_points_z, 6))
            self.add_output(forces_name, shape=size, val=0.)

            arange = np.arange(6 * num_points_z)
            self.declare_partials(forces_name, loads_name, rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            loads_name = '{}_loads'.format(lifting_surface_name)
            forces_name = '{}_forces'.format(lifting_surface_name)

            outputs[forces_name][:6 * num_points_z] = inputs[loads_name].reshape((6 * num_points_z))
