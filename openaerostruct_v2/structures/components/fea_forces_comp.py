from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent


class FEAForcesComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('lifting_surfaces', type_=list)

    def setup(self):
        lifting_surfaces = self.metadata['lifting_surfaces']

        self.add_input('dummy')

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            size = 6 * num_points_z + 6

            forces_name = '{}_forces'.format(lifting_surface_name)

            self.add_output(forces_name, shape=size)

    def compute(self, inputs, outputs):
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            forces_name = '{}_forces'.format(lifting_surface_name)

            outputs[forces_name] = 0.
            outputs[forces_name][1:6 * num_points_z:6] = 1.
