from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent


class FEAComplianceComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('lifting_surfaces', type_=list)

    def setup(self):
        lifting_surfaces = self.metadata['lifting_surfaces']

        self.add_output('compliance')

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            size = 6 * num_points_z + 6

            forces_name = '{}_forces'.format(lifting_surface_name)
            states_name = '{}_states'.format(lifting_surface_name)

            self.add_input(forces_name, shape=size)
            self.add_input(states_name, shape=size)

            self.declare_partials('compliance', forces_name)
            self.declare_partials('compliance', states_name)

    def compute(self, inputs, outputs):
        lifting_surfaces = self.metadata['lifting_surfaces']

        outputs['compliance'] = 0.

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            forces_name = '{}_forces'.format(lifting_surface_name)
            states_name = '{}_states'.format(lifting_surface_name)

            outputs['compliance'] += np.dot(inputs[forces_name], inputs[states_name])

    def compute_partials(self, inputs, partials):
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            forces_name = '{}_forces'.format(lifting_surface_name)
            states_name = '{}_states'.format(lifting_surface_name)

            partials['compliance', forces_name][0, :] = inputs[states_name]
            partials['compliance', states_name][0, :] = inputs[forces_name]
