from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent


class FEAVolumeComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('lifting_surfaces', type_=list)

    def setup(self):
        lifting_surfaces = self.metadata['lifting_surfaces']

        self.add_output('structural_volume')

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            A_name = '{}_element_{}'.format(lifting_surface_name, 'A')
            L_name = '{}_element_{}'.format(lifting_surface_name, 'L')

            self.add_input(A_name, shape=num_points_z - 1)
            self.add_input(L_name, shape=num_points_z - 1)

            self.declare_partials('structural_volume', A_name)
            self.declare_partials('structural_volume', L_name)

    def compute(self, inputs, outputs):
        lifting_surfaces = self.metadata['lifting_surfaces']

        outputs['structural_volume'] = 0.

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            A_name = '{}_element_{}'.format(lifting_surface_name, 'A')
            L_name = '{}_element_{}'.format(lifting_surface_name, 'L')

            outputs['structural_volume'] += np.dot(inputs[A_name], inputs[L_name])

    def compute_partials(self, inputs, partials):
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            A_name = '{}_element_{}'.format(lifting_surface_name, 'A')
            L_name = '{}_element_{}'.format(lifting_surface_name, 'L')

            partials['structural_volume', A_name][0, :] = inputs[L_name]
            partials['structural_volume', L_name][0, :] = inputs[A_name]
