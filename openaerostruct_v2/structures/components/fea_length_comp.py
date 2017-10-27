from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent

from openaerostruct_v2.utils.vector_algebra import compute_norm


class FEALengthComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('lifting_surfaces', type_=list)

    def setup(self):
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            mesh_name = '{}_fea_mesh'.format(lifting_surface_name)
            length_name = '{}_element_{}'.format(lifting_surface_name, 'L')

            self.add_input(mesh_name, shape=(num_points_z, 3))
            self.add_output(length_name, shape=num_points_z - 1)

            mesh_indices = np.arange(3 * num_points_z).reshape((num_points_z, 3))
            length_indices = np.arange(num_points_z - 1)

            rows = np.tile(np.outer(length_indices, np.ones(3, int)).flatten(), 2)
            cols = np.concatenate([
                mesh_indices[:-1, :].flatten(),
                mesh_indices[1: , :].flatten(),
            ])
            self.declare_partials(length_name, mesh_name, rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            mesh_name = '{}_fea_mesh'.format(lifting_surface_name)
            length_name = '{}_element_{}'.format(lifting_surface_name, 'L')

            vec = inputs[mesh_name][1:, :] - inputs[mesh_name][:-1, :]

            outputs[length_name] = np.linalg.norm(vec, axis=-1)

    def compute_partials(self, inputs, partials):
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            mesh_name = '{}_fea_mesh'.format(lifting_surface_name)
            length_name = '{}_element_{}'.format(lifting_surface_name, 'L')

            vec = inputs[mesh_name][1:, :] - inputs[mesh_name][:-1, :]
            vec_deriv = np.einsum('i,jk->ijk', np.ones(num_points_z - 1), np.eye(3))

            derivs = partials[length_name, mesh_name].reshape((2, num_points_z - 1, 3))
            derivs[0, :, :] = -vec / compute_norm(vec)
            derivs[1, :, :] =  vec / compute_norm(vec)
