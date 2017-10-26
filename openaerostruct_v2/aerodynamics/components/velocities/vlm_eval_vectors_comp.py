from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent


class VLMEvalVectorsComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('lifting_surfaces', type_=list)
        self.metadata.declare('eval_name', type_=str)
        self.metadata.declare('num_eval_points', type_=int)

    def setup(self):
        lifting_surfaces = self.metadata['lifting_surfaces']
        eval_name = self.metadata['eval_name']
        num_eval_points = self.metadata['num_eval_points']

        self.declare_partials('*', '*', dependent=False)

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            mesh_name = '{}_vortex_mesh'.format(lifting_surface_name)
            vectors_name = '{}_{}_vectors'.format(lifting_surface_name, eval_name)

            self.add_input(eval_name, shape=(num_eval_points, 3))
            self.add_input(mesh_name, shape=(num_points_x, num_points_z, 3))
            self.add_output(vectors_name, shape=(num_eval_points, num_points_x, num_points_z, 3))

            vector_indices = np.arange(num_eval_points * num_points_x * num_points_z * 3)
            mesh_indices = np.outer(
                np.ones(num_eval_points, int),
                np.arange(num_points_x * num_points_z * 3),
            ).flatten()
            eval_indices = np.einsum('il,jk->ijkl',
                np.arange(num_eval_points * 3).reshape((num_eval_points, 3)),
                np.ones((num_points_x, num_points_z), int),
            ).flatten()

            self.declare_partials(vectors_name, mesh_name, val=-1., rows=vector_indices, cols=mesh_indices)
            self.declare_partials(vectors_name, eval_name, val= 1., rows=vector_indices, cols=eval_indices)

    def compute(self, inputs, outputs):
        lifting_surfaces = self.metadata['lifting_surfaces']
        eval_name = self.metadata['eval_name']
        num_eval_points = self.metadata['num_eval_points']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            mesh_name = '{}_vortex_mesh'.format(lifting_surface_name)
            vectors_name = '{}_{}_vectors'.format(lifting_surface_name, eval_name)

            mesh_reshaped = np.einsum('i,jkl->ijkl', np.ones(num_eval_points), inputs[mesh_name])
            eval_points_reshaped = np.einsum('il,jk->ijkl',
                inputs[eval_name],
                np.ones((num_points_x, num_points_z)),
            )

            outputs[vectors_name] = eval_points_reshaped - mesh_reshaped
