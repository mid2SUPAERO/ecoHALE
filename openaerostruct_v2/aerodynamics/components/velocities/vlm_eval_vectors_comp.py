from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent

from openaerostruct_v2.utils.misc_utils import tile_sparse_jac


class VLMEvalVectorsComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)
        self.metadata.declare('lifting_surfaces', types=list)
        self.metadata.declare('eval_name', types=str)
        self.metadata.declare('num_eval_points', types=int)

    def setup(self):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']
        eval_name = self.metadata['eval_name']
        num_eval_points = self.metadata['num_eval_points']

        self.declare_partials('*', '*', dependent=False)

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            mesh_name = '{}_vortex_mesh'.format(lifting_surface_name)
            vectors_name = '{}_{}_vectors'.format(lifting_surface_name, eval_name)

            self.add_input(eval_name, shape=(num_nodes, num_eval_points, 3))
            self.add_input(mesh_name, shape=(num_nodes, num_points_x, num_points_z, 3))
            self.add_output(vectors_name, shape=(num_nodes, num_eval_points, num_points_x, num_points_z, 3))

            vector_indices = np.arange(num_eval_points * num_points_x * num_points_z * 3)
            mesh_indices = np.outer(
                np.ones(num_eval_points, int),
                np.arange(num_points_x * num_points_z * 3),
            ).flatten()
            eval_indices = np.einsum('il,jk->ijkl',
                np.arange(num_eval_points * 3).reshape((num_eval_points, 3)),
                np.ones((num_points_x, num_points_z), int),
            ).flatten()

            _, rows, cols = tile_sparse_jac(1., vector_indices, mesh_indices,
                num_eval_points * num_points_x * num_points_z * 3,
                num_points_x * num_points_z * 3, num_nodes)
            self.declare_partials(vectors_name, mesh_name, val=-1., rows=rows, cols=cols)
            _, rows, cols = tile_sparse_jac(1., vector_indices, eval_indices,
                num_eval_points * num_points_x * num_points_z * 3,
                num_eval_points * 3, num_nodes)
            self.declare_partials(vectors_name, eval_name, val= 1., rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        lifting_surfaces = self.metadata['lifting_surfaces']
        eval_name = self.metadata['eval_name']
        num_eval_points = self.metadata['num_eval_points']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            mesh_name = '{}_vortex_mesh'.format(lifting_surface_name)
            vectors_name = '{}_{}_vectors'.format(lifting_surface_name, eval_name)

            mesh_reshaped = np.einsum('j,iklm->ijklm', np.ones(num_eval_points), inputs[mesh_name])
            eval_points_reshaped = np.einsum('ijm,kl->ijklm',
                inputs[eval_name],
                np.ones((num_points_x, num_points_z)),
            )

            outputs[vectors_name] = eval_points_reshaped - mesh_reshaped
