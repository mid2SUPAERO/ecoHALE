from __future__ import print_function, division
import numpy as np

from openmdao.api import ExplicitComponent


class GetVectors(ExplicitComponent):

    def initialize(self):
        self.options.declare('surfaces', types=list)
        self.options.declare('num_eval_points', types=int)
        self.options.declare('eval_name', types=str)

    def setup(self):
        surfaces = self.options['surfaces']
        num_eval_points = self.options['num_eval_points']
        eval_name = self.options['eval_name']

        self.declare_partials('*', '*', dependent=False)

        self.add_input(eval_name, val=np.zeros((num_eval_points, 3)), units='m')
        for surface in surfaces:
            ny = surface['num_y']
            nx = surface['num_x']
            name = surface['name']
            vectors_name = '{}_{}_vectors'.format(name, eval_name)

            if surface['symmetry']:
                actual_ny_size = ny * 2 - 1
            else:
                actual_ny_size = ny

            self.add_input(name + '_vortex_mesh', val=np.zeros((nx, actual_ny_size, 3)), units='m')
            self.add_output(vectors_name, val=np.ones((num_eval_points, nx, actual_ny_size, 3)), units='m')

            vector_indices = np.arange(num_eval_points * nx * actual_ny_size * 3)
            mesh_indices = np.outer(
                np.ones(num_eval_points, int),
                np.arange(nx * actual_ny_size * 3),
            ).flatten()
            eval_indices = np.einsum('il,jk->ijkl',
                np.arange(num_eval_points * 3).reshape((num_eval_points, 3)),
                np.ones((nx, actual_ny_size), int),
            ).flatten()

            self.declare_partials(vectors_name, name + '_vortex_mesh', val=-1., rows=vector_indices, cols=mesh_indices)
            self.declare_partials(vectors_name, eval_name, val= 1., rows=vector_indices, cols=eval_indices)

    def compute(self, inputs, outputs):
        surfaces = self.options['surfaces']
        num_eval_points = self.options['num_eval_points']
        eval_name = self.options['eval_name']

        for surface in surfaces:
            ny = surface['num_y']
            nx = surface['num_x']
            name = surface['name']

            mesh_name = name + '_vortex_mesh'
            vectors_name = '{}_{}_vectors'.format(name, eval_name)

            mesh_reshaped = np.einsum('i,jkl->ijkl', np.ones(num_eval_points), inputs[mesh_name])

            if surface['symmetry']:
                eval_points_reshaped = np.einsum('il,jk->ijkl',
                    inputs[eval_name],
                    np.ones((nx, 2*ny-1)),
                )
            else:
                eval_points_reshaped = np.einsum('il,jk->ijkl',
                    inputs[eval_name],
                    np.ones((nx, ny)),
                )

            outputs[vectors_name] = eval_points_reshaped - mesh_reshaped
