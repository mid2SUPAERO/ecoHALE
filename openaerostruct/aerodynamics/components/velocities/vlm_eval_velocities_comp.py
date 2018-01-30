from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent

from openaerostruct.utils.misc_utils import tile_sparse_jac


class VLMEvalVelocitiesComp(ExplicitComponent):

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

        system_size = 0

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            system_size += (num_points_x - 1) * (num_points_z - 1)

        self.system_size = system_size

        velocities_name = '{}_eval_velocities'.format(eval_name)

        self.add_input('circulations', shape=(num_nodes, system_size))
        self.add_output(velocities_name, shape=(num_nodes, num_eval_points, 3))

        circulations_indices = np.arange(system_size)
        velocities_indices = np.arange(num_eval_points * 3).reshape((num_eval_points, 3))

        rows = np.einsum('ik,j->ijk',
            velocities_indices, np.ones(system_size, int)).flatten()
        cols = np.einsum('ik,j->ijk',
            np.ones((num_eval_points, 3), int), circulations_indices).flatten()
        _, rows, cols = tile_sparse_jac(1., rows, cols,
            num_eval_points * 3, system_size, num_nodes)
        self.declare_partials(velocities_name, 'circulations', rows=rows, cols=cols)

        ind_1 = 0
        ind_2 = 0
        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1
            num = (num_points_x - 1) * (num_points_z - 1)

            ind_2 += num

            vel_mtx_name = '{}_{}_vel_mtx'.format(lifting_surface_name, eval_name)

            self.add_input(vel_mtx_name,
                shape=(num_nodes, num_eval_points, num_points_x - 1, num_points_z - 1, 3))

            vel_mtx_indices = np.arange(num_eval_points * num * 3).reshape(
                (num_eval_points, num, 3))

            rows = np.einsum('ik,j->ijk', velocities_indices, np.ones(num, int)).flatten()
            cols = vel_mtx_indices.flatten()
            _, rows, cols = tile_sparse_jac(1., rows, cols,
                num_eval_points * 3, num_eval_points * (num_points_x - 1) * (num_points_z - 1) * 3, num_nodes)
            self.declare_partials(velocities_name, vel_mtx_name, rows=rows, cols=cols)

            ind_1 += num

        self.set_check_partial_options('*', method='cs')

    def compute(self, inputs, outputs):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']
        eval_name = self.metadata['eval_name']
        num_eval_points = self.metadata['num_eval_points']

        system_size = self.system_size

        velocities_name = '{}_eval_velocities'.format(eval_name)

        outputs[velocities_name] = 0.

        ind_1 = 0
        ind_2 = 0
        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1
            num = (num_points_x - 1) * (num_points_z - 1)

            ind_2 += num

            vel_mtx_name = '{}_{}_vel_mtx'.format(lifting_surface_name, eval_name)

            outputs[velocities_name] += np.einsum('ijkl,ik->ijl',
                inputs[vel_mtx_name].reshape((num_nodes, num_eval_points, num, 3)),
                inputs['circulations'][:, ind_1:ind_2],
            )

            ind_1 += num

    def compute_partials(self, inputs, partials):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']
        eval_name = self.metadata['eval_name']
        num_eval_points = self.metadata['num_eval_points']

        system_size = self.system_size

        velocities_name = '{}_eval_velocities'.format(eval_name)

        dv_dcirc = partials[velocities_name, 'circulations'].reshape(
            (num_nodes, num_eval_points, system_size, 3))

        ind_1 = 0
        ind_2 = 0
        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1
            num = (num_points_x - 1) * (num_points_z - 1)

            ind_2 += num

            vel_mtx_name = '{}_{}_vel_mtx'.format(lifting_surface_name, eval_name)

            partials[velocities_name, vel_mtx_name] = np.einsum('jkl,ik->ijkl',
                np.ones((num_eval_points, num, 3)),
                inputs['circulations'][:, ind_1:ind_2],
            ).flatten()

            dv_dcirc[:, :, ind_1:ind_2, :] = inputs[vel_mtx_name].reshape(
                (num_nodes, num_eval_points, num, 3))

            ind_1 += num
