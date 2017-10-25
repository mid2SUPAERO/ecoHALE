from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent


class VLMEvalVelocitiesComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('lifting_surfaces', type_=list)
        self.metadata.declare('eval_name', type_=str)
        self.metadata.declare('num_eval_points', type_=int)

    def setup(self):
        lifting_surfaces = self.metadata['lifting_surfaces']
        eval_name = self.metadata['eval_name']
        num_eval_points = self.metadata['num_eval_points']

        system_size = 0

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            system_size += (num_points_x - 1) * (num_points_z - 1)

        self.system_size = system_size

        velocities_name = '{}_velocities'.format(eval_name)

        self.add_input('inflow_velocities', shape=(system_size, 3))
        self.add_input('circulations', shape=system_size)
        self.add_output(velocities_name, shape=(num_eval_points, 3))

        circulations_indices = np.arange(system_size)
        velocities_indices = np.arange(num_eval_points * 3).reshape((num_eval_points, 3))

        self.declare_partials(velocities_name, 'circulations',
            rows=np.einsum('ik,j->ijk',
                velocities_indices, np.ones(system_size, int)).flatten(),
            cols=np.einsum('ik,j->ijk',
                np.ones((num_eval_points, 3), int), circulations_indices).flatten(),
        )

        self.declare_partials(velocities_name, 'inflow_velocities', val=1.,
            rows=np.arange(3 * num_eval_points),
            cols=np.arange(3 * num_eval_points),
        )

        ind_1 = 0
        ind_2 = 0
        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1
            num = (num_points_x - 1) * (num_points_z - 1)

            ind_2 += num

            vel_mtx_name = '{}_{}_vel_mtx'.format(lifting_surface_name, eval_name)

            self.add_input(vel_mtx_name,
                shape=(num_eval_points, num_points_x - 1, num_points_z - 1, 3))

            vel_mtx_indices = np.arange(num_eval_points * num * 3).reshape(
                (num_eval_points, num, 3))

            self.declare_partials(velocities_name, vel_mtx_name,
                rows=np.einsum('ik,j->ijk', velocities_indices, np.ones(num, int)).flatten(),
                cols=vel_mtx_indices.flatten(),
            )

            ind_1 += num

    def compute(self, inputs, outputs):
        lifting_surfaces = self.metadata['lifting_surfaces']
        eval_name = self.metadata['eval_name']
        num_eval_points = self.metadata['num_eval_points']

        system_size = self.system_size

        velocities_name = '{}_velocities'.format(eval_name)

        outputs[velocities_name] = inputs['inflow_velocities']

        ind_1 = 0
        ind_2 = 0
        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1
            num = (num_points_x - 1) * (num_points_z - 1)

            ind_2 += num

            vel_mtx_name = '{}_{}_vel_mtx'.format(lifting_surface_name, eval_name)

            outputs[velocities_name] += np.einsum('ijk,j->ik',
                inputs[vel_mtx_name].reshape((num_eval_points, num, 3)),
                inputs['circulations'][ind_1:ind_2],
            )

            ind_1 += num

    def compute_partials(self, inputs, partials):
        lifting_surfaces = self.metadata['lifting_surfaces']
        eval_name = self.metadata['eval_name']
        num_eval_points = self.metadata['num_eval_points']

        system_size = self.system_size

        velocities_name = '{}_velocities'.format(eval_name)

        dv_dcirc = np.zeros((num_eval_points, system_size, 3))

        ind_1 = 0
        ind_2 = 0
        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1
            num = (num_points_x - 1) * (num_points_z - 1)

            ind_2 += num

            vel_mtx_name = '{}_{}_vel_mtx'.format(lifting_surface_name, eval_name)

            partials[velocities_name, vel_mtx_name] = np.einsum('ijk,j->ijk',
                np.ones((num_eval_points, num, 3)),
                inputs['circulations'][ind_1:ind_2],
            ).flatten()

            dv_dcirc[:, ind_1:ind_2, :] = inputs[vel_mtx_name].reshape((num_eval_points, num, 3))

            ind_1 += num

        partials[velocities_name, 'circulations'] = dv_dcirc.flatten()
