from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent


class EvalVelocities(ExplicitComponent):

    def initialize(self):
        self.options.declare('surfaces', types=list)
        self.options.declare('eval_name', types=str)
        self.options.declare('num_eval_points', types=int)

    def setup(self):
        surfaces = self.options['surfaces']
        eval_name = self.options['eval_name']
        num_eval_points = self.options['num_eval_points']

        system_size = 0

        for surface in surfaces:
            ny = surface['num_y']
            nx = surface['num_x']
            name = surface['name']

            system_size += (nx - 1) * (ny - 1)

        self.system_size = system_size

        velocities_name = '{}_velocities'.format(eval_name)

        self.add_input('inflow_velocities', shape=(system_size, 3), units='m/s')
        self.add_input('circulations', shape=system_size, units='m**2/s')
        self.add_output(velocities_name, shape=(num_eval_points, 3), units='m/s')

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
        for surface in surfaces:
            ny = surface['num_y']
            nx = surface['num_x']
            name = surface['name']
            num = (nx - 1) * (ny - 1)

            ind_2 += num

            vel_mtx_name = '{}_{}_vel_mtx'.format(name, eval_name)

            self.add_input(vel_mtx_name,
                shape=(num_eval_points, nx - 1, ny - 1, 3), units='1/m')

            vel_mtx_indices = np.arange(num_eval_points * num * 3).reshape(
                (num_eval_points, num, 3))

            self.declare_partials(velocities_name, vel_mtx_name,
                rows=np.einsum('ik,j->ijk', velocities_indices, np.ones(num, int)).flatten(),
                cols=vel_mtx_indices.flatten(),
            )

            ind_1 += num

    def compute(self, inputs, outputs):
        surfaces = self.options['surfaces']
        eval_name = self.options['eval_name']
        num_eval_points = self.options['num_eval_points']

        system_size = self.system_size

        velocities_name = '{}_velocities'.format(eval_name)

        outputs[velocities_name] = inputs['inflow_velocities']

        ind_1 = 0
        ind_2 = 0
        for surface in surfaces:
            ny = surface['num_y']
            nx = surface['num_x']
            name = surface['name']
            num = (nx - 1) * (ny - 1)

            ind_2 += num

            vel_mtx_name = '{}_{}_vel_mtx'.format(name, eval_name)

            outputs[velocities_name] += np.einsum('ijk,j->ik',
                inputs[vel_mtx_name].reshape((num_eval_points, num, 3)),
                inputs['circulations'][ind_1:ind_2],
            )

            tmp = np.einsum('ijk,j->ik',
                inputs[vel_mtx_name].reshape((num_eval_points, num, 3)),
                inputs['circulations'][ind_1:ind_2],
            )

            ind_1 += num

    def compute_partials(self, inputs, partials):
        surfaces = self.options['surfaces']
        eval_name = self.options['eval_name']
        num_eval_points = self.options['num_eval_points']

        system_size = self.system_size

        velocities_name = '{}_velocities'.format(eval_name)

        dv_dcirc = np.zeros((num_eval_points, system_size, 3))

        ind_1 = 0
        ind_2 = 0
        for surface in surfaces:
            ny = surface['num_y']
            nx = surface['num_x']
            name = surface['name']
            num = (nx - 1) * (ny - 1)

            ind_2 += num

            vel_mtx_name = '{}_{}_vel_mtx'.format(name, eval_name)

            partials[velocities_name, vel_mtx_name] = np.einsum('ijk,j->ijk',
                np.ones((num_eval_points, num, 3)),
                inputs['circulations'][ind_1:ind_2],
            ).flatten()

            dv_dcirc[:, ind_1:ind_2, :] = inputs[vel_mtx_name].reshape((num_eval_points, num, 3))

            ind_1 += num

        partials[velocities_name, 'circulations'] = dv_dcirc.flatten()
