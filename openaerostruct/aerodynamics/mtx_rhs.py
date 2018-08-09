from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent


class VLMMtxRHSComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('surfaces', types=list)

    def setup(self):
        surfaces = self.options['surfaces']

        system_size = 0

        for surface in surfaces:
            nx = surface['num_x']
            ny = surface['num_y']
            name = surface['name']

            system_size += (nx - 1) * (ny - 1)

        self.system_size = system_size

        self.add_input('inflow_velocities', shape=(system_size, 3), units='m/s')
        self.add_output('mtx', shape=(system_size, system_size), units='1/m')
        self.add_output('rhs', shape=system_size, units='m/s')

        inflow_indices = np.arange(system_size * 3).reshape((system_size, 3))
        mtx_indices = np.arange(system_size * system_size).reshape((system_size, system_size))
        rhs_indices = np.arange(system_size)

        self.declare_partials('*', '*', dependent=False)

        self.declare_partials('rhs', 'inflow_velocities',
            rows=np.einsum('i,j->ij', rhs_indices, np.ones(3, int)).flatten(),
            cols=inflow_indices.flatten()
        )

        ind_1 = 0
        ind_2 = 0
        for surface in surfaces:
            nx = surface['num_x']
            ny = surface['num_y']
            name = surface['name']
            num = (nx - 1) * (ny - 1)

            ind_2 += num

            vel_mtx_name = '{}_{}_vel_mtx'.format(name, 'coll_pts')
            normals_name = '{}_normals'.format(name)

            self.add_input(vel_mtx_name,
                shape=(system_size, nx - 1, ny - 1, 3), units='1/m')
            self.add_input(normals_name, shape=(nx - 1, ny - 1, 3))

            velocities_indices = np.arange(system_size * num * 3).reshape(
                (system_size, nx - 1, ny - 1, 3)
            )
            normals_indices = np.arange(num * 3).reshape((num, 3))

            self.declare_partials('mtx', vel_mtx_name,
                rows=np.einsum('ij,k->ijk', mtx_indices[:, ind_1:ind_2], np.ones(3, int)).flatten(),
                cols=velocities_indices.flatten(),
            )
            self.declare_partials('mtx', normals_name,
                rows=np.einsum('ij,k->ijk', mtx_indices[ind_1:ind_2, :], np.ones(3, int)).flatten(),
                cols=np.einsum('ik,j->ijk', normals_indices, np.ones(system_size, int)).flatten(),
            )
            self.declare_partials('rhs', normals_name,
                rows=np.outer(rhs_indices[ind_1:ind_2], np.ones(3, int)).flatten(),
                cols=normals_indices.flatten(),
            )

            ind_1 += num

        self.mtx_n_n_3 = np.zeros((system_size, system_size, 3))
        self.normals_n_3 = np.zeros((system_size, 3))
        self.set_check_partial_options(wrt='*', method='fd', step=1e-5)

    def compute(self, inputs, outputs):
        surfaces = self.options['surfaces']

        system_size = self.system_size

        ind_1 = 0
        ind_2 = 0
        for surface in surfaces:
            nx = surface['num_x']
            ny = surface['num_y']
            name = surface['name']
            num = (nx - 1) * (ny - 1)

            ind_2 += num

            vel_mtx_name = '{}_{}_vel_mtx'.format(name, 'coll_pts')
            normals_name = '{}_normals'.format(name)

            self.mtx_n_n_3[:, ind_1:ind_2, :] = inputs[vel_mtx_name].reshape((system_size, num, 3))
            self.normals_n_3[ind_1:ind_2, :] = inputs[normals_name].reshape((num, 3))

            ind_1 += num

        outputs['mtx'] = np.einsum('ijk,ik->ij', self.mtx_n_n_3, self.normals_n_3)
        outputs['rhs'] = -np.einsum('ij,ij->i', inputs['inflow_velocities'], self.normals_n_3)

    def compute_partials(self, inputs, partials):
        surfaces = self.options['surfaces']

        system_size = self.system_size

        ind_1 = 0
        ind_2 = 0
        for surface in surfaces:
            nx = surface['num_x']
            ny = surface['num_y']
            name = surface['name']
            num = (nx - 1) * (ny - 1)

            ind_2 += num

            vel_mtx_name = '{}_{}_vel_mtx'.format(name, 'coll_pts')
            normals_name = '{}_normals'.format(name)

            partials['mtx', vel_mtx_name] = np.einsum('ijk,ik->ijk',
                np.ones((system_size, num, 3)),
                self.normals_n_3,
            ).flatten()

            partials['mtx', normals_name] = self.mtx_n_n_3[ind_1:ind_2, :, :].flatten()

            partials['rhs', normals_name] = -inputs['inflow_velocities'][ind_1:ind_2, :].flatten()

            ind_1 += num

        partials['rhs', 'inflow_velocities'] = -self.normals_n_3.flatten()
