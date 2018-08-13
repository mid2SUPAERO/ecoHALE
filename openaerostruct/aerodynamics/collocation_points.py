from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent


non_singular_force_pts = False


class CollocationPoints(ExplicitComponent):

    def initialize(self):
        self.options.declare('surfaces', types=list)

    def setup(self):
        num_eval_points = 0

        for surface in self.options['surfaces']:
            nx = surface['num_x']
            ny = surface['num_y']

            num_eval_points += (nx - 1) * (ny - 1)

        self.add_output('coll_pts', shape=(num_eval_points, 3), units='m')
        self.add_output('force_pts', shape=(num_eval_points, 3), units='m')
        self.add_output('bound_vecs', shape=(num_eval_points, 3), units='m')

        eval_indices = np.arange(num_eval_points * 3).reshape((num_eval_points, 3))

        ind_eval_points_1 = 0
        ind_eval_points_2 = 0
        for surface in self.options['surfaces']:
            nx = surface['num_x']
            ny = surface['num_y']
            name = surface['name']

            ind_eval_points_2 += (nx - 1) * (ny - 1)

            mesh_name = name + '_def_mesh'
            self.add_input(mesh_name, shape=(nx, ny, 3), units='m')

            mesh_indices = np.arange(nx * ny * 3).reshape(
                (nx, ny, 3))

            rows = np.tile(eval_indices[ind_eval_points_1:ind_eval_points_2, :].flatten(), 4)
            cols = np.concatenate([
                mesh_indices[0:-1, 0:-1, :].flatten(),
                mesh_indices[1:  , 0:-1, :].flatten(),
                mesh_indices[0:-1, 1:  , :].flatten(),
                mesh_indices[1:  , 1:  , :].flatten(),
            ])

            data = np.concatenate([
                0.25 * 0.5 * np.ones((nx - 1) * (ny - 1) * 3),  # FR
                0.75 * 0.5 * np.ones((nx - 1) * (ny - 1) * 3),  # BR
                0.25 * 0.5 * np.ones((nx - 1) * (ny - 1) * 3),  # FL
                0.75 * 0.5 * np.ones((nx - 1) * (ny - 1) * 3),  # BL
            ])
            self.declare_partials('coll_pts', mesh_name, val=data, rows=rows, cols=cols)

            if non_singular_force_pts:
                data = np.concatenate([
                    0.5 * 0.5 * np.ones((nx - 1) * (ny - 1) * 3),  # FR
                    0.5 * 0.5 * np.ones((nx - 1) * (ny - 1) * 3),  # BR
                    0.5 * 0.5 * np.ones((nx - 1) * (ny - 1) * 3),  # FL
                    0.5 * 0.5 * np.ones((nx - 1) * (ny - 1) * 3),  # BL
                ])
            else:
                data = np.concatenate([
                    0.75 * 0.5 * np.ones((nx - 1) * (ny - 1) * 3),  # FR
                    0.25 * 0.5 * np.ones((nx - 1) * (ny - 1) * 3),  # BR
                    0.75 * 0.5 * np.ones((nx - 1) * (ny - 1) * 3),  # FL
                    0.25 * 0.5 * np.ones((nx - 1) * (ny - 1) * 3),  # BL
                ])
            self.declare_partials('force_pts', mesh_name, val=data, rows=rows, cols=cols)

            data = np.concatenate([
                 0.75 * np.ones((nx - 1) * (ny - 1) * 3),  # FR
                 0.25 * np.ones((nx - 1) * (ny - 1) * 3),  # BR
                -0.75 * np.ones((nx - 1) * (ny - 1) * 3),  # FL
                -0.25 * np.ones((nx - 1) * (ny - 1) * 3),  # BL
            ])
            self.declare_partials('bound_vecs', mesh_name, val=data, rows=rows, cols=cols)

            ind_eval_points_1 += (nx - 1) * (ny - 1)

    def compute(self, inputs, outputs):
        ind_eval_points_1 = 0
        ind_eval_points_2 = 0
        for surface in self.options['surfaces']:
            nx = surface['num_x']
            ny = surface['num_y']
            name = surface['name']

            ind_eval_points_2 += (nx - 1) * (ny - 1)

            mesh_name = name + '_def_mesh'

            outputs['coll_pts'][ind_eval_points_1:ind_eval_points_2, :] = (
                0.25 * 0.5 * inputs[mesh_name][0:-1, 0:-1, :] +
                0.75 * 0.5 * inputs[mesh_name][1:  , 0:-1, :] +
                0.25 * 0.5 * inputs[mesh_name][0:-1, 1:  , :] +
                0.75 * 0.5 * inputs[mesh_name][1:  , 1:  , :]
            ).reshape(((nx - 1) * (ny - 1), 3))

            if non_singular_force_pts:
                outputs['force_pts'][ind_eval_points_1:ind_eval_points_2, :] = (
                    0.5 * 0.5 * inputs[mesh_name][0:-1, 0:-1, :] +
                    0.5 * 0.5 * inputs[mesh_name][1:  , 0:-1, :] +
                    0.5 * 0.5 * inputs[mesh_name][0:-1, 1:  , :] +
                    0.5 * 0.5 * inputs[mesh_name][1:  , 1:  , :]
                ).reshape(((nx - 1) * (ny - 1), 3))
            else:
                outputs['force_pts'][ind_eval_points_1:ind_eval_points_2, :] = (
                    0.75 * 0.5 * inputs[mesh_name][0:-1, 0:-1, :] +
                    0.25 * 0.5 * inputs[mesh_name][1:  , 0:-1, :] +
                    0.75 * 0.5 * inputs[mesh_name][0:-1, 1:  , :] +
                    0.25 * 0.5 * inputs[mesh_name][1:  , 1:  , :]
                ).reshape(((nx - 1) * (ny - 1), 3))

            outputs['bound_vecs'][ind_eval_points_1:ind_eval_points_2, :] = (
                 0.75 * inputs[mesh_name][0:-1, 0:-1, :] +
                 0.25 * inputs[mesh_name][1:  , 0:-1, :] +
                -0.75 * inputs[mesh_name][0:-1, 1:  , :] +
                -0.25 * inputs[mesh_name][1:  , 1:  , :]
            ).reshape(((nx - 1) * (ny - 1), 3))

            ind_eval_points_1 += (nx - 1) * (ny - 1)
