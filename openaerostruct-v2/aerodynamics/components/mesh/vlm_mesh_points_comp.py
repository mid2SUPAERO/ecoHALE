from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent


non_singular_force_pts = False


class VLMMeshPointsComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('lifting_surfaces', type_=list)

    def setup(self):
        lifting_surfaces = self.metadata['lifting_surfaces']

        num_eval_points = 0

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            num_eval_points += (num_points_x - 1) * (num_points_z - 1)

        self.add_output('coll_pts', shape=(num_eval_points, 3))
        self.add_output('force_pts', shape=(num_eval_points, 3))
        self.add_output('bound_vecs', shape=(num_eval_points, 3))

        eval_indices = np.arange(num_eval_points * 3).reshape((num_eval_points, 3))

        ind_eval_points_1 = 0
        ind_eval_points_2 = 0
        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            ind_eval_points_2 += (num_points_x - 1) * (num_points_z - 1)

            mesh_name = '{}_mesh'.format(lifting_surface_name)
            self.add_input(mesh_name, shape=(num_points_x, num_points_z, 3))

            mesh_indices = np.arange(num_points_x * num_points_z * 3).reshape(
                (num_points_x, num_points_z, 3))

            rows = np.tile(eval_indices[ind_eval_points_1:ind_eval_points_2, :].flatten(), 4)
            cols = np.concatenate([
                mesh_indices[0:-1, 0:-1, :].flatten(),
                mesh_indices[1:  , 0:-1, :].flatten(),
                mesh_indices[0:-1, 1:  , :].flatten(),
                mesh_indices[1:  , 1:  , :].flatten(),
            ])

            data = np.concatenate([
                0.25 * 0.5 * np.ones((num_points_x - 1) * (num_points_z - 1) * 3),  # FR
                0.75 * 0.5 * np.ones((num_points_x - 1) * (num_points_z - 1) * 3),  # BR
                0.25 * 0.5 * np.ones((num_points_x - 1) * (num_points_z - 1) * 3),  # FL
                0.75 * 0.5 * np.ones((num_points_x - 1) * (num_points_z - 1) * 3),  # BL
            ])
            self.declare_partials('coll_pts', mesh_name, val=data, rows=rows, cols=cols)

            if non_singular_force_pts:
                data = np.concatenate([
                    0.5 * 0.5 * np.ones((num_points_x - 1) * (num_points_z - 1) * 3),  # FR
                    0.5 * 0.5 * np.ones((num_points_x - 1) * (num_points_z - 1) * 3),  # BR
                    0.5 * 0.5 * np.ones((num_points_x - 1) * (num_points_z - 1) * 3),  # FL
                    0.5 * 0.5 * np.ones((num_points_x - 1) * (num_points_z - 1) * 3),  # BL
                ])
            else:
                data = np.concatenate([
                    0.75 * 0.5 * np.ones((num_points_x - 1) * (num_points_z - 1) * 3),  # FR
                    0.25 * 0.5 * np.ones((num_points_x - 1) * (num_points_z - 1) * 3),  # BR
                    0.75 * 0.5 * np.ones((num_points_x - 1) * (num_points_z - 1) * 3),  # FL
                    0.25 * 0.5 * np.ones((num_points_x - 1) * (num_points_z - 1) * 3),  # BL
                ])
            self.declare_partials('force_pts', mesh_name, val=data, rows=rows, cols=cols)

            data = np.concatenate([
                 0.75 * np.ones((num_points_x - 1) * (num_points_z - 1) * 3),  # FR
                 0.25 * np.ones((num_points_x - 1) * (num_points_z - 1) * 3),  # BR
                -0.75 * np.ones((num_points_x - 1) * (num_points_z - 1) * 3),  # FL
                -0.25 * np.ones((num_points_x - 1) * (num_points_z - 1) * 3),  # BL
            ])
            self.declare_partials('bound_vecs', mesh_name, val=data, rows=rows, cols=cols)

            ind_eval_points_1 += (num_points_x - 1) * (num_points_z - 1)

    def compute(self, inputs, outputs):
        lifting_surfaces = self.metadata['lifting_surfaces']

        ind_eval_points_1 = 0
        ind_eval_points_2 = 0
        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            ind_eval_points_2 += (num_points_x - 1) * (num_points_z - 1)

            mesh_name = '{}_mesh'.format(lifting_surface_name)

            outputs['coll_pts'][ind_eval_points_1:ind_eval_points_2, :] = (
                0.25 * 0.5 * inputs[mesh_name][0:-1, 0:-1, :] +
                0.75 * 0.5 * inputs[mesh_name][1:  , 0:-1, :] +
                0.25 * 0.5 * inputs[mesh_name][0:-1, 1:  , :] +
                0.75 * 0.5 * inputs[mesh_name][1:  , 1:  , :]
            ).reshape(((num_points_x - 1) * (num_points_z - 1), 3))

            if non_singular_force_pts:
                outputs['force_pts'][ind_eval_points_1:ind_eval_points_2, :] = (
                    0.5 * 0.5 * inputs[mesh_name][0:-1, 0:-1, :] +
                    0.5 * 0.5 * inputs[mesh_name][1:  , 0:-1, :] +
                    0.5 * 0.5 * inputs[mesh_name][0:-1, 1:  , :] +
                    0.5 * 0.5 * inputs[mesh_name][1:  , 1:  , :]
                ).reshape(((num_points_x - 1) * (num_points_z - 1), 3))
            else:
                outputs['force_pts'][ind_eval_points_1:ind_eval_points_2, :] = (
                    0.75 * 0.5 * inputs[mesh_name][0:-1, 0:-1, :] +
                    0.25 * 0.5 * inputs[mesh_name][1:  , 0:-1, :] +
                    0.75 * 0.5 * inputs[mesh_name][0:-1, 1:  , :] +
                    0.25 * 0.5 * inputs[mesh_name][1:  , 1:  , :]
                ).reshape(((num_points_x - 1) * (num_points_z - 1), 3))

            outputs['bound_vecs'][ind_eval_points_1:ind_eval_points_2, :] = (
                 0.75 * inputs[mesh_name][0:-1, 0:-1, :] +
                 0.25 * inputs[mesh_name][1:  , 0:-1, :] +
                -0.75 * inputs[mesh_name][0:-1, 1:  , :] +
                -0.25 * inputs[mesh_name][1:  , 1:  , :]
            ).reshape(((num_points_x - 1) * (num_points_z - 1), 3))

            ind_eval_points_1 += (num_points_x - 1) * (num_points_z - 1)
