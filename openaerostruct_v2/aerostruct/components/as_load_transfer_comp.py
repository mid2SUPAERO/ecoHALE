from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent

from openaerostruct_v2.utils.misc_utils import get_array_indices
from openaerostruct_v2.utils.vector_algebra import compute_cross, compute_cross_deriv1, compute_cross_deriv2


class ASLoadTransferComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('lifting_surfaces', type_=list)

    def setup(self):
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            forces_name = '{}_panel_forces'.format(lifting_surface_name)
            vlm_mesh_name = '{}_mesh_cp'.format(lifting_surface_name)
            fea_mesh_name = '{}_fea_mesh'.format(lifting_surface_name)
            loads_name = '{}_loads'.format(lifting_surface_name)

            self.add_input(forces_name, shape=(num_points_x - 1, num_points_z - 1, 3))
            self.add_input(vlm_mesh_name, shape=(num_points_x - 1, num_points_z - 1, 3))
            self.add_input(fea_mesh_name, shape=(num_points_z, 3))
            self.add_output(loads_name, shape=(num_points_z, 6))

            forces_indices = get_array_indices(num_points_x - 1, num_points_z - 1, 3)
            mesh_indices = get_array_indices(num_points_x - 1, num_points_z - 1, 3)
            fea_mesh_indices = get_array_indices(num_points_z, 3)
            loads_indices = get_array_indices(num_points_z, 3)

            ones = np.ones(num_points_x - 1, int)
            ones2 = np.ones((num_points_x - 1, 3), int)
            rows = np.concatenate([
                np.einsum('jk,i->ijk', loads_indices[:-1, :3], ones).flatten(),
                np.einsum('jk,i->ijk', loads_indices[ 1:, :3], ones).flatten(),
                # np.einsum('jk,il->ijkl', loads_indices[:-1, 3:], ones2).flatten(),
                # np.einsum('jk,il->ijkl', loads_indices[ 1:, 3:], ones2).flatten(),
            ])
            cols = np.concatenate([
                forces_indices.flatten(),
                forces_indices.flatten(),
                # np.einsum('ijl,k->ijkl', forces_indices, np.ones(3, int)).flatten(),
                # np.einsum('ijl,k->ijkl', forces_indices, np.ones(3, int)).flatten(),
            ])
            self.declare_partials(loads_name, forces_name, val=0.5, rows=rows, cols=cols)

            # rows = np.concatenate([
            #     np.einsum('jk,il->ijkl',
            #         loads_indices[:-1, 3:], np.ones((num_points_x - 1, 3), int)).flatten(),
            #     np.einsum('jk,il->ijkl',
            #         loads_indices[ 1:, 3:], np.ones((num_points_x - 1, 3), int)).flatten(),
            # ])
            # cols = np.tile(np.einsum('ijl,k->ijkl', forces_indices, np.ones(3, int)).flatten(), 2)
            # self.declare_partials(loads_name, vlm_mesh_name, rows=rows, cols=cols)
            #
            # rows = np.einsum('ij,k->ijk', loads_indices[:, 3:], np.ones(3, int)).flatten()
            # cols = np.einsum('ik,j->ijk', fea_mesh_indices, np.ones(3, int)).flatten()
            # self.declare_partials(loads_name, fea_mesh_name, rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            forces_name = '{}_panel_forces'.format(lifting_surface_name)
            vlm_mesh_name = '{}_mesh_cp'.format(lifting_surface_name)
            fea_mesh_name = '{}_fea_mesh'.format(lifting_surface_name)
            loads_name = '{}_loads'.format(lifting_surface_name)

            outputs[loads_name] = 0.

            aero_pts = inputs[vlm_mesh_name]
            struct_pts = np.einsum('i,jk->ijk', np.ones(num_points_x - 1), inputs[fea_mesh_name][:-1, :])
            moments = compute_cross(aero_pts - struct_pts, inputs[forces_name])
            forces = inputs[forces_name]
            outputs[loads_name][:-1, :3] += 0.5 * np.sum( forces, axis=0)
            # outputs[loads_name][:-1, 3:] += 0.5 * np.sum(moments, axis=0)

            aero_pts = inputs[vlm_mesh_name]
            struct_pts = np.einsum('i,jk->ijk', np.ones(num_points_x - 1), inputs[fea_mesh_name][1:, :])
            moments = compute_cross(aero_pts - struct_pts, inputs[forces_name])
            forces = inputs[forces_name]
            outputs[loads_name][1: , :3] += 0.5 * np.sum( forces, axis=0)
            # outputs[loads_name][1: , 3:] += 0.5 * np.sum(moments, axis=0)

    # def compute_partials(self, inputs, partials):
    #     lifting_surfaces = self.metadata['lifting_surfaces']
    #
    #     for lifting_surface_name, lifting_surface_data in lifting_surfaces:
    #         num_points_x = lifting_surface_data['num_points_x']
    #         num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1
    #
    #         forces_name = '{}_panel_forces'.format(lifting_surface_name)
    #         vlm_mesh_name = '{}_mesh_cp'.format(lifting_surface_name)
    #         fea_mesh_name = '{}_fea_mesh'.format(lifting_surface_name)
    #         loads_name = '{}_loads'.format(lifting_surface_name)
    #
    #         ind1 = 0
    #         ind2 = (num_points_x - 1) * (num_points_z - 1) * 3
    #         derivs = partials[loads_name, forces_name][ind1:ind2].reshape((num_points_x - 1, num_points_z - 1, 3))
