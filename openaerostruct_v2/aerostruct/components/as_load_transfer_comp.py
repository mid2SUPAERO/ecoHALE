from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent

from openaerostruct_v2.utils.misc_utils import get_array_indices, tile_sparse_jac
from openaerostruct_v2.utils.vector_algebra import compute_cross, compute_cross_deriv1, compute_cross_deriv2


class ASLoadTransferComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)
        self.metadata.declare('lifting_surfaces', types=list)

    def setup(self):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            forces_name = '{}_panel_forces'.format(lifting_surface_name)
            vlm_mesh_name = '{}_mesh_cp'.format(lifting_surface_name)
            fea_mesh_name = '{}_fea_mesh'.format(lifting_surface_name)
            loads_name = '{}_loads'.format(lifting_surface_name)

            self.add_input(forces_name, shape=(num_nodes, num_points_x - 1, num_points_z - 1, 3))
            self.add_input(vlm_mesh_name, shape=(num_nodes, num_points_x - 1, num_points_z - 1, 3))
            self.add_input(fea_mesh_name, shape=(num_nodes, num_points_z, 3))
            self.add_output(loads_name, shape=(num_nodes, num_points_z, 6))

            forces_indices = get_array_indices(num_points_x - 1, num_points_z - 1, 3)
            vlm_mesh_indices = get_array_indices(num_points_x - 1, num_points_z - 1, 3)
            fea_mesh_indices = get_array_indices(num_points_z, 3)
            loads_indices = get_array_indices(num_points_z, 6)

            ones = np.ones(num_points_x - 1, int)
            ones2 = np.ones((num_points_x - 1, 3), int)
            rows = np.concatenate([
                np.einsum('i,jk->ijk', ones, loads_indices[:-1, :3]).flatten(),
                np.einsum('i,jk->ijk', ones, loads_indices[ 1:, :3]).flatten(),
                np.einsum('jk,il->ijkl', loads_indices[:-1, 3:], ones2).flatten(),
                np.einsum('jk,il->ijkl', loads_indices[ 1:, 3:], ones2).flatten(),
            ])
            cols = np.concatenate([
                forces_indices.flatten(),
                forces_indices.flatten(),
                np.einsum('ijl,k->ijkl', forces_indices, np.ones(3, int)).flatten(),
                np.einsum('ijl,k->ijkl', forces_indices, np.ones(3, int)).flatten(),
            ])
            _, rows, cols = tile_sparse_jac(1., rows, cols,
                num_points_z * 6, (num_points_x - 1) * (num_points_z - 1) * 3, num_nodes)
            self.declare_partials(loads_name, forces_name, val=0.5, rows=rows, cols=cols)

            rows = np.concatenate([
                np.einsum('jk,il->ijkl',
                    loads_indices[:-1, 3:], np.ones((num_points_x - 1, 3), int)).flatten(),
                np.einsum('jk,il->ijkl',
                    loads_indices[ 1:, 3:], np.ones((num_points_x - 1, 3), int)).flatten(),
            ])
            cols = np.tile(np.einsum('ijl,k->ijkl', vlm_mesh_indices, np.ones(3, int)).flatten(), 2)
            _, rows, cols = tile_sparse_jac(1., rows, cols,
                num_points_z * 6, (num_points_x - 1) * (num_points_z - 1) * 3, num_nodes)
            self.declare_partials(loads_name, vlm_mesh_name, rows=rows, cols=cols)

            rows = np.concatenate([
                np.einsum('jk,il->ijkl',
                    loads_indices[:-1, 3:], np.ones((num_points_x - 1, 3), int)).flatten(),
                np.einsum('jk,il->ijkl',
                    loads_indices[ 1:, 3:], np.ones((num_points_x - 1, 3), int)).flatten(),
            ])
            cols = np.concatenate([
                np.einsum('jl,ik->ijkl',
                    fea_mesh_indices[:-1, :], np.ones((num_points_x - 1, 3), int)).flatten(),
                np.einsum('jl,ik->ijkl',
                    fea_mesh_indices[ 1:, :], np.ones((num_points_x - 1, 3), int)).flatten(),
            ])
            _, rows, cols = tile_sparse_jac(1., rows, cols,
                num_points_z * 6, num_points_z * 3, num_nodes)
            self.declare_partials(loads_name, fea_mesh_name, rows=rows, cols=cols)

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
            struct_pts = np.einsum('j,ikl->ijkl',
                np.ones(num_points_x - 1), inputs[fea_mesh_name][:, :-1, :])
            moments = compute_cross(aero_pts - struct_pts, inputs[forces_name])
            forces = inputs[forces_name]
            outputs[loads_name][:, :-1, :3] += 0.5 * np.sum( forces, axis=1)
            outputs[loads_name][:, :-1, 3:] += 0.5 * np.sum(moments, axis=1)

            aero_pts = inputs[vlm_mesh_name]
            struct_pts = np.einsum('j,ikl->ijkl',
                np.ones(num_points_x - 1), inputs[fea_mesh_name][:, 1:, :])
            moments = compute_cross(aero_pts - struct_pts, inputs[forces_name])
            forces = inputs[forces_name]
            outputs[loads_name][:, 1: , :3] += 0.5 * np.sum( forces, axis=1)
            outputs[loads_name][:, 1: , 3:] += 0.5 * np.sum(moments, axis=1)

    def compute_partials(self, inputs, partials):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            forces_name = '{}_panel_forces'.format(lifting_surface_name)
            vlm_mesh_name = '{}_mesh_cp'.format(lifting_surface_name)
            fea_mesh_name = '{}_fea_mesh'.format(lifting_surface_name)
            loads_name = '{}_loads'.format(lifting_surface_name)

            deriv_array = np.einsum('...,ij->...ij',
                np.ones((num_nodes, num_points_x - 1, num_points_z - 1)),
                np.eye(3))

            # ------------------------------------------------------------------------------------

            size = 2 * (num_points_x - 1) * (num_points_z - 1) * 3 + 2 * (num_points_x - 1) * (num_points_z - 1) * 3 * 3
            derivs0 = partials[loads_name, forces_name].reshape((num_nodes, size))

            # -----

            aero_pts = inputs[vlm_mesh_name]
            struct_pts = np.einsum('j,ikl->ijkl',
                np.ones(num_points_x - 1), inputs[fea_mesh_name][:, :-1, :])

            ind1 = 2 * (num_points_x - 1) * (num_points_z - 1) * 3 + 0 * (num_points_x - 1) * (num_points_z - 1) * 3 * 3
            ind2 = 2 * (num_points_x - 1) * (num_points_z - 1) * 3 + 1 * (num_points_x - 1) * (num_points_z - 1) * 3 * 3

            derivs = derivs0[:, ind1:ind2].reshape(
                (num_nodes, num_points_x - 1, num_points_z - 1, 3, 3))
            derivs[:, :, :, :, :] = 0.5 * compute_cross_deriv2(aero_pts - struct_pts, deriv_array)

            aero_pts = inputs[vlm_mesh_name]
            struct_pts = np.einsum('j,ikl->ijkl',
                np.ones(num_points_x - 1), inputs[fea_mesh_name][:, 1:, :])

            ind1 = 2 * (num_points_x - 1) * (num_points_z - 1) * 3 + 1 * (num_points_x - 1) * (num_points_z - 1) * 3 * 3
            ind2 = 2 * (num_points_x - 1) * (num_points_z - 1) * 3 + 2 * (num_points_x - 1) * (num_points_z - 1) * 3 * 3

            derivs = derivs0[:, ind1:ind2].reshape(
                (num_nodes, num_points_x - 1, num_points_z - 1, 3, 3))
            derivs[:, :, :, :, :] = 0.5 * compute_cross_deriv2(aero_pts - struct_pts, deriv_array)

            # ------------------------------------------------------------------------------------

            derivs = partials[loads_name, vlm_mesh_name].reshape((num_nodes, 2, num_points_x - 1, num_points_z - 1, 3, 3))
            derivs[:, 0, :, :, :, :] = 0.5 * compute_cross_deriv1(deriv_array, inputs[forces_name])
            derivs[:, 1, :, :, :, :] = 0.5 * compute_cross_deriv1(deriv_array, inputs[forces_name])

            # ------------------------------------------------------------------------------------

            derivs = partials[loads_name, fea_mesh_name].reshape((num_nodes, 2, num_points_x - 1, num_points_z - 1, 3, 3))
            derivs[:, 0, :, :, :, :] = -0.5 * compute_cross_deriv1(deriv_array, inputs[forces_name])
            derivs[:, 1, :, :, :, :] = -0.5 * compute_cross_deriv1(deriv_array, inputs[forces_name])
