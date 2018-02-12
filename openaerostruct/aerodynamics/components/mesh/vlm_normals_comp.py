from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent

from openaerostruct.utils.vector_algebra import add_ones_axis
from openaerostruct.utils.vector_algebra import compute_cross, compute_cross_deriv1, compute_cross_deriv2
from openaerostruct.utils.vector_algebra import compute_norm, compute_norm_deriv
from openaerostruct.utils.misc_utils import tile_sparse_jac


class VLMNormalsComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)
        self.metadata.declare('lifting_surfaces', types=list)

    def setup(self):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data.num_points_x
            num_points_z = 2 * lifting_surface_data.num_points_z_half - 1

            mesh_name = '{}_mesh'.format(lifting_surface_name)
            normals_name = '{}_normals'.format(lifting_surface_name)

            self.add_input(mesh_name, shape=(num_nodes, num_points_x, num_points_z, 3))
            self.add_output(normals_name, shape=(num_nodes, num_points_x - 1, num_points_z - 1, 3))

            mesh_indices = np.arange(num_points_x * num_points_z * 3).reshape(
                (num_points_x, num_points_z, 3))

            rows = np.tile(np.outer(
                np.arange((num_points_x - 1) * (num_points_z - 1) * 3),
                np.ones(3, int)
            ).flatten(), 4)
            cols = np.concatenate([
                np.einsum('ijl,k->ijkl', mesh_indices[0:-1, 0:-1, :], np.ones(3, int)).flatten(),
                np.einsum('ijl,k->ijkl', mesh_indices[1:  , 0:-1, :], np.ones(3, int)).flatten(),
                np.einsum('ijl,k->ijkl', mesh_indices[0:-1, 1:  , :], np.ones(3, int)).flatten(),
                np.einsum('ijl,k->ijkl', mesh_indices[1:  , 1:  , :], np.ones(3, int)).flatten(),
            ])
            _, rows, cols = tile_sparse_jac(1., rows, cols,
                (num_points_x - 1) * (num_points_z - 1) * 3,
                num_points_x * num_points_z * 3, num_nodes)
            self.declare_partials(normals_name, mesh_name, rows=rows, cols=cols)

        self.set_check_partial_options('*', method='cs')

    def compute(self, inputs, outputs):
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data.num_points_x
            num_points_z = 2 * lifting_surface_data.num_points_z_half - 1

            mesh_name = '{}_mesh'.format(lifting_surface_name)
            normals_name = '{}_normals'.format(lifting_surface_name)

            fr = inputs[mesh_name][:, 0:-1, 0:-1, :]
            br = inputs[mesh_name][:, 1:  , 0:-1, :]
            fl = inputs[mesh_name][:, 0:-1, 1:  , :]
            bl = inputs[mesh_name][:, 1:  , 1:  , :]

            cross = compute_cross(fl - br, bl - fr)
            norm = compute_norm(cross)
            outputs[normals_name] = cross / norm

    def compute_partials(self, inputs, partials):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data.num_points_x
            num_points_z = 2 * lifting_surface_data.num_points_z_half - 1

            mesh_name = '{}_mesh'.format(lifting_surface_name)
            normals_name = '{}_normals'.format(lifting_surface_name)

            fr = inputs[mesh_name][:, 0:-1, 0:-1, :]
            br = inputs[mesh_name][:, 1:  , 0:-1, :]
            fl = inputs[mesh_name][:, 0:-1, 1:  , :]
            bl = inputs[mesh_name][:, 1:  , 1:  , :]

            cross = compute_cross(fl - br, bl - fr)
            norm = compute_norm(cross)

            cross_ones = add_ones_axis(cross)
            norm_ones = add_ones_axis(norm)

            deriv_array = np.einsum('ijk,lm->ijklm',
                np.ones((num_nodes, num_points_x - 1, num_points_z - 1)),
                np.eye(3))

            deriv_cross_fr = compute_cross_deriv2(fl - br, -deriv_array)
            deriv_cross_br = compute_cross_deriv1(-deriv_array, bl - fr)
            deriv_cross_fl = compute_cross_deriv1( deriv_array, bl - fr)
            deriv_cross_bl = compute_cross_deriv2(fl - br,  deriv_array)

            deriv_norm_fr = compute_norm_deriv(cross, deriv_cross_fr)
            deriv_norm_br = compute_norm_deriv(cross, deriv_cross_br)
            deriv_norm_fl = compute_norm_deriv(cross, deriv_cross_fl)
            deriv_norm_bl = compute_norm_deriv(cross, deriv_cross_bl)

            derivs = partials[normals_name, mesh_name].reshape(
                (num_nodes, 4, num_points_x - 1, num_points_z - 1, 3, 3))
            derivs[:, 0, :, :, :] = (deriv_cross_fr * norm_ones - cross_ones * deriv_norm_fr) / norm_ones ** 2
            derivs[:, 1, :, :, :] = (deriv_cross_br * norm_ones - cross_ones * deriv_norm_br) / norm_ones ** 2
            derivs[:, 2, :, :, :] = (deriv_cross_fl * norm_ones - cross_ones * deriv_norm_fl) / norm_ones ** 2
            derivs[:, 3, :, :, :] = (deriv_cross_bl * norm_ones - cross_ones * deriv_norm_bl) / norm_ones ** 2
