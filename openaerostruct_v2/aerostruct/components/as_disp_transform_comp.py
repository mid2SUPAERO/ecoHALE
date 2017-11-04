from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent

from openaerostruct_v2.utils.misc_utils import get_array_indices, get_airfoils, tile_sparse_jac
from openaerostruct_v2.utils.vector_algebra import compute_cross, compute_cross_deriv1, compute_cross_deriv2


class ASDispTransformComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('num_nodes', type_=int)
        self.metadata.declare('lifting_surfaces', type_=list)

    def setup(self):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            disp_name = '{}_disp'.format(lifting_surface_name)
            transform_name = '{}_transform_mtx'.format(lifting_surface_name)

            self.add_input(disp_name, shape=(num_nodes, num_points_z, 6))
            self.add_output(transform_name, shape=(num_nodes, num_points_z, 3, 3))

            disp_indices = get_array_indices(num_points_z, 6)
            transform_indices = get_array_indices(num_points_z, 3, 3)

            rows = np.einsum('ijk,l->ijkl',
                transform_indices,
                np.ones(3, int)).flatten()
            cols = np.einsum('il,jk->ijkl',
                get_array_indices(num_points_z, 6)[:, 3:],
                np.ones((3, 3), int)).flatten()
            _, rows, cols = tile_sparse_jac(1., rows, cols,
                num_points_z * 3 * 3, num_points_z * 6, num_nodes)
            self.declare_partials(transform_name, disp_name, rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            disp_name = '{}_disp'.format(lifting_surface_name)
            transform_name = '{}_transform_mtx'.format(lifting_surface_name)

            outputs[transform_name] = 0.
            for i in range(3):
                outputs[transform_name][:, :, i, i] -= 2.

            rx = inputs[disp_name][:, :, 3]
            ry = inputs[disp_name][:, :, 4]
            rz = inputs[disp_name][:, :, 5]

            # T[ 1:,  1:] += [[cos(rx), -sin(rx)], [ sin(rx), cos(rx)]]
            outputs[transform_name][:, :, 1, 1] += np.cos(rx)
            outputs[transform_name][:, :, 1, 2] -= np.sin(rx)
            outputs[transform_name][:, :, 2, 1] += np.sin(rx)
            outputs[transform_name][:, :, 2, 2] += np.cos(rx)

            # T[::2, ::2] += [[cos(ry),  sin(ry)], [-sin(ry), cos(ry)]]
            outputs[transform_name][:, :, 0, 0] += np.cos(ry)
            outputs[transform_name][:, :, 0, 2] += np.sin(ry)
            outputs[transform_name][:, :, 2, 0] -= np.sin(ry)
            outputs[transform_name][:, :, 2, 2] += np.cos(ry)

            # T[ :2,  :2] += [[cos(rz), -sin(rz)], [ sin(rz), cos(rz)]]
            outputs[transform_name][:, :, 0, 0] += np.cos(rz)
            outputs[transform_name][:, :, 0, 1] -= np.sin(rz)
            outputs[transform_name][:, :, 1, 0] += np.sin(rz)
            outputs[transform_name][:, :, 1, 1] += np.cos(rz)

    def compute_partials(self, inputs, partials):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            disp_name = '{}_disp'.format(lifting_surface_name)
            transform_name = '{}_transform_mtx'.format(lifting_surface_name)

            rx = inputs[disp_name][:, :, 3]
            ry = inputs[disp_name][:, :, 4]
            rz = inputs[disp_name][:, :, 5]

            derivs = partials[transform_name, disp_name].reshape((num_nodes, num_points_z, 3, 3, 3))
            derivs[:, :, :, :, :] = 0.

            derivs[:, :, 1, 1, 0] -= np.sin(rx)
            derivs[:, :, 1, 2, 0] -= np.cos(rx)
            derivs[:, :, 2, 1, 0] += np.cos(rx)
            derivs[:, :, 2, 2, 0] -= np.sin(rx)

            derivs[:, :, 0, 0, 1] -= np.sin(ry)
            derivs[:, :, 0, 2, 1] += np.cos(ry)
            derivs[:, :, 2, 0, 1] -= np.cos(ry)
            derivs[:, :, 2, 2, 1] -= np.sin(ry)

            derivs[:, :, 0, 0, 2] -= np.sin(rz)
            derivs[:, :, 0, 1, 2] -= np.cos(rz)
            derivs[:, :, 1, 0, 2] += np.cos(rz)
            derivs[:, :, 1, 1, 2] -= np.sin(rz)
