from __future__ import division, print_function
import numpy as np

from openmdao.api import ExplicitComponent

from openaerostruct_v2.utils.misc_utils import get_array_indices, get_airfoils, tile_sparse_jac


def cartesian_vector(index, length):
    vec = np.zeros(length)
    vec[index] = 1.0
    return vec


class FEALocalDispComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)
        self.metadata.declare('lifting_surfaces', types=list)

    def setup(self):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            transform_name = '{}_transform'.format(lifting_surface_name)
            disp_name = '{}_disp'.format(lifting_surface_name)
            local_disp_name = '{}_local_disp'.format(lifting_surface_name)

            self.add_input(transform_name, shape=(num_nodes, num_points_z - 1, 12, 12))
            self.add_input(disp_name, shape=(num_nodes, num_points_z, 6))
            self.add_output(local_disp_name, shape=(num_nodes, num_points_z - 1, 12))

            transform_indices = get_array_indices(num_nodes, num_points_z - 1, 12, 12)
            disp_indices = get_array_indices(num_nodes, num_points_z, 6)
            local_disp_indices = get_array_indices(num_nodes, num_points_z - 1, 12)

            rows = np.einsum('ijk,l->ijkl', local_disp_indices, np.ones(12, int)).flatten()
            cols = transform_indices.flatten()
            self.declare_partials(local_disp_name, transform_name, rows=rows, cols=cols)

            rows = np.einsum('ijk,l->ijkl', local_disp_indices, np.ones(12, int)).flatten()
            cols = np.zeros((num_nodes, num_points_z - 1, 12, 12), int)
            cols[:, :, :, :6] = np.einsum('ijl,k->ijkl', disp_indices[:, :-1, :], np.ones(12, int))
            cols[:, :, :, 6:] = np.einsum('ijl,k->ijkl', disp_indices[:, 1: , :], np.ones(12, int))
            cols = cols.flatten()
            self.declare_partials(local_disp_name, disp_name, rows=rows, cols=cols)

        self.set_check_partial_options('*', method='cs')

    def compute(self, inputs, outputs):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            transform_name = '{}_transform'.format(lifting_surface_name)
            disp_name = '{}_disp'.format(lifting_surface_name)
            local_disp_name = '{}_local_disp'.format(lifting_surface_name)

            transform = inputs[transform_name]
            disp = np.empty((num_nodes, num_points_z - 1, 12, 12), dtype=inputs[disp_name].dtype)
            disp[:, :, :, :6] = np.einsum('ijl,k->ijkl', inputs[disp_name][:, :-1, :], np.ones((12)))
            disp[:, :, :, 6:] = np.einsum('ijl,k->ijkl', inputs[disp_name][:, 1: , :], np.ones((12)))

            outputs[local_disp_name] = np.sum(transform * disp, axis=-1)

    def compute_partials(self, inputs, partials):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            transform_name = '{}_transform'.format(lifting_surface_name)
            disp_name = '{}_disp'.format(lifting_surface_name)
            local_disp_name = '{}_local_disp'.format(lifting_surface_name)

            transform = inputs[transform_name]
            disp = np.empty((num_nodes, num_points_z - 1, 12, 12))
            disp[:, :, :, :6] = np.einsum('ijl,k->ijkl', inputs[disp_name][:, :-1, :], np.ones((12)))
            disp[:, :, :, 6:] = np.einsum('ijl,k->ijkl', inputs[disp_name][:, 1: , :], np.ones((12)))

            partials[local_disp_name, transform_name] = disp.flatten()
            partials[local_disp_name, disp_name] = transform.flatten()
