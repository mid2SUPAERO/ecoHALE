from __future__ import division, print_function
import numpy as np

from openmdao.api import ExplicitComponent

from openaerostruct_v2.utils.misc_utils import get_array_indices, get_airfoils, tile_sparse_jac

def norm(vec):
    return np.sqrt(np.sum(vec**2))

def unit(vec):
    return vec / norm(vec)


class FEAVonmisesComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)
        self.metadata.declare('lifting_surfaces', types=list)

    def setup(self):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            transform_name = '{}_transform'.format(lifting_surface_name)
            radius_name = '{}_tube_radius'.format(lifting_surface_name)
            disp_name = '{}_disp'.format(lifting_surface_name)
            length_name = '{}_element_L'.format(lifting_surface_name)
            vonmises_name = '{}_vonmises'.format(lifting_surface_name)

            self.add_input(transform_name, shape=(num_nodes, num_points_z - 1, 12, 12))
            self.add_input(radius_name, shape=(num_nodes, num_points_z - 1))
            self.add_input(disp_name, shape=(num_nodes, num_points_z, 6))
            self.add_input(length_name, shape=(num_nodes, num_points_z - 1))
            self.add_output(vonmises_name, shape=(num_nodes, num_points_z - 1, 2))

            transform_indices = get_array_indices(num_nodes, num_points_z - 1, 12, 12)
            radius_indices = get_array_indices(num_nodes, num_points_z - 1)
            disp_indices = get_array_indices(num_nodes, num_points_z, 6)
            length_indices = get_array_indices(num_nodes, num_points_z - 1)
            vonmises_indices = get_array_indices(num_nodes, num_points_z - 1, 2)

            rows = np.einsum('ijk,lm->ijklm', vonmises_indices, np.ones((12, 12), int))
            cols = np.einsum('ijlm,k->ijklm', transform_indices, np.ones(2, int))
            self.declare_partials(vonmises_name, transform_name, rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1
            E = lifting_surface_data['E']
            G = lifting_surface_data['G']

            transform_name = '{}_transform'.format(lifting_surface_name)
            radius_name = '{}_tube_radius'.format(lifting_surface_name)
            disp_name = '{}_disp'.format(lifting_surface_name)
            length_name = '{}_element_L'.format(lifting_surface_name)
            vonmises_name = '{}_vonmises'.format(lifting_surface_name)

            transform = inputs[transform_name]
            radius = inputs[radius_name]
            disp = np.empty((num_nodes, num_points_z - 1, 12))
            disp[:, :, :6] = inputs[disp_name][:, :-1, :]
            disp[:, :, 6:] = inputs[disp_name][:, 1: , :]
            length = inputs[length_name]

            local_disp = np.einsum('ijkl,ijl->ijk', transform, disp)
            u0x = local_disp[:, :,  0]
            u0y = local_disp[:, :,  1]
            u0z = local_disp[:, :,  2]
            r0x = local_disp[:, :,  3]
            r0y = local_disp[:, :,  4]
            r0z = local_disp[:, :,  5]
            u1x = local_disp[:, :,  6]
            u1y = local_disp[:, :,  7]
            u1z = local_disp[:, :,  8]
            r1x = local_disp[:, :,  9]
            r1y = local_disp[:, :, 10]
            r1z = local_disp[:, :, 11]

            tmp = np.sqrt((r1y - r0y)**2 + (r1z - r0z)**2)
            sxx0 = E * (u1x - u0x) / length + E * radius / length * tmp
            sxx1 = E * (u0x - u1x) / length + E * radius / length * tmp
            sxt = G * radius * (r1x - r0x) / length

            outputs[vonmises_name][:, :, 0] = np.sqrt(sxx0 ** 2 + 3 * sxt ** 2)
            outputs[vonmises_name][:, :, 1] = np.sqrt(sxx1 ** 2 + 3 * sxt ** 2)

    def compute_partials(self, inputs, partials):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1
            E = lifting_surface_data['E']
            G = lifting_surface_data['G']

            transform_name = '{}_transform'.format(lifting_surface_name)
            radius_name = '{}_tube_radius'.format(lifting_surface_name)
            disp_name = '{}_disp'.format(lifting_surface_name)
            length_name = '{}_element_L'.format(lifting_surface_name)
            vonmises_name = '{}_vonmises'.format(lifting_surface_name)

            transform = inputs[transform_name].flatten()
            radius = inputs[radius_name].flatten()
            disp = np.empty((num_nodes, num_points_z - 1, 12))
            disp[:, :, :6] = inputs[disp_name][:, :-1, :]
            disp[:, :, 6:] = inputs[disp_name][:, 1: , :]
            disp = disp.flatten()
            length = inputs[length_name].flatten()
