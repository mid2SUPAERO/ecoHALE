from __future__ import division, print_function
import numpy as np

from openmdao.api import ExplicitComponent

from openaerostruct.utils.misc_utils import get_array_indices, get_airfoils, tile_sparse_jac


def cartesian_vector(index, length):
    vec = np.zeros(length)
    vec[index] = 1.0
    return vec


class FEAVonmisesComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)
        self.metadata.declare('lifting_surfaces', types=list)

    def setup(self):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            local_disp_name = '{}_local_disp'.format(lifting_surface_name)
            radius_name = '{}_tube_radius'.format(lifting_surface_name)
            length_name = '{}_element_L'.format(lifting_surface_name)
            vonmises_name = '{}_vonmises'.format(lifting_surface_name)

            self.add_input(local_disp_name, shape=(num_nodes, num_points_z - 1, 12))
            self.add_input(radius_name, shape=(num_nodes, num_points_z - 1))
            self.add_input(length_name, shape=(num_nodes, num_points_z - 1))
            self.add_output(vonmises_name, shape=(num_nodes, num_points_z - 1, 2))

            local_disp_indices = get_array_indices(num_nodes, num_points_z - 1, 12)
            radius_indices = get_array_indices(num_nodes, num_points_z - 1)
            length_indices = get_array_indices(num_nodes, num_points_z - 1)
            vonmises_indices = get_array_indices(num_nodes, num_points_z - 1, 2)

            rows = np.einsum('ijk,l->ijkl', vonmises_indices, np.ones(12, int)).flatten()
            cols = np.einsum('ijl,k->ijkl', local_disp_indices, np.ones(2, int)).flatten()
            self.declare_partials(vonmises_name, local_disp_name, rows=rows, cols=cols)

            rows = vonmises_indices.flatten()
            cols = np.einsum('ij,k->ijk', radius_indices, np.ones(2, int)).flatten()
            self.declare_partials(vonmises_name, radius_name, rows=rows, cols=cols)

            rows = vonmises_indices.flatten()
            cols = np.einsum('ij,k->ijk', length_indices, np.ones(2, int)).flatten()
            self.declare_partials(vonmises_name, length_name, rows=rows, cols=cols)

        self.set_check_partial_options('*', method='cs')

    def compute(self, inputs, outputs):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            E = lifting_surface_data['E']
            G = lifting_surface_data['G']
            sigma_y = lifting_surface_data['sigma_y']

            local_disp_name = '{}_local_disp'.format(lifting_surface_name)
            radius_name = '{}_tube_radius'.format(lifting_surface_name)
            length_name = '{}_element_L'.format(lifting_surface_name)
            vonmises_name = '{}_vonmises'.format(lifting_surface_name)

            local_disp = inputs[local_disp_name]
            radius = inputs[radius_name]
            length = inputs[length_name]

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

            tmp = np.sqrt( (r1y - r0y) ** 2 + (r1z - r0z) ** 2 )
            sxx0 = E * (u1x - u0x) / length + E * radius / length * tmp
            sxx1 = E * (u0x - u1x) / length + E * radius / length * tmp
            sxt = G * radius * (r1x - r0x) / length
            vm0 = np.sqrt(sxx0 ** 2 + 3 * sxt ** 2)
            vm1 = np.sqrt(sxx1 ** 2 + 3 * sxt ** 2)

            outputs[vonmises_name][:, :, 0] = vm0 / sigma_y
            outputs[vonmises_name][:, :, 1] = vm1 / sigma_y

    def compute_partials(self, inputs, partials):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            E = lifting_surface_data['E']
            G = lifting_surface_data['G']
            sigma_y = lifting_surface_data['sigma_y']

            local_disp_name = '{}_local_disp'.format(lifting_surface_name)
            radius_name = '{}_tube_radius'.format(lifting_surface_name)
            length_name = '{}_element_L'.format(lifting_surface_name)
            vonmises_name = '{}_vonmises'.format(lifting_surface_name)

            local_disp = inputs[local_disp_name]
            radius = np.einsum('ij,k->ijk', inputs[radius_name], np.ones(12))
            length = np.einsum('ij,k->ijk', inputs[length_name], np.ones(12))

            u0x = np.einsum('ij,k->ijk', local_disp[:, :,  0], np.ones(12))
            u0y = np.einsum('ij,k->ijk', local_disp[:, :,  1], np.ones(12))
            u0z = np.einsum('ij,k->ijk', local_disp[:, :,  2], np.ones(12))
            r0x = np.einsum('ij,k->ijk', local_disp[:, :,  3], np.ones(12))
            r0y = np.einsum('ij,k->ijk', local_disp[:, :,  4], np.ones(12))
            r0z = np.einsum('ij,k->ijk', local_disp[:, :,  5], np.ones(12))
            u1x = np.einsum('ij,k->ijk', local_disp[:, :,  6], np.ones(12))
            u1y = np.einsum('ij,k->ijk', local_disp[:, :,  7], np.ones(12))
            u1z = np.einsum('ij,k->ijk', local_disp[:, :,  8], np.ones(12))
            r1x = np.einsum('ij,k->ijk', local_disp[:, :,  9], np.ones(12))
            r1y = np.einsum('ij,k->ijk', local_disp[:, :, 10], np.ones(12))
            r1z = np.einsum('ij,k->ijk', local_disp[:, :, 11], np.ones(12))

            tmp = np.sqrt( (r1y - r0y) ** 2 + (r1z - r0z) ** 2 )
            sxx0 = E * (u1x - u0x) / length + E * radius / length * tmp
            sxx1 = E * (u0x - u1x) / length + E * radius / length * tmp
            sxt = G * radius * (r1x - r0x) / length
            vm0 = np.sqrt(sxx0 ** 2 + 3 * sxt ** 2)
            vm1 = np.sqrt(sxx1 ** 2 + 3 * sxt ** 2)

            ones = np.ones((num_nodes, num_points_z - 1))
            deriv_u0x = np.einsum('ij,k->ijk', ones, cartesian_vector( 0, 12))
            deriv_u0y = np.einsum('ij,k->ijk', ones, cartesian_vector( 1, 12))
            deriv_u0z = np.einsum('ij,k->ijk', ones, cartesian_vector( 2, 12))
            deriv_r0x = np.einsum('ij,k->ijk', ones, cartesian_vector( 3, 12))
            deriv_r0y = np.einsum('ij,k->ijk', ones, cartesian_vector( 4, 12))
            deriv_r0z = np.einsum('ij,k->ijk', ones, cartesian_vector( 5, 12))
            deriv_u1x = np.einsum('ij,k->ijk', ones, cartesian_vector( 6, 12))
            deriv_u1y = np.einsum('ij,k->ijk', ones, cartesian_vector( 7, 12))
            deriv_u1z = np.einsum('ij,k->ijk', ones, cartesian_vector( 8, 12))
            deriv_r1x = np.einsum('ij,k->ijk', ones, cartesian_vector( 9, 12))
            deriv_r1y = np.einsum('ij,k->ijk', ones, cartesian_vector(10, 12))
            deriv_r1z = np.einsum('ij,k->ijk', ones, cartesian_vector(11, 12))

            deriv_tmp = 0.5 / tmp * ( 2 * (r1y - r0y) * (deriv_r1y - deriv_r0y) + 2 * (r1z - r0z) * (deriv_r1z - deriv_r0z) )
            deriv_sxx0 = E * (deriv_u1x - deriv_u0x) / length + E * radius / length * deriv_tmp
            deriv_sxx1 = E * (deriv_u0x - deriv_u1x) / length + E * radius / length * deriv_tmp
            deriv_sxt = G * radius * (deriv_r1x - deriv_r0x) / length
            deriv_vm0 = 0.5 / vm0 * (2 * sxx0 * deriv_sxx0 + 6 * sxt * deriv_sxt)
            deriv_vm1 = 0.5 / vm1 * (2 * sxx1 * deriv_sxx1 + 6 * sxt * deriv_sxt)

            derivs = partials[vonmises_name, local_disp_name].reshape((num_nodes, num_points_z - 1, 2, 12))
            derivs[:, :, 0, :] = deriv_vm0 / sigma_y
            derivs[:, :, 1, :] = deriv_vm1 / sigma_y

            # ---------------------------------------------------------------

            local_disp = inputs[local_disp_name]
            radius = inputs[radius_name]
            length = inputs[length_name]

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

            tmp = np.sqrt( (r1y - r0y) ** 2 + (r1z - r0z) ** 2 )
            sxx0 = E * (u1x - u0x) / length + E * radius / length * tmp
            sxx1 = E * (u0x - u1x) / length + E * radius / length * tmp
            sxt = G * radius * (r1x - r0x) / length
            vm0 = np.sqrt(sxx0 ** 2 + 3 * sxt ** 2)
            vm1 = np.sqrt(sxx1 ** 2 + 3 * sxt ** 2)

            # ---------------------

            deriv_sxx0 = E / length * tmp
            deriv_sxx1 = E / length * tmp
            deriv_sxt = G * (r1x - r0x) / length
            deriv_vm0 = 0.5 / np.sqrt(sxx0 ** 2 + 3 * sxt ** 2) * (2 * sxx0 * deriv_sxx0 + 6 * sxt * deriv_sxt)
            deriv_vm1 = 0.5 / np.sqrt(sxx1 ** 2 + 3 * sxt ** 2) * (2 * sxx1 * deriv_sxx1 + 6 * sxt * deriv_sxt)

            derivs = partials[vonmises_name, radius_name].reshape((num_nodes, num_points_z - 1, 2))
            derivs[:, :, 0] = deriv_vm0 / sigma_y
            derivs[:, :, 1] = deriv_vm1 / sigma_y

            # ---------------------

            deriv_sxx0 = -E * (u1x - u0x) / length ** 2 - E * radius / length ** 2 * tmp
            deriv_sxx1 = -E * (u0x - u1x) / length ** 2 - E * radius / length ** 2 * tmp
            deriv_sxt = -G * radius * (r1x - r0x) / length ** 2
            deriv_vm0 = 0.5 / np.sqrt(sxx0 ** 2 + 3 * sxt ** 2) * (2 * sxx0 * deriv_sxx0 + 6 * sxt * deriv_sxt)
            deriv_vm1 = 0.5 / np.sqrt(sxx1 ** 2 + 3 * sxt ** 2) * (2 * sxx1 * deriv_sxx1 + 6 * sxt * deriv_sxt)

            derivs = partials[vonmises_name, length_name].reshape((num_nodes, num_points_z - 1, 2))
            derivs[:, :, 0] = deriv_vm0 / sigma_y
            derivs[:, :, 1] = deriv_vm1 / sigma_y
