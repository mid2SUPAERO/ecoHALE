from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent

from openaerostruct_v2.utils.vector_algebra import add_ones_axis
from openaerostruct_v2.utils.vector_algebra import compute_norm, compute_norm_deriv
from openaerostruct_v2.utils.vector_algebra import compute_cross, compute_cross_deriv1, compute_cross_deriv2


class FEATransformComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('lifting_surfaces', type_=list)

    def setup(self):
        lifting_surfaces = self.metadata['lifting_surfaces']

        self.ref_axis = ref_axis = {}

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            mesh_name = '{}_mesh'.format(lifting_surface_name)
            transform_name = '{}_transform'.format(lifting_surface_name)

            self.add_input(mesh_name, shape=(num_points_z, 3))
            self.add_output(transform_name, shape=(num_points_z - 1, 12, 12))

            mesh_indices = np.arange(3 * num_points_z).reshape((num_points_z, 3))
            transform_indices = np.arange(144 * (num_points_z - 1)).reshape((num_points_z - 1, 12, 12))

            rows = np.concatenate([
                np.einsum('ijk,l->ijkl', transform_indices, np.ones(3, int)).flatten(),
                np.einsum('ijk,l->ijkl', transform_indices, np.ones(3, int)).flatten(),
            ])
            cols = np.concatenate([
                np.einsum('il,jk->ijkl', mesh_indices[:-1, :], np.ones((12, 12), int)).flatten(),
                np.einsum('il,jk->ijkl', mesh_indices[ 1:, :], np.ones((12, 12), int)).flatten(),
            ])
            self.declare_partials(transform_name, mesh_name, rows=rows, cols=cols)

            ref_axis[lifting_surface_name] = np.outer(np.ones(num_points_z - 1), np.array([1., 0., 0.]))

    def compute(self, inputs, outputs):
        lifting_surfaces = self.metadata['lifting_surfaces']

        ref_axis = self.ref_axis

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            mesh_name = '{}_mesh'.format(lifting_surface_name)
            transform_name = '{}_transform'.format(lifting_surface_name)

            P0 = inputs[mesh_name][:-1, :]
            P1 = inputs[mesh_name][ 1:, :]
            norm = compute_norm(P1 - P0)
            row0 = (P1 - P0) / norm

            cross = compute_cross(row0, ref_axis[lifting_surface_name])
            norm = compute_norm(cross)
            row1 = cross / norm

            cross = compute_cross(row0, row1)
            row2 = cross

            outputs[transform_name] = 0.
            for k in range(4):
                outputs[transform_name][:, 3*k + 0, 3*k : 3*k + 3] = row0
                outputs[transform_name][:, 3*k + 1, 3*k : 3*k + 3] = row1
                outputs[transform_name][:, 3*k + 2, 3*k : 3*k + 3] = row2

    def compute_partials(self, inputs, partials):
        lifting_surfaces = self.metadata['lifting_surfaces']

        ref_axis = self.ref_axis

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            mesh_name = '{}_mesh'.format(lifting_surface_name)
            transform_name = '{}_transform'.format(lifting_surface_name)

            P0 = inputs[mesh_name][:-1, :]
            P1 = inputs[mesh_name][ 1:, :]
            P_deriv = np.einsum('i,jk->ijk', np.ones(num_points_z - 1), np.eye(3))
            norm = compute_norm(P1 - P0)
            norm_deriv = compute_norm_deriv(P1 - P0, P_deriv)
            row0 = (P1 - P0) / norm
            row0_deriv = P_deriv / add_ones_axis(norm) - add_ones_axis(P1 - P0) / add_ones_axis(norm) ** 2 * norm_deriv

            cross = compute_cross(row0, ref_axis[lifting_surface_name])
            cross_deriv = compute_cross_deriv1(row0_deriv, ref_axis[lifting_surface_name])
            norm = compute_norm(cross)
            norm_deriv = compute_norm_deriv(cross, cross_deriv)
            row1 = cross / norm
            row1_deriv = cross_deriv / add_ones_axis(norm) - add_ones_axis(cross) / add_ones_axis(norm) ** 2 * norm_deriv

            cross = compute_cross(row0, row1)
            cross_deriv = (
                compute_cross_deriv1(row0_deriv, row1) +
                compute_cross_deriv2(row0, row1_deriv)
            )
            row2 = cross
            row2_deriv = cross_deriv

            derivs = partials[transform_name, mesh_name].reshape((2, num_points_z - 1, 12, 12, 3))

            for k in range(4):
                derivs[0, :, 3*k + 0, 3*k : 3*k + 3, :] = -row0_deriv
                derivs[1, :, 3*k + 0, 3*k : 3*k + 3, :] =  row0_deriv

                derivs[0, :, 3*k + 1, 3*k : 3*k + 3, :] = -row1_deriv
                derivs[1, :, 3*k + 1, 3*k : 3*k + 3, :] =  row1_deriv

                derivs[0, :, 3*k + 2, 3*k : 3*k + 3, :] = -row2_deriv
                derivs[1, :, 3*k + 2, 3*k : 3*k + 3, :] =  row2_deriv
