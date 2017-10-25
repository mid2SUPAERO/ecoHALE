from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent

from dep_mdo.utils.vector_algebra import add_ones_axis
from dep_mdo.utils.vector_algebra import compute_dot, compute_dot_deriv
from dep_mdo.utils.vector_algebra import compute_cross, compute_cross_deriv1, compute_cross_deriv2
from dep_mdo.utils.vector_algebra import compute_norm, compute_norm_deriv


def compute_finite_vortex(r1, r2):
    r1_norm = compute_norm(r1)
    r2_norm = compute_norm(r2)

    r1_x_r2 = compute_cross(r1, r2)
    r1_d_r2 = compute_dot(r1, r2)

    num = (1. / r1_norm + 1. / r2_norm) * r1_x_r2
    den = r1_norm * r2_norm + r1_d_r2

    result = num / den / 4 / np.pi
    result[den == 0] = 0.
    return result

def compute_finite_vortex_deriv1(r1, r2, r1_deriv):
    r1_norm = add_ones_axis(compute_norm(r1))
    r2_norm = add_ones_axis(compute_norm(r2))
    r1_norm_deriv = compute_norm_deriv(r1, r1_deriv)

    r1_x_r2 = add_ones_axis(compute_cross(r1, r2))
    r1_d_r2 = add_ones_axis(compute_dot(r1, r2))
    r1_x_r2_deriv = compute_cross_deriv1(r1_deriv, r2)
    r1_d_r2_deriv = compute_dot_deriv(r2, r1_deriv)

    num = (1. / r1_norm + 1. / r2_norm) * r1_x_r2
    num_deriv = (-r1_norm_deriv / r1_norm ** 2) * r1_x_r2 \
        + (1. / r1_norm + 1. / r2_norm) * r1_x_r2_deriv

    den = r1_norm * r2_norm + r1_d_r2
    den_deriv = r1_norm_deriv * r2_norm + r1_d_r2_deriv

    # return num / den / 4 / np.pi

    result = (num_deriv * den - num * den_deriv) / den ** 2 / 4 / np.pi
    result[den == 0] = 0.
    return result

def compute_finite_vortex_deriv2(r1, r2, r2_deriv):
    r1_norm = add_ones_axis(compute_norm(r1))
    r2_norm = add_ones_axis(compute_norm(r2))
    r2_norm_deriv = compute_norm_deriv(r2, r2_deriv)

    r1_x_r2 = add_ones_axis(compute_cross(r1, r2))
    r1_d_r2 = add_ones_axis(compute_dot(r1, r2))
    r1_x_r2_deriv = compute_cross_deriv2(r1, r2_deriv)
    r1_d_r2_deriv = compute_dot_deriv(r1, r2_deriv)

    num = (1. / r1_norm + 1. / r2_norm) * r1_x_r2
    num_deriv = (-r2_norm_deriv / r2_norm ** 2) * r1_x_r2 \
        + (1. / r1_norm + 1. / r2_norm) * r1_x_r2_deriv

    den = r1_norm * r2_norm + r1_d_r2
    den_deriv = r1_norm * r2_norm_deriv + r1_d_r2_deriv

    # return num / den / 4 / np.pi

    result = (num_deriv * den - num * den_deriv) / den ** 2 / 4 / np.pi
    result[den == 0] = 0.
    return result

def compute_semi_infinite_vortex(u, r):
    r_norm = compute_norm(r)
    u_x_r = compute_cross(u, r)
    u_d_r = compute_dot(u, r)

    num = u_x_r
    den = r_norm * (r_norm - u_d_r)
    return num / den / 4 / np.pi

def compute_semi_infinite_vortex_deriv(u, r, r_deriv):
    r_norm = add_ones_axis(compute_norm(r))
    r_norm_deriv = compute_norm_deriv(r, r_deriv)

    u_x_r = add_ones_axis(compute_cross(u, r))
    u_x_r_deriv = compute_cross_deriv2(u, r_deriv)

    u_d_r = add_ones_axis(compute_dot(u, r))
    u_d_r_deriv = compute_dot_deriv(u, r_deriv)

    num = u_x_r
    num_deriv = u_x_r_deriv

    den = r_norm * (r_norm - u_d_r)
    den_deriv = r_norm_deriv * (r_norm - u_d_r) + r_norm * (r_norm_deriv - u_d_r_deriv)

    # return num / den / 4 / np.pi
    return (num_deriv * den - num * den_deriv) / den ** 2 / 4 / np.pi


class VLMEvalVelMtxComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('lifting_surfaces', type_=list)
        self.metadata.declare('eval_name', type_=str)
        self.metadata.declare('num_eval_points', type_=int)

    def setup(self):
        lifting_surfaces = self.metadata['lifting_surfaces']
        eval_name = self.metadata['eval_name']
        num_eval_points = self.metadata['num_eval_points']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            vectors_name = '{}_{}_vectors'.format(lifting_surface_name, eval_name)
            vel_mtx_name = '{}_{}_vel_mtx'.format(lifting_surface_name, eval_name)

            self.add_input(vectors_name,
                shape=(num_eval_points, num_points_x, num_points_z, 3))
            self.add_output(vel_mtx_name,
                shape=(num_eval_points, num_points_x - 1, num_points_z - 1, 3))

            vectors_indices = np.arange(num_eval_points * num_points_x * num_points_z * 3).reshape(
                (num_eval_points, num_points_x, num_points_z, 3))
            vel_mtx_indices = np.arange(num_eval_points * (num_points_x - 1) * (num_points_z - 1) * 3).reshape(
                (num_eval_points, num_points_x - 1, num_points_z - 1, 3))

            rows = np.concatenate([
                np.einsum('ijkl,m->ijklm', vel_mtx_indices, np.ones(3, int)).flatten(),
                np.einsum('ijkl,m->ijklm', vel_mtx_indices, np.ones(3, int)).flatten(),
                np.einsum('ijkl,m->ijklm', vel_mtx_indices, np.ones(3, int)).flatten(),
                np.einsum('ijkl,m->ijklm', vel_mtx_indices, np.ones(3, int)).flatten(),
            ])
            cols = np.concatenate([
                np.einsum('ijkm,l->ijklm', vectors_indices[:, 0:-1, 0:-1, :], np.ones(3, int)).flatten(),
                np.einsum('ijkm,l->ijklm', vectors_indices[:, 1:  , 0:-1, :], np.ones(3, int)).flatten(),
                np.einsum('ijkm,l->ijklm', vectors_indices[:, 0:-1, 1:  , :], np.ones(3, int)).flatten(),
                np.einsum('ijkm,l->ijklm', vectors_indices[:, 1:  , 1:  , :], np.ones(3, int)).flatten(),
            ])
            self.declare_partials(vel_mtx_name, vectors_name, rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        lifting_surfaces = self.metadata['lifting_surfaces']
        eval_name = self.metadata['eval_name']
        num_eval_points = self.metadata['num_eval_points']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            u = np.einsum('ijk,l->ijkl',
                np.ones((num_eval_points, 1, num_points_z - 1)),
                np.array([1., 0., 0.]))

            vectors_name = '{}_{}_vectors'.format(lifting_surface_name, eval_name)
            vel_mtx_name = '{}_{}_vel_mtx'.format(lifting_surface_name, eval_name)

            outputs[vel_mtx_name] = 0.

            # front vortex
            r1 = inputs[vectors_name][:, 0:-1, 1:  , :]
            r2 = inputs[vectors_name][:, 0:-1, 0:-1, :]
            outputs[vel_mtx_name] += compute_finite_vortex(r1, r2)

            # right vortex
            r1 = inputs[vectors_name][:, 0:-1, 0:-1, :]
            r2 = inputs[vectors_name][:, 1:  , 0:-1, :]
            outputs[vel_mtx_name] += compute_finite_vortex(r1, r2)

            # rear vortex
            r1 = inputs[vectors_name][:, 1:  , 0:-1, :]
            r2 = inputs[vectors_name][:, 1:  , 1:  , :]
            outputs[vel_mtx_name] += compute_finite_vortex(r1, r2)

            # left vortex
            r1 = inputs[vectors_name][:, 1:  , 1:  , :]
            r2 = inputs[vectors_name][:, 0:-1, 1:  , :]
            outputs[vel_mtx_name] += compute_finite_vortex(r1, r2)

            # ----------------- last row -----------------

            r1 = inputs[vectors_name][:, -1:, 1:  , :]
            r2 = inputs[vectors_name][:, -1:, 0:-1, :]
            outputs[vel_mtx_name][:, -1:, :, :] += compute_finite_vortex(r1, r2)
            outputs[vel_mtx_name][:, -1:, :, :] -= compute_semi_infinite_vortex(u, r1)
            outputs[vel_mtx_name][:, -1:, :, :] += compute_semi_infinite_vortex(u, r2)

    def compute_partials(self, inputs, partials):
        lifting_surfaces = self.metadata['lifting_surfaces']
        eval_name = self.metadata['eval_name']
        num_eval_points = self.metadata['num_eval_points']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            u = np.einsum('ijk,l->ijkl',
                np.ones((num_eval_points, 1, num_points_z - 1)),
                np.array([1., 0., 0.]))

            vectors_name = '{}_{}_vectors'.format(lifting_surface_name, eval_name)
            vel_mtx_name = '{}_{}_vel_mtx'.format(lifting_surface_name, eval_name)

            deriv_array = np.einsum('...,ij->...ij',
                np.ones((num_eval_points, num_points_x - 1, num_points_z - 1)),
                np.eye(3))
            trailing_array = np.einsum('...,ij->...ij',
                np.ones((num_eval_points, 1, num_points_z - 1)),
                np.eye(3))

            derivs = np.zeros((4, num_eval_points, num_points_x - 1, num_points_z - 1, 3, 3))

            # front vortex
            r1 = inputs[vectors_name][:, 0:-1, 1:  , :]
            r2 = inputs[vectors_name][:, 0:-1, 0:-1, :]
            derivs[2, :, :, :, :] += compute_finite_vortex_deriv1(r1, r2, deriv_array)
            derivs[0, :, :, :, :] += compute_finite_vortex_deriv2(r1, r2, deriv_array)

            # right vortex
            r1 = inputs[vectors_name][:, 0:-1, 0:-1, :]
            r2 = inputs[vectors_name][:, 1:  , 0:-1, :]
            derivs[0, :, :, :, :] += compute_finite_vortex_deriv1(r1, r2, deriv_array)
            derivs[1, :, :, :, :] += compute_finite_vortex_deriv2(r1, r2, deriv_array)

            # rear vortex
            r1 = inputs[vectors_name][:, 1:  , 0:-1, :]
            r2 = inputs[vectors_name][:, 1:  , 1:  , :]
            derivs[1, :, :, :, :] += compute_finite_vortex_deriv1(r1, r2, deriv_array)
            derivs[3, :, :, :, :] += compute_finite_vortex_deriv2(r1, r2, deriv_array)

            # left vortex
            r1 = inputs[vectors_name][:, 1:  , 1:  , :]
            r2 = inputs[vectors_name][:, 0:-1, 1:  , :]
            derivs[3, :, :, :, :] += compute_finite_vortex_deriv1(r1, r2, deriv_array)
            derivs[2, :, :, :, :] += compute_finite_vortex_deriv2(r1, r2, deriv_array)

            # ----------------- last row -----------------

            r1 = inputs[vectors_name][:, -1:, 1:  , :]
            r2 = inputs[vectors_name][:, -1:, 0:-1, :]
            derivs[3, :, -1:, :, :] += compute_finite_vortex_deriv1(r1, r2, trailing_array)
            derivs[1, :, -1:, :, :] += compute_finite_vortex_deriv2(r1, r2, trailing_array)
            derivs[3, :, -1:, :, :] -= compute_semi_infinite_vortex_deriv(u, r1, trailing_array)
            derivs[1, :, -1:, :, :] += compute_semi_infinite_vortex_deriv(u, r2, trailing_array)

            partials[vel_mtx_name, vectors_name] = derivs.flatten()
