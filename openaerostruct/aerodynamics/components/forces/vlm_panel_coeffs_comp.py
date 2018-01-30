from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent

from openaerostruct.utils.vector_algebra import compute_cross, compute_cross_deriv1, compute_cross_deriv2

from openaerostruct.utils.misc_utils import get_array_indices, tile_sparse_jac


class VLMPanelCoeffsComp(ExplicitComponent):
    """
    Lift and drag coefficients by section, by surface.
    """

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)
        self.metadata.declare('lifting_surfaces', types=list)

    def setup(self):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']

        system_size = 0

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            system_size += (num_points_x - 1) * (num_points_z - 1)

        self.system_size = system_size

        self.add_input('rho_kg_m3', shape=num_nodes)
        self.add_input('v_m_s', shape=num_nodes)
        self.add_input('panel_forces_rotated', shape=(num_nodes, system_size, 3))

        panel_forces_rotated_arange = get_array_indices(system_size, 3)

        ind1, ind2 = 0, 0
        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1
            num = (num_points_x - 1) * (num_points_z - 1)

            sec_C_L_name = '{}_sec_C_L'.format(lifting_surface_name)
            sec_C_D_name = '{}_sec_C_D'.format(lifting_surface_name)
            sec_areas_name = '{}_sec_areas_m2'.format(lifting_surface_name)

            ind2 += num

            self.add_input(sec_areas_name, shape=(num_nodes, num_points_z - 1))
            self.add_output(sec_C_L_name, shape=(num_nodes, num_points_z - 1))
            self.add_output(sec_C_D_name, shape=(num_nodes, num_points_z - 1))

            rows = np.arange(num_points_z - 1)
            cols = np.zeros(num_points_z - 1, int)
            _, rows, cols = tile_sparse_jac(1., rows, cols, num_points_z - 1, 1, num_nodes)
            self.declare_partials(sec_C_L_name, 'rho_kg_m3', rows=rows, cols=cols)

            rows = np.arange(num_points_z - 1)
            cols = np.zeros(num_points_z - 1, int)
            _, rows, cols = tile_sparse_jac(1., rows, cols, num_points_z - 1, 1, num_nodes)
            self.declare_partials(sec_C_L_name, 'v_m_s', rows=rows, cols=cols)

            rows = np.outer(
                np.ones(num_points_x - 1, int),
                np.arange(num_points_z - 1),
            ).flatten()
            cols = panel_forces_rotated_arange[ind1:ind2, 1]
            _, rows, cols = tile_sparse_jac(1., rows, cols, num_points_z - 1, system_size * 3, num_nodes)
            self.declare_partials(sec_C_L_name, 'panel_forces_rotated', rows=rows, cols=cols)

            rows = np.arange(num_points_z - 1)
            cols = np.arange(num_points_z - 1)
            _, rows, cols = tile_sparse_jac(1., rows, cols, num_points_z - 1, num_points_z - 1, num_nodes)
            self.declare_partials(sec_C_L_name, sec_areas_name, rows=rows, cols=cols)

            rows = np.arange(num_points_z - 1)
            cols = np.zeros(num_points_z - 1, int)
            _, rows, cols = tile_sparse_jac(1., rows, cols, num_points_z - 1, 1, num_nodes)
            self.declare_partials(sec_C_D_name, 'rho_kg_m3', rows=rows, cols=cols)

            rows = np.arange(num_points_z - 1)
            cols = np.zeros(num_points_z - 1, int)
            _, rows, cols = tile_sparse_jac(1., rows, cols, num_points_z - 1, 1, num_nodes)
            self.declare_partials(sec_C_D_name, 'v_m_s', rows=rows, cols=cols)

            rows = np.outer(
                np.ones(num_points_x - 1, int),
                np.arange(num_points_z - 1),
            ).flatten()
            cols = panel_forces_rotated_arange[ind1:ind2, 0]
            _, rows, cols = tile_sparse_jac(1., rows, cols, num_points_z - 1, system_size * 3, num_nodes)
            self.declare_partials(sec_C_D_name, 'panel_forces_rotated', rows=rows, cols=cols)

            rows = np.arange(num_points_z - 1)
            cols = np.arange(num_points_z - 1)
            _, rows, cols = tile_sparse_jac(1., rows, cols, num_points_z - 1, num_points_z - 1, num_nodes)
            self.declare_partials(sec_C_D_name, sec_areas_name, rows=rows, cols=cols)

            ind1 += num

        self.set_check_partial_options('*', method='cs')

    def compute(self, inputs, outputs):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']

        ind1, ind2 = 0, 0
        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1
            num = (num_points_x - 1) * (num_points_z - 1)

            sec_C_L_name = '{}_sec_C_L'.format(lifting_surface_name)
            sec_C_D_name = '{}_sec_C_D'.format(lifting_surface_name)
            sec_areas_name = '{}_sec_areas_m2'.format(lifting_surface_name)

            ones = np.ones(num_points_z - 1)
            rho_kg_m3 = np.outer(inputs['rho_kg_m3'], ones)
            v_m_s = np.outer(inputs['v_m_s'], ones)

            ind2 += num

            panel_lift = inputs['panel_forces_rotated'][:, ind1:ind2, 1].reshape(
                (num_nodes, num_points_x - 1, num_points_z - 1))
            panel_drag = inputs['panel_forces_rotated'][:, ind1:ind2, 0].reshape(
                (num_nodes, num_points_x - 1, num_points_z - 1))

            outputs[sec_C_L_name] = np.sum(panel_lift, axis=1) \
                / (0.5 * rho_kg_m3 * v_m_s ** 2 * inputs[sec_areas_name])
            outputs[sec_C_D_name] = np.sum(panel_drag, axis=1) \
                / (0.5 * rho_kg_m3 * v_m_s ** 2 * inputs[sec_areas_name])

            ind1 += num

    def compute_partials(self, inputs, partials):
        num_nodes = self.metadata['num_nodes']

        lifting_surfaces = self.metadata['lifting_surfaces']

        # panel_lift = inputs['panel_forces_rotated'][:, :, 1]
        # panel_drag = inputs['panel_forces_rotated'][:, :, 0]

        ind1, ind2 = 0, 0
        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1
            num = (num_points_x - 1) * (num_points_z - 1)

            sec_C_L_name = '{}_sec_C_L'.format(lifting_surface_name)
            sec_C_D_name = '{}_sec_C_D'.format(lifting_surface_name)
            sec_areas_name = '{}_sec_areas_m2'.format(lifting_surface_name)

            ones = np.ones(num_points_z - 1)
            rho_kg_m3 = np.outer(inputs['rho_kg_m3'], ones)
            v_m_s = np.outer(inputs['v_m_s'], ones)

            ind2 += num

            panel_lift = inputs['panel_forces_rotated'][:, ind1:ind2, 1].reshape(
                (num_nodes, num_points_x - 1, num_points_z - 1))
            panel_drag = inputs['panel_forces_rotated'][:, ind1:ind2, 0].reshape(
                (num_nodes, num_points_x - 1, num_points_z - 1))

            partials[sec_C_L_name, 'rho_kg_m3'] = (-np.sum(panel_lift, axis=1)
                / (0.5 * rho_kg_m3 ** 2 * v_m_s ** 2 * inputs[sec_areas_name])).flatten()
            partials[sec_C_L_name, 'v_m_s'] = (-2. * np.sum(panel_lift, axis=1)
                / (0.5 * rho_kg_m3 * v_m_s ** 3 * inputs[sec_areas_name])).flatten()
            partials[sec_C_L_name, 'panel_forces_rotated'] = np.einsum('ik,j->ijk',
                1. / (0.5 * rho_kg_m3 * v_m_s ** 2 * inputs[sec_areas_name]),
                np.ones(num_points_x - 1),
            ).flatten()
            partials[sec_C_L_name, sec_areas_name] = (-np.sum(panel_lift, axis=1)
                / (0.5 * rho_kg_m3 * v_m_s ** 2 * inputs[sec_areas_name] ** 2)).flatten()

            partials[sec_C_D_name, 'rho_kg_m3'] = (-np.sum(panel_drag, axis=1)
                / (0.5 * rho_kg_m3 ** 2 * v_m_s ** 2 * inputs[sec_areas_name])).flatten()
            partials[sec_C_D_name, 'v_m_s'] = (-2. * np.sum(panel_drag, axis=1)
                / (0.5 * rho_kg_m3 * v_m_s ** 3 * inputs[sec_areas_name])).flatten()
            partials[sec_C_D_name, 'panel_forces_rotated'] = np.einsum('ik,j->ijk',
                1. / (0.5 * rho_kg_m3 * v_m_s ** 2 * inputs[sec_areas_name]),
                np.ones(num_points_x - 1),
            ).flatten()
            partials[sec_C_D_name, sec_areas_name] = (-np.sum(panel_drag, axis=1)
                / (0.5 * rho_kg_m3 * v_m_s ** 2 * inputs[sec_areas_name] ** 2)).flatten()

            ind1 += num
