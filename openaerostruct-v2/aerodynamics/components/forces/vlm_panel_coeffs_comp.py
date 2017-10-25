from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent

from openaerostruct_v2.utils.vector_algebra import compute_cross, compute_cross_deriv1, compute_cross_deriv2


class VLMPanelCoeffsComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('lifting_surfaces', type_=list)

    def setup(self):
        lifting_surfaces = self.metadata['lifting_surfaces']

        system_size = 0

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            system_size += (num_points_x - 1) * (num_points_z - 1)

        self.system_size = system_size

        self.add_input('rho_kg_m3')
        self.add_input('v_m_s')
        self.add_input('panel_forces', shape=(system_size, 3))

        panel_forces_arange = np.arange(3 * system_size).reshape((system_size, 3))

        ind1, ind2 = 0, 0
        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1
            num = (num_points_x - 1) * (num_points_z - 1)

            sec_C_L_name = '{}_sec_C_L'.format(lifting_surface_name)
            sec_C_D_name = '{}_sec_C_D'.format(lifting_surface_name)
            sec_areas_name = '{}_sec_areas_m2'.format(lifting_surface_name)

            ind2 += num

            self.add_input(sec_areas_name, shape=(num_points_z - 1))
            self.add_output(sec_C_L_name, shape=(num_points_z - 1))
            self.add_output(sec_C_D_name, shape=(num_points_z - 1))

            self.declare_partials(sec_C_L_name, 'rho_kg_m3',
                rows=np.arange(num_points_z - 1),
                cols=np.zeros(num_points_z - 1, int),
            )
            self.declare_partials(sec_C_L_name, 'v_m_s',
                rows=np.arange(num_points_z - 1),
                cols=np.zeros(num_points_z - 1, int),
            )
            self.declare_partials(sec_C_L_name, 'panel_forces',
                rows=np.outer(
                    np.ones(num_points_x - 1, int),
                    np.arange(num_points_z - 1),
                ).flatten(),
                cols=panel_forces_arange[ind1:ind2, 1],
            )
            self.declare_partials(sec_C_L_name, sec_areas_name,
                rows=np.arange(num_points_z - 1),
                cols=np.arange(num_points_z - 1),
            )

            self.declare_partials(sec_C_D_name, 'rho_kg_m3',
                rows=np.arange(num_points_z - 1),
                cols=np.zeros(num_points_z - 1, int),
            )
            self.declare_partials(sec_C_D_name, 'v_m_s',
                rows=np.arange(num_points_z - 1),
                cols=np.zeros(num_points_z - 1, int),
            )
            self.declare_partials(sec_C_D_name, 'panel_forces',
                rows=np.outer(
                    np.ones(num_points_x - 1, int),
                    np.arange(num_points_z - 1),
                ).flatten(),
                cols=panel_forces_arange[ind1:ind2, 0],
            )
            self.declare_partials(sec_C_D_name, sec_areas_name,
                rows=np.arange(num_points_z - 1),
                cols=np.arange(num_points_z - 1),
            )

            ind1 += num

    def compute(self, inputs, outputs):
        rho_kg_m3 = inputs['rho_kg_m3']
        v_m_s = inputs['v_m_s']

        lifting_surfaces = self.metadata['lifting_surfaces']

        ind1, ind2 = 0, 0
        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1
            num = (num_points_x - 1) * (num_points_z - 1)

            sec_C_L_name = '{}_sec_C_L'.format(lifting_surface_name)
            sec_C_D_name = '{}_sec_C_D'.format(lifting_surface_name)
            sec_areas_name = '{}_sec_areas_m2'.format(lifting_surface_name)

            ind2 += num

            panel_lift = inputs['panel_forces'][ind1:ind2, 1].reshape((num_points_x - 1, num_points_z - 1))
            panel_drag = inputs['panel_forces'][ind1:ind2, 0].reshape((num_points_x - 1, num_points_z - 1))

            outputs[sec_C_L_name] = np.sum(panel_lift, axis=0) \
                / (0.5 * rho_kg_m3 * v_m_s ** 2 * inputs[sec_areas_name])
            outputs[sec_C_D_name] = np.sum(panel_drag, axis=0) \
                / (0.5 * rho_kg_m3 * v_m_s ** 2 * inputs[sec_areas_name])

            ind1 += num

    def compute_partials(self, inputs, partials):
        rho_kg_m3 = inputs['rho_kg_m3']
        v_m_s = inputs['v_m_s']
        panel_lift = inputs['panel_forces'][:, 1]
        panel_drag = inputs['panel_forces'][:, 0]

        lifting_surfaces = self.metadata['lifting_surfaces']

        ind1, ind2 = 0, 0
        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1
            num = (num_points_x - 1) * (num_points_z - 1)

            sec_C_L_name = '{}_sec_C_L'.format(lifting_surface_name)
            sec_C_D_name = '{}_sec_C_D'.format(lifting_surface_name)
            sec_areas_name = '{}_sec_areas_m2'.format(lifting_surface_name)

            ind2 += num

            panel_lift = inputs['panel_forces'][ind1:ind2, 1].reshape((num_points_x - 1, num_points_z - 1))
            panel_drag = inputs['panel_forces'][ind1:ind2, 0].reshape((num_points_x - 1, num_points_z - 1))

            partials[sec_C_L_name, 'rho_kg_m3'] = -np.sum(panel_lift, axis=0) / (0.5 * rho_kg_m3 ** 2 * v_m_s ** 2 * inputs[sec_areas_name])
            partials[sec_C_L_name, 'v_m_s'] = -2. * np.sum(panel_lift, axis=0) / (0.5 * rho_kg_m3 * v_m_s ** 3 * inputs[sec_areas_name])
            partials[sec_C_L_name, 'panel_forces'] = 1. / (0.5 * rho_kg_m3 * v_m_s ** 2
                * np.outer(
                    np.ones(num_points_x - 1),
                    inputs[sec_areas_name],
                ).flatten()
            )
            partials[sec_C_L_name, sec_areas_name] = -np.sum(panel_lift, axis=0) / (0.5 * rho_kg_m3 * v_m_s ** 2 * inputs[sec_areas_name] ** 2)

            partials[sec_C_D_name, 'rho_kg_m3'] = -np.sum(panel_drag, axis=0) / (0.5 * rho_kg_m3 ** 2 * v_m_s ** 2 * inputs[sec_areas_name])
            partials[sec_C_D_name, 'v_m_s'] = -2. * np.sum(panel_drag, axis=0) / (0.5 * rho_kg_m3 * v_m_s ** 3 * inputs[sec_areas_name])
            partials[sec_C_D_name, 'panel_forces'] = 1. / (0.5 * rho_kg_m3 * v_m_s ** 2
                * np.outer(
                    np.ones(num_points_x - 1),
                    inputs[sec_areas_name],
                ).flatten()
            )
            partials[sec_C_D_name, sec_areas_name] = -np.sum(panel_drag, axis=0) / (0.5 * rho_kg_m3 * v_m_s ** 2 * inputs[sec_areas_name] ** 2)

            ind1 += num
