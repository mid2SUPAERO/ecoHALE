from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent

from openaerostruct.utils.misc_utils import get_array_indices


class VLMVortexAirfoilComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)
        self.metadata.declare('lifting_surfaces', types=list)

    def setup(self):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            airfoil_x_name = '{}_airfoil_x'.format(lifting_surface_name)
            airfoil_y_name = '{}_airfoil_y'.format(lifting_surface_name)

            vortex_airfoil_x_name = '{}_vortex_airfoil_x'.format(lifting_surface_name)
            vortex_airfoil_y_name = '{}_vortex_airfoil_y'.format(lifting_surface_name)

            self.add_input(airfoil_x_name, shape=(num_nodes, num_points_x, num_points_z))
            self.add_input(airfoil_y_name, shape=(num_nodes, num_points_x, num_points_z))
            self.add_output(vortex_airfoil_x_name, shape=(num_nodes, num_points_x, num_points_z))
            self.add_output(vortex_airfoil_y_name, shape=(num_nodes, num_points_x, num_points_z))

            indices = get_array_indices(num_nodes, num_points_x, num_points_z)

            data = np.concatenate([
                0.75 * np.ones(num_nodes * (num_points_x - 1) * num_points_z),
                0.25 * np.ones(num_nodes * (num_points_x - 1) * num_points_z),
                np.ones(num_nodes * num_points_z),
            ])
            rows = np.concatenate([
                indices[:, :-1, :].flatten(),
                indices[:, :-1, :].flatten(),
                indices[:, -1, :].flatten(),
            ])
            cols = np.concatenate([
                indices[:, :-1, :].flatten(),
                indices[:, 1:, :].flatten(),
                indices[:, -1, :].flatten(),
            ])
            self.declare_partials(vortex_airfoil_x_name, airfoil_x_name, val=data, rows=rows, cols=cols)
            self.declare_partials(vortex_airfoil_y_name, airfoil_y_name, val=data, rows=rows, cols=cols)

        self.set_check_partial_options('*', method='cs')

    def compute(self, inputs, outputs):
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            airfoil_x_name = '{}_airfoil_x'.format(lifting_surface_name)
            airfoil_y_name = '{}_airfoil_y'.format(lifting_surface_name)

            vortex_airfoil_x_name = '{}_vortex_airfoil_x'.format(lifting_surface_name)
            vortex_airfoil_y_name = '{}_vortex_airfoil_y'.format(lifting_surface_name)

            outputs[vortex_airfoil_x_name] = 0.
            outputs[vortex_airfoil_x_name][:, :-1, :] += 0.75 * inputs[airfoil_x_name][:, :-1, :]
            outputs[vortex_airfoil_x_name][:, :-1, :] += 0.25 * inputs[airfoil_x_name][:, 1:, :]
            outputs[vortex_airfoil_x_name][:, -1, :] += inputs[airfoil_x_name][:, -1, :]

            outputs[vortex_airfoil_y_name] = 0.
            outputs[vortex_airfoil_y_name][:, :-1, :] += 0.75 * inputs[airfoil_y_name][:, :-1, :]
            outputs[vortex_airfoil_y_name][:, :-1, :] += 0.25 * inputs[airfoil_y_name][:, 1:, :]
            outputs[vortex_airfoil_y_name][:, -1, :] += inputs[airfoil_y_name][:, -1, :]
