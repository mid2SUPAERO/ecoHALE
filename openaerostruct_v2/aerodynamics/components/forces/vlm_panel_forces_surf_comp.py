from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent

from openaerostruct_v2.utils.vector_algebra import compute_cross, compute_cross_deriv1, compute_cross_deriv2


class VLMPanelForcesSurfComp(ExplicitComponent):
    """
    Total forces by panel (flattened), aligned with the freestream (the lift and drag axes).
    """

    def initialize(self):
        self.metadata.declare('lifting_surfaces', type_=list)

    def setup(self):
        lifting_surfaces = self.metadata['lifting_surfaces']

        system_size = 0

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            system_size += (num_points_x - 1) * (num_points_z - 1)

        arange = np.arange(3 * system_size)

        self.add_input('panel_forces_rotated', shape=(system_size, 3))

        ind1, ind2 = 0, 0
        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            out_name = '{}_panel_forces_rotated'.format(lifting_surface_name)

            ind2 += (num_points_x - 1) * (num_points_z - 1) * 3

            self.add_output(out_name, shape=(num_points_x - 1, num_points_z - 1, 3))

            rows = np.arange((num_points_x - 1) * (num_points_z - 1) * 3)
            cols = arange[ind1:ind2]
            self.declare_partials(out_name, 'panel_forces_rotated', val=1., rows=rows, cols=cols)

            ind1 += (num_points_x - 1) * (num_points_z - 1) * 3

    def compute(self, inputs, outputs):
        lifting_surfaces = self.metadata['lifting_surfaces']

        ind1, ind2 = 0, 0
        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            out_name = '{}_panel_forces_rotated'.format(lifting_surface_name)

            ind2 += (num_points_x - 1) * (num_points_z - 1) * 3

            outputs[out_name] = inputs['panel_forces_rotated'][ind1:ind2].reshape(
                (num_points_x - 1, num_points_z - 1, 3))

            ind1 += (num_points_x - 1) * (num_points_z - 1) * 3
