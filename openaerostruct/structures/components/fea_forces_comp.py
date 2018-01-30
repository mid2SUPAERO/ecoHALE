from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent

from openaerostruct.utils.misc_utils import get_array_indices, tile_sparse_jac


class FEAForcesComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)
        self.metadata.declare('lifting_surfaces', types=list)
        self.metadata.declare('fea_scaler', types=float)

    def setup(self):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']
        fea_scaler = self.metadata['fea_scaler']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            size = 6 * num_points_z + 6

            loads_name = '{}_loads'.format(lifting_surface_name)
            ext_loads_name = '{}_ext_loads'.format(lifting_surface_name)
            forces_name = '{}_forces'.format(lifting_surface_name)

            self.add_input(loads_name, shape=(num_nodes, num_points_z, 6))
            self.add_input(ext_loads_name, shape=(num_nodes, num_points_z, 6), val=0.)
            self.add_output(forces_name, shape=(num_nodes, size), val=0.)

            arange = np.arange(6 * num_points_z)
            _, rows, cols = tile_sparse_jac(1., arange, arange,
                size, num_points_z * 6, num_nodes)
            self.declare_partials(forces_name, loads_name, val=1. / fea_scaler, rows=rows, cols=cols)
            self.declare_partials(forces_name, ext_loads_name, val=1. / fea_scaler, rows=rows, cols=cols)

        self.set_check_partial_options('*', method='cs')

    def compute(self, inputs, outputs):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']
        fea_scaler = self.metadata['fea_scaler']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            loads_name = '{}_loads'.format(lifting_surface_name)
            ext_loads_name = '{}_ext_loads'.format(lifting_surface_name)
            forces_name = '{}_forces'.format(lifting_surface_name)

            outputs[forces_name][:, :6 * num_points_z] = (
                inputs[loads_name].reshape((num_nodes, 6 * num_points_z)) +
                inputs[ext_loads_name].reshape((num_nodes, 6 * num_points_z))
            ) / fea_scaler
