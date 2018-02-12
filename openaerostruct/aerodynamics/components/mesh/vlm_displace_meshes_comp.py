from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent

from openaerostruct.utils.misc_utils import tile_sparse_jac


class VLMDisplaceMeshesComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)
        self.metadata.declare('lifting_surfaces', types=list)

    def setup(self):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']

        self.airfoils = airfoils = {}

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data.num_points_x
            num_points_z = 2 * lifting_surface_data.num_points_z_half - 1

            undeformed_vortex_mesh_name = '{}_undeformed_vortex_mesh'.format(lifting_surface_name)
            undeformed_mesh_name = '{}_undeformed_mesh'.format(lifting_surface_name)
            vortex_mesh_displacement = '{}_vortex_mesh_displacement'.format(lifting_surface_name)
            mesh_displacement = '{}_mesh_displacement'.format(lifting_surface_name)

            self.add_input(undeformed_vortex_mesh_name, shape=(num_nodes, num_points_x, num_points_z, 3))
            self.add_input(undeformed_mesh_name, shape=(num_nodes, num_points_x, num_points_z, 3))
            self.add_input(vortex_mesh_displacement, shape=(num_nodes, num_points_x, num_points_z, 3), val=0.)
            self.add_input(mesh_displacement, shape=(num_nodes, num_points_x, num_points_z, 3), val=0.)

            vortex_mesh_name = '{}_vortex_mesh'.format(lifting_surface_name)
            mesh_name = '{}_mesh'.format(lifting_surface_name)

            self.add_output(vortex_mesh_name, shape=(num_nodes, num_points_x, num_points_z, 3),
                val=np.random.rand(num_nodes, num_points_x, num_points_z, 3))
            self.add_output(mesh_name, shape=(num_nodes, num_points_x, num_points_z, 3),
                val=np.random.rand(num_nodes, num_points_x, num_points_z, 3))

            arange = np.arange(num_points_x * num_points_z * 3)

            _, rows, cols = tile_sparse_jac(1., arange, arange,
                num_points_x * num_points_z * 3, num_points_x * num_points_z * 3, num_nodes)

            self.declare_partials(vortex_mesh_name, undeformed_vortex_mesh_name,
                rows=rows, cols=cols, val=1.)
            self.declare_partials(vortex_mesh_name, vortex_mesh_displacement,
                rows=rows, cols=cols, val=1.)

            self.declare_partials(mesh_name, undeformed_mesh_name,
                rows=rows, cols=cols, val=1.)
            self.declare_partials(mesh_name, mesh_displacement,
                rows=rows, cols=cols, val=1.)

        self.set_check_partial_options('*', method='cs')

    def compute(self, inputs, outputs):
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data.num_points_x
            num_points_z = 2 * lifting_surface_data.num_points_z_half - 1

            undeformed_vortex_mesh_name = '{}_undeformed_vortex_mesh'.format(lifting_surface_name)
            undeformed_mesh_name = '{}_undeformed_mesh'.format(lifting_surface_name)
            vortex_mesh_displacement = '{}_vortex_mesh_displacement'.format(lifting_surface_name)
            mesh_displacement = '{}_mesh_displacement'.format(lifting_surface_name)
            vortex_mesh_name = '{}_vortex_mesh'.format(lifting_surface_name)
            mesh_name = '{}_mesh'.format(lifting_surface_name)

            outputs[mesh_name] = inputs[undeformed_mesh_name] + inputs[mesh_displacement]
            outputs[vortex_mesh_name] = inputs[undeformed_vortex_mesh_name] + inputs[vortex_mesh_displacement]
