from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent


class VLMDisplaceMeshesComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('lifting_surfaces', type_=list)

    def setup(self):
        lifting_surfaces = self.metadata['lifting_surfaces']

        self.airfoils = airfoils = {}

        self.declare_partials('*', '*', dependent=False)

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            undeformed_vortex_mesh_name = '{}_undeformed_vortex_mesh'.format(lifting_surface_name)
            undeformed_mesh_name = '{}_undeformed_mesh'.format(lifting_surface_name)
            vortex_mesh_displacement = '{}_vortex_mesh_displacement'.format(lifting_surface_name)
            mesh_displacement = '{}_mesh_displacement'.format(lifting_surface_name)

            self.add_input(undeformed_vortex_mesh_name, shape=(num_points_x, num_points_z, 3))
            self.add_input(undeformed_mesh_name, shape=(num_points_x, num_points_z, 3))
            self.add_input(vortex_mesh_displacement, shape=(num_points_x, num_points_z, 3), val=0.)
            self.add_input(mesh_displacement, shape=(num_points_x, num_points_z, 3), val=0.)

            vortex_mesh_name = '{}_vortex_mesh'.format(lifting_surface_name)
            mesh_name = '{}_mesh'.format(lifting_surface_name)

            self.add_output(vortex_mesh_name, shape=(num_points_x, num_points_z, 3),
                val=np.random.rand(num_points_x, num_points_z, 3))
            self.add_output(mesh_name, shape=(num_points_x, num_points_z, 3),
                val=np.random.rand(num_points_x, num_points_z, 3))

            arange = np.arange(num_points_x * num_points_z * 3)

            self.declare_partials(vortex_mesh_name, undeformed_vortex_mesh_name,
                rows=arange, cols=arange, val=1.)
            self.declare_partials(vortex_mesh_name, vortex_mesh_displacement,
                rows=arange, cols=arange, val=1.)

            self.declare_partials(mesh_name, undeformed_mesh_name,
                rows=arange, cols=arange, val=1.)
            self.declare_partials(mesh_name, mesh_displacement,
                rows=arange, cols=arange, val=1.)

    def compute(self, inputs, outputs):
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            undeformed_vortex_mesh_name = '{}_undeformed_vortex_mesh'.format(lifting_surface_name)
            undeformed_mesh_name = '{}_undeformed_mesh'.format(lifting_surface_name)
            vortex_mesh_displacement = '{}_vortex_mesh_displacement'.format(lifting_surface_name)
            mesh_displacement = '{}_mesh_displacement'.format(lifting_surface_name)
            vortex_mesh_name = '{}_vortex_mesh'.format(lifting_surface_name)
            mesh_name = '{}_mesh'.format(lifting_surface_name)

            outputs[mesh_name] = inputs[undeformed_mesh_name] + inputs[mesh_displacement]
            outputs[vortex_mesh_name] = inputs[undeformed_vortex_mesh_name] + inputs[vortex_mesh_displacement]
