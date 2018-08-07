from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent


class VortexMesh(ExplicitComponent):

    def initialize(self):
        self.options.declare('surfaces', types=list)

    def setup(self):
        surfaces = self.options['surfaces']

        for surface in surfaces:
            nx = surface['num_x']
            ny = surface['num_y']
            name = surface['name']

            mesh_name = '{}_def_mesh'.format(name)
            vortex_mesh_name = '{}_vortex_mesh'.format(name)

            self.add_input(mesh_name, shape=(nx, ny, 3))
            self.add_output(vortex_mesh_name, shape=(nx, ny, 3))

            # TODO: actually get the derivatives of this
            self.declare_partials(vortex_mesh_name, mesh_name, method='fd')

    def compute(self, inputs, outputs):
        surfaces = self.options['surfaces']

        for surface in surfaces:
            nx = surface['num_x']
            ny = surface['num_y']
            name = surface['name']

            mesh_name = '{}_def_mesh'.format(name)
            vortex_mesh_name = '{}_vortex_mesh'.format(name)

            mesh = inputs[mesh_name]

            outputs[vortex_mesh_name][:-1, :, :] = 0.75 * mesh[:-1, :, :] + 0.25 * mesh[1:, :, :]
            outputs[vortex_mesh_name][-1, :, :] = mesh[-1, :, :]
