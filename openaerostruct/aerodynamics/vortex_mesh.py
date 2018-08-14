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

            self.add_input(mesh_name, shape=(nx, ny, 3), units='m')

            if surface['symmetry']:
                self.add_output(vortex_mesh_name, shape=(nx, ny*2-1, 3), units='m')

                mesh_indices = np.arange(nx * ny * 3).reshape((nx, ny, 3))
                vor_indices = np.arange(nx * (2*ny-1) * 3).reshape((nx, (2*ny-1), 3))

                rows = np.tile(vor_indices[:(nx-1), :ny, :].flatten(), 2)
                rows = np.hstack((rows, vor_indices[-1  , :ny, :].flatten()))
                rows = np.hstack((rows, np.tile(vor_indices[:(nx-1), ny:, :][:, ::-1, :].flatten(), 2)))
                rows = np.hstack((rows, vor_indices[-1, ny:, :].flatten()))
                rows = np.hstack((rows, np.tile(vor_indices[:(nx-1), ny:, 1][:, ::-1].flatten(), 2)))
                rows = np.hstack((rows, vor_indices[-1, ny:, 1].flatten()))

                cols = np.concatenate([
                    mesh_indices[:-1, :, :].flatten(),
                    mesh_indices[1:  , :, :].flatten(),
                    mesh_indices[-1  , :, :].flatten(),
                    mesh_indices[:-1, :-1, :].flatten(),
                    mesh_indices[1:  , :-1, :].flatten(),
                    mesh_indices[-1  , :-1, :][::-1, :].flatten(),
                    mesh_indices[:-1, :-1, 1].flatten(),
                    mesh_indices[1:  , :-1, 1].flatten(),
                    mesh_indices[-1  , :-1, 1][::-1].flatten(),
                ])

                data = np.concatenate([
                    0.75 * np.ones((nx-1) * ny * 3),
                    0.25 * np.ones((nx-1) * ny * 3),
                    np.ones(ny * 3),  # back row
                    0.75 * np.ones((nx-1) * (ny-1) * 3),
                    0.25 * np.ones((nx-1) * (ny-1) * 3),
                    np.ones((ny-1) * 3),  # back row
                    -1.5 * np.ones((nx-1) * (ny-1)),
                    -.5  * np.ones((nx-1) * (ny-1)),
                    -2 * np.ones((ny-1)),  # back row
                ])

                self.declare_partials(vortex_mesh_name, mesh_name, val=data, rows=rows, cols=cols)

            else:
                self.add_output(vortex_mesh_name, shape=(nx, ny, 3), units='m')

                mesh_indices = np.arange(nx * ny * 3).reshape(
                    (nx, ny, 3))

                rows = np.tile(mesh_indices[:(nx-1), :, :].flatten(), 2)
                rows = np.hstack((rows, mesh_indices[-1  , :, :].flatten()))
                cols = np.concatenate([
                    mesh_indices[:-1, :, :].flatten(),
                    mesh_indices[1:  , :, :].flatten(),
                    mesh_indices[-1  , :, :].flatten(),
                ])

                data = np.concatenate([
                    0.75 * np.ones((nx-1) * ny * 3),
                    0.25 * np.ones((nx-1) * ny * 3),
                    np.ones(ny * 3),  # back row
                ])

                self.declare_partials(vortex_mesh_name, mesh_name, val=data, rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        surfaces = self.options['surfaces']

        for surface in surfaces:
            nx = surface['num_x']
            ny = surface['num_y']
            name = surface['name']

            mesh_name = '{}_def_mesh'.format(name)
            vortex_mesh_name = '{}_vortex_mesh'.format(name)

            if surface['symmetry']:
                mesh = np.zeros((nx, ny*2-1, 3))
                mesh[:, :ny, :] = inputs[mesh_name]
                mesh[:, ny:, :] = inputs[mesh_name][:, :-1, :][:, ::-1, :]
                mesh[:, ny:, 1] *= -1.
            else:
                mesh = inputs[mesh_name]

            outputs[vortex_mesh_name][:-1, :, :] = 0.75 * mesh[:-1, :, :] + 0.25 * mesh[1:, :, :]
            outputs[vortex_mesh_name][-1, :, :] = mesh[-1, :, :]
