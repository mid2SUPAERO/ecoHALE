from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent

from openaerostruct.utils.vector_algebra import compute_norm


class Length(ExplicitComponent):

    def initialize(self):
        self.options.declare('surface', types=dict)

    def setup(self):
        surface = self.options['surface']

        ny = surface['num_y']

        self.add_input('nodes', shape=(ny, 3))
        self.add_output('element_lengths', shape=ny - 1)

        mesh_indices = np.arange(3 * ny).reshape((ny, 3))
        length_indices = np.arange(ny - 1)

        rows = np.tile(np.outer(length_indices, np.ones(3, int)).flatten(), 2)
        cols = np.concatenate([
            mesh_indices[:-1, :].flatten(),
            mesh_indices[1: , :].flatten(),
        ])
        self.declare_partials('element_lengths', 'nodes', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        surface = self.options['surface']

        ny = surface['num_y']

        vec = inputs['nodes'][1:, :] - inputs['nodes'][:-1, :]

        outputs['element_lengths'] = np.linalg.norm(vec, axis=-1)

    def compute_partials(self, inputs, partials):
        surface = self.options['surface']

        ny = surface['num_y']

        vec = inputs['nodes'][1:, :] - inputs['nodes'][:-1, :]
        vec_deriv = np.einsum('i,jk->ijk', np.ones(ny - 1), np.eye(3))

        derivs = partials['element_lengths', 'nodes'].reshape((2, ny - 1, 3))
        derivs[0, :, :] = -vec / compute_norm(vec)
        derivs[1, :, :] =  vec / compute_norm(vec)
