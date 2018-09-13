from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent

from openaerostruct.utils.vector_algebra import add_ones_axis
from openaerostruct.utils.vector_algebra import compute_norm, compute_norm_deriv
from openaerostruct.utils.vector_algebra import compute_cross, compute_cross_deriv1, compute_cross_deriv2


class GlobalStiff(ExplicitComponent):

    def initialize(self):
        self.options.declare('surface', types=dict)

    def setup(self):
        surface = self.options['surface']

        ny = surface['num_y']

        size = 6 * ny + 6

        self.add_input('nodes', shape=(ny, 3))
        self.add_input('local_stiff_transformed', shape=(ny - 1, 12, 12))
        self.add_output('K', shape=(size, size))

        arange = np.arange(ny - 1)

        rows = np.empty((ny - 1, 12, 12), int)
        for i in range(12):
            for j in range(12):
                mtx_i = 6 * arange + i
                mtx_j = 6 * arange + j
                rows[:, i, j] = size * mtx_i + mtx_j
        rows = rows.flatten()
        cols = np.arange(144 * (ny - 1))
        self.declare_partials('K', 'local_stiff_transformed', val=1., rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        surface = self.options['surface']

        ny = surface['num_y']

        size = 6 * ny + 6

        arange = np.arange(ny - 1)

        outputs['K'] = 0.
        for i in range(12):
            for j in range(12):
                outputs['K'][6 * arange + i, 6 * arange + j] += inputs['local_stiff_transformed'][:, i, j]

        # Find constrained nodes based on closeness to central point
        nodes = inputs['nodes']
        dist = nodes - np.array([5., 0, 0])
        idx = (np.linalg.norm(dist, axis=1)).argmin()
        index = 6 * idx
        num_dofs = 6 * ny

        arange = np.arange(6)

        outputs['K'][index + arange, num_dofs + arange] = 1.e9
        outputs['K'][num_dofs + arange, index + arange] = 1.e9
