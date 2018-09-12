from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent

from openaerostruct.utils.vector_algebra import add_ones_axis
from openaerostruct.utils.vector_algebra import compute_norm, compute_norm_deriv
from openaerostruct.utils.vector_algebra import compute_cross, compute_cross_deriv1, compute_cross_deriv2


class LocalStiffTransformed(ExplicitComponent):

    def initialize(self):
        self.options.declare('surface', types=dict)

    def setup(self):
        surface = self.options['surface']

        ny = surface['num_y']

        self.add_input('transform', shape=(ny - 1, 12, 12))
        self.add_input('local_stiff_permuted', shape=(ny - 1, 12, 12))
        self.add_output('local_stiff_transformed', shape=(ny - 1, 12, 12))

        indices = np.arange(144 * (ny - 1)).reshape((ny - 1, 12, 12))
        ones = np.ones((12, 12), int)

        rows = np.einsum('ijk,lm->ijklm', indices, ones).flatten()
        cols = np.einsum('ilm,jk->ijklm', indices, ones).flatten()

        self.declare_partials('local_stiff_transformed', 'transform', rows=rows, cols=cols)
        self.declare_partials('local_stiff_transformed', 'local_stiff_permuted', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        surface = self.options['surface']

        ny = surface['num_y']

        outputs['local_stiff_transformed'] = np.einsum('ilj,ilm,imk->ijk',
            inputs['transform'], inputs['local_stiff_permuted'], inputs['transform'])

    def compute_partials(self, inputs, partials):
        surface = self.options['surface']

        ny = surface['num_y']

        partials['local_stiff_transformed', 'local_stiff_permuted'] = np.einsum('ilj,imk->ijklm',
            inputs['transform'], inputs['transform']).flatten()

        partials['local_stiff_transformed', 'transform'] = (
            np.einsum('ilj,ilp,kq->ijkpq',
                inputs['transform'], inputs['local_stiff_permuted'], np.eye(12)) +
            np.einsum('jq,ipm,imk->ijkpq',
                np.eye(12), inputs['local_stiff_permuted'], inputs['transform'])
        ).flatten()
