from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent

from openaerostruct.utils.vector_algebra import get_array_indices, compute_cross, compute_cross_deriv1, compute_cross_deriv2


class ComputeTransformationMatrix(ExplicitComponent):

    def initialize(self):
        self.options.declare('surface', types=dict)

    def setup(self):
        self.surface = surface = self.options['surface']

        ny = self.ny = surface['num_y']
        nx = self.nx = surface['num_x']

        self.add_input('disp', val=np.zeros((ny, 6)), units='m')
        self.add_output('transformation_matrix', shape=(ny, 3, 3))

        disp_indices = get_array_indices(ny, 6)
        transform_indices = get_array_indices(ny, 3, 3)

        rows = np.einsum('ijk,l->ijkl',
            transform_indices,
            np.ones(3, int)).flatten()
        cols = np.einsum('il,jk->ijkl',
            get_array_indices(ny, 6)[:, 3:],
            np.ones((3, 3), int)).flatten()
        self.declare_partials('transformation_matrix', 'disp', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        outputs['transformation_matrix'] = 0.
        for i in range(3):
            outputs['transformation_matrix'][:, i, i] -= 2.

        rx = inputs['disp'][:, 3]
        ry = inputs['disp'][:, 4]
        rz = inputs['disp'][:, 5]

        # T[ 1:,  1:] += [[cos(rx), -sin(rx)], [ sin(rx), cos(rx)]]
        outputs['transformation_matrix'][:, 1, 1] += np.cos(rx)
        outputs['transformation_matrix'][:, 1, 2] -= np.sin(rx)
        outputs['transformation_matrix'][:, 2, 1] += np.sin(rx)
        outputs['transformation_matrix'][:, 2, 2] += np.cos(rx)

        # T[::2, ::2] += [[cos(ry),  sin(ry)], [-sin(ry), cos(ry)]]
        outputs['transformation_matrix'][:, 0, 0] += np.cos(ry)
        outputs['transformation_matrix'][:, 0, 2] += np.sin(ry)
        outputs['transformation_matrix'][:, 2, 0] -= np.sin(ry)
        outputs['transformation_matrix'][:, 2, 2] += np.cos(ry)

        # T[ :2,  :2] += [[cos(rz), -sin(rz)], [ sin(rz), cos(rz)]]
        outputs['transformation_matrix'][:, 0, 0] += np.cos(rz)
        outputs['transformation_matrix'][:, 0, 1] -= np.sin(rz)
        outputs['transformation_matrix'][:, 1, 0] += np.sin(rz)
        outputs['transformation_matrix'][:, 1, 1] += np.cos(rz)

    def compute_partials(self, inputs, partials):
        rx = inputs['disp'][:, 3]
        ry = inputs['disp'][:, 4]
        rz = inputs['disp'][:, 5]

        derivs = partials['transformation_matrix', 'disp'].reshape((self.ny, 3, 3, 3))
        derivs[:, :, :, :] = 0.

        derivs[:, 1, 1, 0] -= np.sin(rx)
        derivs[:, 1, 2, 0] -= np.cos(rx)
        derivs[:, 2, 1, 0] += np.cos(rx)
        derivs[:, 2, 2, 0] -= np.sin(rx)

        derivs[:, 0, 0, 1] -= np.sin(ry)
        derivs[:, 0, 2, 1] += np.cos(ry)
        derivs[:, 2, 0, 1] -= np.cos(ry)
        derivs[:, 2, 2, 1] -= np.sin(ry)

        derivs[:, 0, 0, 2] -= np.sin(rz)
        derivs[:, 0, 1, 2] -= np.cos(rz)
        derivs[:, 1, 0, 2] += np.cos(rz)
        derivs[:, 1, 1, 2] -= np.sin(rz)
