from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent

from openaerostruct_v2.utils.vector_algebra import add_ones_axis
from openaerostruct_v2.utils.vector_algebra import compute_norm, compute_norm_deriv
from openaerostruct_v2.utils.vector_algebra import compute_cross, compute_cross_deriv1, compute_cross_deriv2


class FEALocalStiffTransformedComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('lifting_surfaces', type_=list)

    def setup(self):
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            transform_name = '{}_transform'.format(lifting_surface_name)
            in_name = '{}_local_stiff_permuted'.format(lifting_surface_name)
            out_name = '{}_local_stiff_transformed'.format(lifting_surface_name)

            self.add_input(transform_name, shape=(num_points_z - 1, 12, 12))
            self.add_input(in_name, shape=(num_points_z - 1, 12, 12))
            self.add_output(out_name, shape=(num_points_z - 1, 12, 12))

            indices = np.arange(144 * (num_points_z - 1)).reshape((num_points_z - 1, 12, 12))
            ones = np.ones((12, 12), int)

            rows = np.einsum('ijk,lm->ijklm', indices, ones).flatten()
            cols = np.einsum('ilm,jk->ijklm', indices, ones).flatten()

            self.declare_partials(out_name, transform_name, rows=rows, cols=cols)
            self.declare_partials(out_name, in_name, rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            transform_name = '{}_transform'.format(lifting_surface_name)
            in_name = '{}_local_stiff_permuted'.format(lifting_surface_name)
            out_name = '{}_local_stiff_transformed'.format(lifting_surface_name)

            outputs[out_name] = np.einsum('ilj,ilm,imk->ijk',
                inputs[transform_name], inputs[in_name], inputs[transform_name])

    def compute_partials(self, inputs, partials):
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            transform_name = '{}_transform'.format(lifting_surface_name)
            in_name = '{}_local_stiff_permuted'.format(lifting_surface_name)
            out_name = '{}_local_stiff_transformed'.format(lifting_surface_name)

            partials[out_name, in_name] = np.einsum('ilj,imk->ijklm',
                inputs[transform_name], inputs[transform_name]).flatten()

            partials[out_name, transform_name] = (
                np.einsum('ilj,ilp,kq->ijkpq',
                    inputs[transform_name], inputs[in_name], np.eye(12)) +
                np.einsum('jq,ipm,imk->ijkpq',
                    np.eye(12), inputs[in_name], inputs[transform_name])
            ).flatten()
