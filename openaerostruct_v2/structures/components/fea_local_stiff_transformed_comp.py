from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent

from openaerostruct_v2.utils.vector_algebra import add_ones_axis
from openaerostruct_v2.utils.vector_algebra import compute_norm, compute_norm_deriv
from openaerostruct_v2.utils.vector_algebra import compute_cross, compute_cross_deriv1, compute_cross_deriv2
from openaerostruct_v2.utils.misc_utils import get_array_indices, tile_sparse_jac


class FEALocalStiffTransformedComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('num_nodes', type_=int)
        self.metadata.declare('lifting_surfaces', type_=list)

    def setup(self):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            transform_name = '{}_transform'.format(lifting_surface_name)
            in_name = '{}_local_stiff_permuted'.format(lifting_surface_name)
            out_name = '{}_local_stiff_transformed'.format(lifting_surface_name)

            self.add_input(transform_name, shape=(num_nodes, num_points_z - 1, 12, 12))
            self.add_input(in_name, shape=(num_nodes, num_points_z - 1, 12, 12))
            self.add_output(out_name, shape=(num_nodes, num_points_z - 1, 12, 12))

            indices = get_array_indices(num_points_z - 1, 12, 12)
            ones = np.ones((12, 12), int)

            rows = np.einsum('ijk,lm->ijklm', indices, ones).flatten()
            cols = np.einsum('ilm,jk->ijklm', indices, ones).flatten()
            _, rows, cols = tile_sparse_jac(1., rows, cols,
                (num_points_z - 1) * 12 * 12, (num_points_z - 1) * 12 * 12, num_nodes)

            self.declare_partials(out_name, transform_name, rows=rows, cols=cols)
            self.declare_partials(out_name, in_name, rows=rows, cols=cols)

            self.set_check_partial_options(transform_name, step=1e-5)
            self.set_check_partial_options(in_name, step=1e0)

    def compute(self, inputs, outputs):
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            transform_name = '{}_transform'.format(lifting_surface_name)
            in_name = '{}_local_stiff_permuted'.format(lifting_surface_name)
            out_name = '{}_local_stiff_transformed'.format(lifting_surface_name)

            outputs[out_name] = np.einsum('ijmk,ijmn,ijnl->ijkl',
                inputs[transform_name], inputs[in_name], inputs[transform_name])

    def compute_partials(self, inputs, partials):
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            transform_name = '{}_transform'.format(lifting_surface_name)
            in_name = '{}_local_stiff_permuted'.format(lifting_surface_name)
            out_name = '{}_local_stiff_transformed'.format(lifting_surface_name)

            partials[out_name, in_name] = np.einsum('ijmk,ijnl->ijklmn',
                inputs[transform_name], inputs[transform_name]).flatten()

            partials[out_name, transform_name] = (
                np.einsum('ijmk,ijmq,lr->ijklqr',
                    inputs[transform_name], inputs[in_name], np.eye(12)) +
                np.einsum('kr,ijqn,ijnl->ijklqr',
                    np.eye(12), inputs[in_name], inputs[transform_name])
            ).flatten()
