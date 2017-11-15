from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent

from openaerostruct_v2.utils.misc_utils import get_array_indices, tile_sparse_jac


rho = 20

class FEAKSComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)
        self.metadata.declare('lifting_surfaces', types=list)

    def setup(self):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            vonmises_name = '{}_vonmises'.format(lifting_surface_name)
            ks_name = '{}_ks'.format(lifting_surface_name)

            self.add_input(vonmises_name, shape=(num_nodes, num_points_z - 1, 2))
            self.add_output(ks_name)

            rows = np.zeros(num_nodes * (num_points_z - 1) * 2, int)
            cols = get_array_indices(num_nodes, num_points_z - 1, 2).flatten()
            self.declare_partials(ks_name, vonmises_name, rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            vonmises_name = '{}_vonmises'.format(lifting_surface_name)
            ks_name = '{}_ks'.format(lifting_surface_name)

            f = inputs[vonmises_name].flatten() - 1.
            indices = np.argmax(f)
            fmax = f[indices]

            outputs[ks_name] = fmax + 1. / rho * np.log(np.sum(np.exp(rho * (f - fmax))))

    def compute_partials(self, inputs, partials):
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            vonmises_name = '{}_vonmises'.format(lifting_surface_name)
            ks_name = '{}_ks'.format(lifting_surface_name)

            f = inputs[vonmises_name].flatten() - 1.
            indices = np.argmax(f)
            fmax = f[indices]

            dfmax_dvm = np.zeros(len(f))
            dfmax_dvm[indices] = 1.

            dks_dvm = dfmax_dvm + 1. / rho / np.sum(np.exp(rho * (f - fmax))) * np.exp(rho * (f - fmax)) * rho
            dks_dvm[indices] -= 1.

            partials[ks_name, vonmises_name] = dks_dvm
