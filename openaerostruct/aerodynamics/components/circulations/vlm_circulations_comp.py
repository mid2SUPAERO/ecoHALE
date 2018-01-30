from __future__ import print_function
import numpy as np
from scipy.linalg import lu_factor, lu_solve

from openmdao.api import ImplicitComponent, AnalysisError

from openaerostruct.utils.misc_utils import tile_sparse_jac


class VLMCirculationsComp(ImplicitComponent):

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)
        self.metadata.declare('lifting_surfaces', types=list)

    def setup(self):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']

        system_size = 0

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            system_size += (num_points_x - 1) * (num_points_z - 1)

        self.system_size = system_size

        self.add_input('mtx', shape=(num_nodes, system_size, system_size))
        self.add_input('rhs', shape=(num_nodes, system_size))
        self.add_output('circulations', shape=(num_nodes, system_size))

        rows = np.outer(np.arange(system_size), np.ones(system_size, int)).flatten()
        cols = np.outer(np.ones(system_size, int), np.arange(system_size)).flatten()
        _, rows, cols = tile_sparse_jac(1., rows, cols,
            system_size, system_size, num_nodes)
        self.declare_partials('circulations', 'circulations', rows=rows, cols=cols)

        rows = np.outer(np.arange(system_size), np.ones(system_size, int)).flatten()
        cols = np.arange(system_size ** 2)
        _, rows, cols = tile_sparse_jac(1., rows, cols,
            system_size, system_size ** 2, num_nodes)
        self.declare_partials('circulations', 'mtx', rows=rows, cols=cols)

        rows = np.arange(system_size)
        cols = np.arange(system_size)
        _, rows, cols = tile_sparse_jac(1., rows, cols,
            system_size, system_size, num_nodes)
        self.declare_partials('circulations', 'rhs', val=-1., rows=rows, cols=cols)

        self.lu = {}

    def apply_nonlinear(self, inputs, outputs, residuals):
        residuals['circulations'] = np.einsum('ijk,ik->ij', inputs['mtx'], outputs['circulations']) \
            - inputs['rhs']

    def solve_nonlinear(self, inputs, outputs):
        num_nodes = self.metadata['num_nodes']

        for i in range(num_nodes):
            try:
                self.lu[i] = lu_factor(inputs['mtx'][i, :, :])
            except ValueError:
                raise AnalysisError

            outputs['circulations'][i, :] = lu_solve(self.lu[i], inputs['rhs'][i, :])

    def linearize(self, inputs, outputs, partials):
        num_nodes = self.metadata['num_nodes']

        system_size = self.system_size

        for i in range(num_nodes):
            try:
                self.lu[i] = lu_factor(inputs['mtx'][i, :, :])
            except ValueError:
                raise AnalysisError

        partials['circulations', 'circulations'] = inputs['mtx'].flatten()
        partials['circulations', 'mtx'] = \
            np.einsum('j,ik->ijk', np.ones(system_size), outputs['circulations']).flatten()

    def solve_linear(self, d_outputs, d_residuals, mode):
        num_nodes = self.metadata['num_nodes']

        if mode == 'fwd':
            for i in range(num_nodes):
                try:
                    d_outputs['circulations'][i, :] = lu_solve(self.lu[i], d_residuals['circulations'][i, :], trans=0)
                except ValueError:
                    raise AnalysisError
        else:
            for i in range(num_nodes):
                d_residuals['circulations'][i, :] = lu_solve(self.lu[i], d_outputs['circulations'][i, :], trans=1)

    def solve_multi_linear(self, d_outputs, d_residuals, mode):
        num_nodes = self.metadata['num_nodes']
        ncol = d_outputs['circulations'].shape[-1]

        if mode == 'fwd':
            for i in range(num_nodes):
                for j in range(ncol):
                    d_outputs['circulations'][i, :, j] = lu_solve(self.lu[i], d_residuals['circulations'][i, :, j], trans=0)
        else:
            for i in range(num_nodes):
                for j in range(ncol):
                    d_residuals['circulations'][i, :, j] = lu_solve(self.lu[i], d_outputs['circulations'][i, :, j], trans=1)
