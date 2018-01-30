from __future__ import print_function
import numpy as np
from scipy.linalg import lu_factor, lu_solve

from openmdao.api import ImplicitComponent

from openaerostruct.utils.misc_utils import get_array_indices, tile_sparse_jac
from openaerostruct.utils.linear_solvers import OASLinearSolver


class FEAStatesComp(ImplicitComponent):

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)
        self.metadata.declare('lifting_surfaces', types=list)

    def setup(self):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            size = 6 * num_points_z + 6

            forces_name = '{}_forces'.format(lifting_surface_name)
            mtx_name = '{}_global_stiff'.format(lifting_surface_name)
            states_name = '{}_states'.format(lifting_surface_name)

            self.add_input(mtx_name, shape=(num_nodes, size, size))
            self.add_input(forces_name, shape=(num_nodes, size))
            self.add_output(states_name, shape=(num_nodes, size))

            rows = np.outer(np.arange(size), np.ones(size, int)).flatten()
            cols = np.arange(size ** 2)
            _, rows, cols = tile_sparse_jac(1., rows, cols,
                size, size ** 2, num_nodes)
            self.declare_partials(states_name, mtx_name, rows=rows, cols=cols)

            arange = np.arange(size)
            _, rows, cols = tile_sparse_jac(1., arange, arange,
                size, size, num_nodes)
            self.declare_partials(states_name, forces_name, val=-1., rows=rows, cols=cols)

            rows = np.outer(np.arange(size), np.ones(size, int)).flatten()
            cols = np.outer(np.ones(size, int), np.arange(size)).flatten()
            _, rows, cols = tile_sparse_jac(1., rows, cols,
                size, size, num_nodes)
            self.declare_partials(states_name, states_name, rows=rows, cols=cols)

        self.lu = {}

        self.solvers = {}
        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            for i in range(num_nodes):
                self.solvers[lifting_surface_name, i] = OASLinearSolver()

    def apply_nonlinear(self, inputs, outputs, residuals):
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            forces_name = '{}_forces'.format(lifting_surface_name)
            mtx_name = '{}_global_stiff'.format(lifting_surface_name)
            states_name = '{}_states'.format(lifting_surface_name)

            residuals[states_name] = np.einsum('ijk,ik->ij', inputs[mtx_name], outputs[states_name]) \
                - inputs[forces_name]

    def solve_nonlinear(self, inputs, outputs):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            forces_name = '{}_forces'.format(lifting_surface_name)
            mtx_name = '{}_global_stiff'.format(lifting_surface_name)
            states_name = '{}_states'.format(lifting_surface_name)

            for i in range(num_nodes):
                # self.lu[lifting_surface_name, i] = lu = lu_factor(inputs[mtx_name][i, :, :])
                # outputs[states_name][i, :] = lu_solve(lu, inputs[forces_name][i, :])

                mtx = inputs[mtx_name][i, :, :]
                solver = self.solvers[lifting_surface_name, i]
                solver.mtx = mtx
                solver.lu = lu_factor(mtx)
                outputs[states_name][i, :] = solver.solve(inputs[forces_name][i, :])

    def linearize(self, inputs, outputs, partials):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            size = 6 * num_points_z + 6

            forces_name = '{}_forces'.format(lifting_surface_name)
            mtx_name = '{}_global_stiff'.format(lifting_surface_name)
            states_name = '{}_states'.format(lifting_surface_name)

            for i in range(num_nodes):
                # self.lu[lifting_surface_name, i] = lu_factor(inputs[mtx_name][i, :, :])

                mtx = inputs[mtx_name][i, :, :]
                solver = self.solvers[lifting_surface_name, i]
                solver.mtx = mtx
                solver.lu = lu_factor(mtx)

            partials[states_name, states_name] = inputs[mtx_name].flatten()
            partials[states_name, mtx_name] = np.einsum('j,ik->ijk',
                np.ones(size), outputs[states_name]).flatten()

    def solve_linear(self, d_outputs, d_residuals, mode):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            forces_name = '{}_forces'.format(lifting_surface_name)
            mtx_name = '{}_global_stiff'.format(lifting_surface_name)
            states_name = '{}_states'.format(lifting_surface_name)

            if mode == 'fwd':
                for i in range(num_nodes):
                    # lu = self.lu[lifting_surface_name, i]
                    # d_outputs[states_name][i, :] = lu_solve(lu, d_residuals[states_name][i, :], trans=0)
                    solver = self.solvers[lifting_surface_name, i]
                    d_outputs[states_name][i, :] = solver.solve(d_residuals[states_name][i, :], mode='fwd')
            else:
                for i in range(num_nodes):
                    # lu = self.lu[lifting_surface_name, i]
                    # d_residuals[states_name][i, :] = lu_solve(lu, d_outputs[states_name][i, :], trans=1)
                    solver = self.solvers[lifting_surface_name, i]
                    d_residuals[states_name][i, :] = solver.solve(d_outputs[states_name][i, :], mode='rev')

    def solve_multi_linear(self, d_outputs, d_residuals, mode):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            forces_name = '{}_forces'.format(lifting_surface_name)
            mtx_name = '{}_global_stiff'.format(lifting_surface_name)
            states_name = '{}_states'.format(lifting_surface_name)

            ncol = d_outputs[states_name].shape[-1]

            if mode == 'fwd':
                for i in range(num_nodes):
                    for j in range(ncol):
                        # lu = self.lu[lifting_surface_name, i]
                        # d_outputs[states_name][i, :] = lu_solve(lu, d_residuals[states_name][i, :], trans=0)
                        solver = self.solvers[lifting_surface_name, i]
                        d_outputs[states_name][i, :, j] = solver.solve(d_residuals[states_name][i, :, j], mode='fwd')
            else:
                for i in range(num_nodes):
                    for j in range(ncol):
                        # lu = self.lu[lifting_surface_name, i]
                        # d_residuals[states_name][i, :] = lu_solve(lu, d_outputs[states_name][i, :], trans=1)
                        solver = self.solvers[lifting_surface_name, i]
                        d_residuals[states_name][i, :, j] = solver.solve(d_outputs[states_name][i, :, j], mode='rev')
