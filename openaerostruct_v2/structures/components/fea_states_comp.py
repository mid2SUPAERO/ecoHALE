from __future__ import print_function
import numpy as np
from scipy.linalg import lu_factor, lu_solve

from openmdao.api import ImplicitComponent


class FEAStatesComp(ImplicitComponent):

    def initialize(self):
        self.metadata.declare('lifting_surfaces', type_=list)

    def setup(self):
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            size = 6 * num_points_z + 6

            forces_name = '{}_forces'.format(lifting_surface_name)
            mtx_name = '{}_global_stiff'.format(lifting_surface_name)
            states_name = '{}_states'.format(lifting_surface_name)

            self.add_input(mtx_name, shape=(size, size))
            self.add_input(forces_name, shape=size)
            self.add_output(states_name, shape=size)

            rows = np.outer(np.arange(size), np.ones(size, int)).flatten()
            cols = np.arange(size ** 2)
            self.declare_partials(states_name, mtx_name, rows=rows, cols=cols)

            arange = np.arange(size)
            self.declare_partials(states_name, forces_name, val=-1., rows=arange, cols=arange)

            self.declare_partials(states_name, states_name)

        self.lu = {}

    def apply_nonlinear(self, inputs, outputs, residuals):
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            forces_name = '{}_forces'.format(lifting_surface_name)
            mtx_name = '{}_global_stiff'.format(lifting_surface_name)
            states_name = '{}_states'.format(lifting_surface_name)

            residuals[states_name] = np.dot(inputs[mtx_name], outputs[states_name]) - inputs[forces_name]

    def solve_nonlinear(self, inputs, outputs):
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            forces_name = '{}_forces'.format(lifting_surface_name)
            mtx_name = '{}_global_stiff'.format(lifting_surface_name)
            states_name = '{}_states'.format(lifting_surface_name)

            self.lu[lifting_surface_name] = lu = lu_factor(inputs[mtx_name])

            outputs[states_name] = lu_solve(lu, inputs[forces_name])

    def linearize(self, inputs, outputs, partials):
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            size = 6 * num_points_z + 6

            forces_name = '{}_forces'.format(lifting_surface_name)
            mtx_name = '{}_global_stiff'.format(lifting_surface_name)
            states_name = '{}_states'.format(lifting_surface_name)

            self.lu[lifting_surface_name] = lu = lu_factor(inputs[mtx_name])

            partials[states_name, mtx_name] = np.outer(np.ones(size), outputs[states_name]).flatten()
            partials[states_name, states_name] = inputs[mtx_name]

    def solve_linear(self, d_outputs, d_residuals, mode):
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            forces_name = '{}_forces'.format(lifting_surface_name)
            mtx_name = '{}_global_stiff'.format(lifting_surface_name)
            states_name = '{}_states'.format(lifting_surface_name)

            lu = self.lu[lifting_surface_name]

            if mode == 'fwd':
                d_outputs[states_name] = lu_solve(lu, d_residuals[states_name], trans=0)
            else:
                d_residuals[states_name] = lu_solve(lu, d_outputs[states_name], trans=1)
