from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent

from openaerostruct.utils.misc_utils import tile_sparse_jac


class VLMFreestreamVelComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)
        self.metadata.declare('size', types=int)

    def setup(self):
        num_nodes = self.metadata['num_nodes']
        size = self.metadata['size']

        self.add_input('alpha_rad', shape=num_nodes)
        self.add_input('v_m_s', shape=num_nodes)
        self.add_output('freestream_vel', shape=(num_nodes, size, 3))

        rows = np.arange(num_nodes * size * 3)
        cols = np.outer(np.arange(num_nodes), np.ones(size * 3)).flatten()
        self.declare_partials('freestream_vel', 'alpha_rad', rows=rows, cols=cols)
        self.declare_partials('freestream_vel', 'v_m_s', rows=rows, cols=cols)

        self.set_check_partial_options('*', method='cs')

    def compute(self, inputs, outputs):
        num_nodes = self.metadata['num_nodes']
        size = self.metadata['size']

        ones = np.ones(size)
        alpha_rad = np.outer(inputs['alpha_rad'], ones)
        v_m_s = np.outer(inputs['v_m_s'], ones)

        outputs['freestream_vel'][:, :, 0] = v_m_s * np.cos(alpha_rad)
        outputs['freestream_vel'][:, :, 1] = v_m_s * np.sin(alpha_rad)
        outputs['freestream_vel'][:, :, 2] = 0.

    def compute_partials(self, inputs, partials):
        num_nodes = self.metadata['num_nodes']
        size = self.metadata['size']

        ones = np.ones(size)
        alpha_rad = np.outer(inputs['alpha_rad'], ones)
        v_m_s = np.outer(inputs['v_m_s'], ones)

        derivs = partials['freestream_vel', 'v_m_s'].reshape((num_nodes, size, 3))
        derivs[:, :, 0] = np.cos(alpha_rad)
        derivs[:, :, 1] = np.sin(alpha_rad)
        derivs[:, :, 2] = 0.

        derivs = partials['freestream_vel', 'alpha_rad'].reshape((num_nodes, size, 3))
        derivs[:, :, 0] = -v_m_s * np.sin(alpha_rad)
        derivs[:, :, 1] =  v_m_s * np.cos(alpha_rad)
        derivs[:, :, 2] = 0.
