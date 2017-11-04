import numpy as np
from scipy.sparse import csc_matrix

from openmdao.api import ExplicitComponent


class StaticDVComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('num_nodes', type_=int)
        self.metadata.declare('num_points', type_=int)
        self.metadata.declare('in_name', type_=str)
        self.metadata.declare('out_name', type_=str)

    def setup(self):
        meta = self.metadata
        num_nodes = meta['num_nodes']
        num_points = meta['num_points']
        in_name = meta['in_name']
        out_name = meta['out_name']

        self.add_input(in_name, val=np.random.rand(num_points))
        self.add_output(out_name, val=np.random.rand(num_nodes, num_points))

        rows = np.arange(num_nodes * num_points)
        cols = np.outer(np.ones(num_nodes, int), np.arange(num_points)).flatten()
        self.declare_partials(out_name, in_name, val=1., rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        meta = self.metadata
        num_nodes = meta['num_nodes']
        in_name = meta['in_name']
        out_name = meta['out_name']

        outputs[out_name] = np.outer(np.ones(num_nodes), inputs[in_name])
