import numpy as np
from scipy.sparse import csc_matrix

from openmdao.api import ExplicitComponent


class ScalarExpansionComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('shape', types=tuple)
        self.metadata.declare('in_name', types=str)
        self.metadata.declare('out_name', types=str)

    def setup(self):
        shape = self.metadata['shape']
        in_name = self.metadata['in_name']
        out_name = self.metadata['out_name']

        self.add_input(in_name)
        self.add_output(out_name, shape=shape)

        size = np.prod(shape)

        rows = np.arange(size)
        cols = np.zeros(size, int)
        self.declare_partials(out_name, in_name, val=1., rows=rows, cols=cols)

        self.set_check_partial_options('*', method='cs')

    def compute(self, inputs, outputs):
        in_name = self.metadata['in_name']
        out_name = self.metadata['out_name']

        outputs[out_name][:] = inputs[in_name]
