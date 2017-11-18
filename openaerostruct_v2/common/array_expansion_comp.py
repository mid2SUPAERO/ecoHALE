import numpy as np
from scipy.sparse import csc_matrix

from openmdao.api import ExplicitComponent

from openaerostruct_v2.utils.misc_utils import get_array_indices, get_array_expansion_data


class ArrayExpansionComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('shape', types=tuple)
        self.metadata.declare('expand_indices', types=list)
        self.metadata.declare('in_name', types=str)
        self.metadata.declare('out_name', types=str)

    def setup(self):
        shape = self.metadata['shape']
        expand_indices = self.metadata['expand_indices']
        in_name = self.metadata['in_name']
        out_name = self.metadata['out_name']

        einsum_string, in_shape, out_shape, ones_shape = get_array_expansion_data(shape, expand_indices)

        self.add_input(in_name, shape=in_shape)
        self.add_output(out_name, shape=out_shape)

        in_indices = get_array_indices(*in_shape)
        out_indices = get_array_indices(*out_shape)

        self.einsum_string = einsum_string
        self.ones_shape = ones_shape

        rows = out_indices.flatten()
        cols = np.einsum(einsum_string, in_indices, np.ones(ones_shape, int)).flatten()
        self.declare_partials(out_name, in_name, val=1., rows=rows, cols=cols)

        self.set_check_partial_options('*', method='cs')

    def compute(self, inputs, outputs):
        in_name = self.metadata['in_name']
        out_name = self.metadata['out_name']

        outputs[out_name] = np.einsum(self.einsum_string, inputs[in_name], np.ones(self.ones_shape))
