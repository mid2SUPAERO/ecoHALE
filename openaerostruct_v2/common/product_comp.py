from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent


class ProductComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('shape', types=tuple)
        self.metadata.declare('in_name1', types=str)
        self.metadata.declare('in_name2', types=str)
        self.metadata.declare('out_name', types=str)

    def setup(self):
        shape = self.metadata['shape']
        in_name1 = self.metadata['in_name1']
        in_name2 = self.metadata['in_name2']
        out_name = self.metadata['out_name']

        self.add_input(in_name1, shape=shape)
        self.add_input(in_name2, shape=shape)
        self.add_output(out_name, shape=shape)

        arange = np.arange(np.prod(shape))
        self.declare_partials(out_name, in_name1, rows=arange, cols=arange)
        self.declare_partials(out_name, in_name2, rows=arange, cols=arange)

        self.set_check_partial_options('*', method='cs')

    def compute(self, inputs, outputs):
        in_name1 = self.metadata['in_name1']
        in_name2 = self.metadata['in_name2']
        out_name = self.metadata['out_name']

        outputs[out_name] = inputs[in_name1] * inputs[in_name2]

    def compute_partials(self, inputs, partials):
        in_name1 = self.metadata['in_name1']
        in_name2 = self.metadata['in_name2']
        out_name = self.metadata['out_name']

        outputs[out_name] = inputs[in_name1] * inputs[in_name2]
