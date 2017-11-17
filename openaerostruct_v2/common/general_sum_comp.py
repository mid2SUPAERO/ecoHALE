from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent


class GeneralSumComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('shape', types=tuple)
        self.metadata.declare('in_names', types=list)
        self.metadata.declare('out_name', types=str)

    def setup(self):
        shape = self.metadata['shape']
        in_names = self.metadata['in_names']
        out_name = self.metadata['out_name']

        for in_name in in_names:
            self.add_input(in_name, shape=shape, val=0.)

        self.add_output(out_name, shape=shape)

        arange = np.arange(np.prod(shape))
        for in_name in in_names:
            self.declare_partials(out_name, in_name, val=1., rows=arange, cols=arange)

        self.set_check_partial_options('*', method='cs')

    def compute(self, inputs, outputs):
        in_names = self.metadata['in_names']
        out_name = self.metadata['out_name']

        outputs[out_name] = 0.
        for in_name in in_names:
            outputs[out_name] += inputs[in_name]
