import numpy as np
from openmdao.api import ExplicitComponent


class MultiCD(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('n_points', type_=int, required=True)

    def setup(self):
        self.n_points = self.metadata['n_points']
        for i in range(self.n_points):
            self.add_input(str(i) + '_CD', val=0.)

        self.add_output('CD', val=0.)

    def compute(self, inputs, outputs):
        outputs['CD'] = 0.
        for i in range(self.n_points):
            outputs['CD'] += inputs[str(i) + '_CD']

    def compute_partials(self, inputs, outputs, partials):
        for i in range(self.n_points):
            partials['CD', str(i) + '_CD'] = 1.

class GeomMatch(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('n_points', type_=int, required=True)
        self.metadata.declare('mx', type_=int, required=True)
        self.metadata.declare('my', type_=int, required=True)

    def setup(self):
        self.n_points = self.metadata['n_points']
        self.mx = self.metadata['mx']
        self.my = self.metadata['my']
        for i in range(self.n_points):
            self.add_input(str(i) + '_shape', val=np.zeros((self.mx, self.my)), units='m')

        self.add_output('shape_diff', val=np.zeros((self.mx, self.my * (self.n_points - 1))), units='m')

        self.approx_partials('*', '*')

    def compute(self, inputs, outputs):
        shape_0 = inputs['0_shape']

        for i in range(1, self.n_points):
            outputs['shape_diff'][:, (i-1)*self.my:i*self.my] = inputs[str(i) + '_shape'] - shape_0
