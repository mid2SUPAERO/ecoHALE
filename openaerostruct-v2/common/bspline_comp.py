import numpy as np

from openmdao.api import ExplicitComponent

from dep_mdo.utils.bsplines import get_bspline_mtx


class BsplinesComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('num_control_points', type_=int)
        self.metadata.declare('num_points', type_=int)
        self.metadata.declare('bspline_order', type_=int)
        self.metadata.declare('in_name', type_=str)
        self.metadata.declare('out_name', type_=str)
        self.metadata.declare('distribution', type_=str)

    def setup(self):
        meta = self.metadata
        num_control_points = meta['num_control_points']
        num_points = meta['num_points']
        bspline_order = meta['bspline_order']
        in_name = meta['in_name']
        out_name = meta['out_name']
        distribution = meta['distribution']

        self.add_input(in_name, val=np.random.random(meta['num_control_points']))
        self.add_output(out_name, val=np.random.random(meta['num_points']))

        self.jac = get_bspline_mtx(num_control_points, num_points,
            order=bspline_order, distribution=distribution)

        jac = self.jac.tocoo()
        self.declare_partials(out_name, in_name, val=jac.data, rows=jac.row, cols=jac.col)

    def compute(self, inputs, outputs):
        meta = self.metadata
        outputs[meta['out_name']] = self.jac * inputs[meta['in_name']]
