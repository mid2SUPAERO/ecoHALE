import numpy as np
from scipy.sparse import csc_matrix

from openmdao.api import ExplicitComponent

from openaerostruct.utils.bsplines import get_bspline_mtx
from openaerostruct.utils.misc_utils import tile_sparse_jac


class BsplineComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)
        self.metadata.declare('num_control_points', types=int)
        self.metadata.declare('num_points', types=int)
        self.metadata.declare('bspline_order', types=int)
        self.metadata.declare('in_name', types=str)
        self.metadata.declare('out_name', types=str)
        self.metadata.declare('distribution', types=str)

    def setup(self):
        meta = self.metadata
        num_nodes = meta['num_nodes']
        num_control_points = meta['num_control_points']
        num_points = meta['num_points']
        bspline_order = meta['bspline_order']
        in_name = meta['in_name']
        out_name = meta['out_name']
        distribution = meta['distribution']

        self.add_input(in_name, val=np.random.rand(num_nodes, num_control_points))
        self.add_output(out_name, val=np.random.rand(num_nodes, num_points))

        jac = get_bspline_mtx(num_control_points, num_points,
            order=bspline_order, distribution=distribution).tocoo()

        data, rows, cols = tile_sparse_jac(jac.data, jac.row, jac.col,
            num_points, num_control_points, num_nodes)

        self.jac = csc_matrix((data, (rows, cols)),
            shape=(num_nodes * num_points, num_nodes * num_control_points))

        self.declare_partials(out_name, in_name, val=data, rows=rows, cols=cols)

        self.set_check_partial_options('*', method='cs')

    def compute(self, inputs, outputs):
        meta = self.metadata
        num_nodes = meta['num_nodes']
        num_control_points = meta['num_control_points']
        num_points = meta['num_points']

        out_shape = (num_nodes, num_points)
        in_shape = (num_nodes, num_control_points)

        out = self.jac * inputs[meta['in_name']].reshape(np.prod(in_shape))
        outputs[meta['out_name']] = out.reshape(out_shape)
