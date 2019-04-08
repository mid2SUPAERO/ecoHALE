"""
Class definition for the NodeForces component.
"""
from __future__ import print_function

import numpy as np
from scipy.sparse import csr_matrix

from openmdao.api import ExplicitComponent


class NodeForces(ExplicitComponent):
    """
    Component that simply converts the forces on the panel to an equivalent set of forces at the
    nodes.

    Here we just assume the leading edge nodes take slightly more of the load so that the centroid
    ends up at the quarter chord. The corresponding weights are stored in the le_node_wt and
    te_node_wt options.

    """
    def initialize(self):
        self.options.declare('surfaces', types=list)
        self.options.declare('le_node_wt', default=0.75 * 0.5)
        self.options.declare('te_node_wt', default=0.25 * 0.5)

    def setup(self):
        surfaces = self.options['surfaces']
        le_node_wt = self.options['le_node_wt']
        te_node_wt = self.options['te_node_wt']

        for surface in surfaces:
            mesh = surface['mesh']
            nx = mesh.shape[0]
            ny = mesh.shape[1]
            name = surface['name']

            sec_forces_name = '{}_sec_forces'.format(name)
            node_forces_name = '{}_node_forces'.format(name)

            self.add_input(sec_forces_name, shape=(nx - 1, ny - 1, 3), units='N')

            # TODO: what should res_ref be when it was np.sqrt(self.comm.size)
            self.add_output(node_forces_name, val=np.zeros((nx, ny, 3)), units='N')

            # Sparse partials
            rowcol = np.arange(3*(ny-1))
            row2 = rowcol + 3

            rows1 = np.concatenate([rowcol, row2])
            cols1 = np.concatenate([rowcol, rowcol])

            le_rows = np.tile(rows1, nx-1) + np.repeat(3*ny*np.arange(nx-1), 6*(ny-1))
            te_le_cols = np.tile(cols1, nx-1) + np.repeat(3*(ny-1)*np.arange(nx-1), 6*(ny-1))

            te_rows = le_rows + 3 * ny

            rows = np.concatenate([le_rows, te_rows])
            cols = np.concatenate([te_le_cols, te_le_cols])

            nn = len(rows)
            nn2 = int(nn/2)
            vals = np.empty((nn, ))

            vals[:nn2] = le_node_wt
            vals[nn2:] = te_node_wt

            self.declare_partials(node_forces_name, sec_forces_name, rows=rows, cols=cols, val=vals)

    def compute(self, inputs, outputs):
        """
        Compute the forces on the nodes from the panel section force.
        """
        surfaces = self.options['surfaces']

        le_node_wt = self.options['le_node_wt']
        te_node_wt = self.options['te_node_wt']
        for surface in surfaces:
            mesh = surface['mesh']
            nx = mesh.shape[0]
            ny = mesh.shape[1]

            name = surface['name']
            sec_forces_name = '{}_sec_forces'.format(name)
            node_forces_name = '{}_node_forces'.format(name)

            sec_forces = inputs[sec_forces_name]

            outputs[node_forces_name][:] = 0.0
            outputs[node_forces_name][:-1, :-1, :] += sec_forces * le_node_wt
            outputs[node_forces_name][1:, :-1, :] += sec_forces * te_node_wt
            outputs[node_forces_name][1:, 1:, :] += sec_forces * te_node_wt
            outputs[node_forces_name][:-1, 1:, :] += sec_forces * le_node_wt
