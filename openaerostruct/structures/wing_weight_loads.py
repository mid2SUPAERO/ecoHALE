from __future__ import division, print_function
import numpy as np

from scipy.sparse import coo_matrix, diags

from openmdao.api import ExplicitComponent
from openaerostruct.structures.utils import norm


class StructureWeightLoads(ExplicitComponent):
    """
    Compute the nodal loads from the weight of the wing structure to be applied to the wing
    structure.

    Parameters
    ----------
    element_weights[ny-1] : numpy array
        Weight for each wing-structure segment.
    nodes[ny, 3] : numpy array
        Flattened array with coordinates for each FEM node.

    Returns
    -------
    struct_weight_loads[ny, 6] : numpy array
        Flattened array containing the loads applied on the FEM component,
        computed from the weight of the wing-structure segments.
    """

    def initialize(self):
        self.options.declare('surface', types=dict)

    def setup(self):
        self.surface = surface = self.options['surface']
        self.ny = surface['num_y']

        self.add_input('element_weights', val=np.zeros((self.ny-1)), units='N')
        self.add_input('nodes', val=np.zeros((self.ny, 3), dtype=complex), units='m')
        self.add_input('load_factor', val=1.05)
        self.add_output('struct_weight_loads', val=np.zeros((self.ny, 6)), units='N')

        self.add_output('element_lengths', val=np.zeros(self.ny-1), units='N')

        self.add_output('bm3', val=np.zeros(self.ny-1), units='N')
        self.add_output('d0', val=np.zeros(self.ny-1), units='N')
        self.add_output('d1', val=np.zeros(self.ny-1), units='N')
        self.add_output('z_moments_for_each', val=np.zeros(self.ny-1), units='N')

        # self.declare_partials('struct_weight_loads', 'nodes',  method='cs')
        self.declare_partials('struct_weight_loads', 'nodes')

        nym1 = self.ny-1
        rows = np.zeros(4*nym1)
        rows[:nym1] = 2+np.arange(nym1)*6
        rows[nym1:2*nym1] = 2+np.arange(1,self.ny)*6
        rows[2*nym1:2*nym1+nym1] = rows[:nym1]+1
        rows[2*nym1+nym1:] = rows[nym1:2*nym1]+1

        cols = np.zeros(4*nym1)
        c = np.arange(nym1)
        cols[:nym1] = c
        cols[nym1:2*nym1] = c
        cols[2*nym1:2*nym1+nym1] = c
        cols[2*nym1+nym1:] = c

        self.declare_partials('struct_weight_loads', 'element_weights', rows=rows, cols=cols)

        rows = np.zeros(3*self.ny)
        cols = np.zeros(3*self.ny)
        for j in range(3):
            for i in range(self.ny):
                idx = i+self.ny*j
                rows[idx] = 6*i+j+2

        self.declare_partials('struct_weight_loads', 'load_factor', rows=rows, cols=cols)

        self.declare_partials('bm3', 'nodes')
        self.declare_partials('d0', 'nodes')
        self.declare_partials('d1', 'nodes')
        self.declare_partials('z_moments_for_each', 'nodes')



    def compute(self, inputs, outputs):

        struct_weights = inputs['element_weights'] * inputs['load_factor']
        nodes = inputs['nodes']

        element_lengths = norm(nodes[1:, :] - nodes[:-1, :], axis=1)


        # And we also need the deltas between consecutive nodes
        deltas = nodes[1:, :] - nodes[:-1, :]

        # Assume weight coincides with the elastic axis
        z_forces_for_each = struct_weights / 2.
        z_moments_for_each = struct_weights / 12. \
                            * (deltas[:, 0]**2 + deltas[:, 1]**2)**0.5

        outputs['z_moments_for_each'] = z_moments_for_each

        loads = np.zeros((self.ny, 6), dtype=complex)
        # Why doesn't this trigger when running ../../tests/test_aerostruct_wingbox_+weight_analysis.py???
        if self.under_complex_step:
            loads = np.zeros((self.ny, 6), dtype=complex)

        # Loads in z-direction
        loads[:-1, 2] += -z_forces_for_each
        loads[1:, 2] += -z_forces_for_each

        # Bending moments for consistency
        bm3 = z_moments_for_each * deltas[: , 1] / element_lengths
        loads[:-1, 3] += -bm3
        loads[1:, 3] += bm3

        outputs['bm3'] = bm3
        outputs['d0'] = deltas[: , 0]
        outputs['d1'] = deltas[: , 1]

        bm4 = z_moments_for_each * deltas[: , 0] / element_lengths
        loads[:-1, 4] += -bm4
        loads[1:, 4] += bm4
        outputs['struct_weight_loads'] = loads



    def compute_partials(self, inputs, J):

        struct_weights = inputs['element_weights'] * inputs['load_factor']
        nodes = inputs['nodes']

        nym1 = self.ny-1

        element_lengths = norm(nodes[1:, :] - nodes[:-1, :], axis=1)

        # And we also need the deltas between consecutive nodes
        deltas = nodes[1:, :] - nodes[:-1, :]

        # Assume weight coincides with the elastic axis
        z_forces_for_each = struct_weights / 2.
        z_moments_for_each = struct_weights / 12. \
                            * (deltas[:, 0]**2 + deltas[:, 1]**2)**0.5

        J['struct_weight_loads', 'element_weights'][:2*nym1] = -inputs['load_factor']/2.
        dswl__dew = inputs['load_factor'] / 12. * \
                    (deltas[:, 0]**2 + deltas[:, 1]**2)**0.5 * \
                    deltas[: , 1] / element_lengths
        J['struct_weight_loads', 'element_weights'][2*nym1:3*nym1] = -dswl__dew
        J['struct_weight_loads', 'element_weights'][3*nym1:4*nym1] = dswl__dew


        J['struct_weight_loads', 'load_factor'][:nym1] = -inputs['element_weights']/2.
        J['struct_weight_loads', 'load_factor'][1:self.ny] += -inputs['element_weights']/2

        dswl__dlf = inputs['element_weights'] / 12. * \
                    (deltas[:, 0]**2 + deltas[:, 1]**2)**0.5 * \
                    deltas[: , 1] / element_lengths
        J['struct_weight_loads', 'load_factor'][self.ny:self.ny+nym1] = -dswl__dlf
        J['struct_weight_loads', 'load_factor'][self.ny+1:self.ny+nym1+1] += dswl__dlf


        dswl__dlf = inputs['element_weights'] / 12. * \
                    (deltas[:, 0]**2 + deltas[:, 1]**2)**0.5 * \
                    deltas[: , 0] / element_lengths
        J['struct_weight_loads', 'load_factor'][2*self.ny:2*self.ny+nym1] += -dswl__dlf
        J['struct_weight_loads', 'load_factor'][2*self.ny+1:2*self.ny+nym1+1] += dswl__dlf


        # TODO: JSG - Finish these
        # ddel__dnodes
        rows = np.zeros(2*nym1)
        cols_d0 = np.zeros(2*nym1)
        cols_d1 = np.zeros(2*nym1)
        data = np.zeros(2*nym1)

        counter = 0
        for i in range(nym1):
            for j in range(2):
                rows[counter] = i
                cols_d0[counter] = 3*(i+j)
                cols_d1[counter] = 3*(i+j)+1
                data[counter] = 2*j-1
                counter += 1

        shape = (3,3*self.ny)
        ddel0__dnodes = coo_matrix((data, (rows, cols_d0)), shape=shape).tocsr()
        ddel1__dnodes = coo_matrix((data, (rows, cols_d1)), shape=shape).tocsr()

        # del__dnodes
        rows = np.zeros(6*nym1)
        cols = np.zeros(6*nym1)
        data = np.ones(6*nym1)
        raw_data = diags(1/element_lengths)*(nodes[1:, :] - nodes[:-1, :])
        data = np.hstack((-raw_data,raw_data)).flatten()
        counter = 0
        for i in range(nym1):
            for j in range(6):
                rows[counter] = i
                cols[counter] = 3*i+j
                counter += 1

        del__dnodes = coo_matrix((data, (rows, cols)), shape=shape).tocsr()

        del0 = deltas[: , 0]
        del1 = deltas[: , 1]

        dzm_dnodes = diags(struct_weights/12*(del0**2 + del1**2)**-.5)*\
                     (diags(del0)*ddel0__dnodes  + diags(del1)*ddel1__dnodes)

        bm3 = z_moments_for_each * deltas[: , 1] / element_lengths

        dbm3_dnodes = diags(del1/element_lengths)*dzm_dnodes \
                    + (diags(z_moments_for_each/element_lengths)*ddel1__dnodes - diags(z_moments_for_each*del1/element_lengths**2)*del__dnodes)

        J['bm3', 'nodes'] = dbm3_dnodes.todense()
        J['z_moments_for_each', 'nodes'] = dzm_dnodes.todense()
        J['d0', 'nodes'] = ddel0__dnodes.todense()
        J['d1', 'nodes'] = ddel1__dnodes.todense()


        dbm4_dnodes = diags(del0/element_lengths)*dzm_dnodes \
                    + diags(z_moments_for_each)*\
                    (diags(1/element_lengths)*ddel0__dnodes - diags(del0/element_lengths**2)*del__dnodes)

        #dswl__dbm1
        rows_1 = np.zeros(2*nym1)
        rows_0 = np.zeros(2*nym1)
        cols = np.zeros(2*nym1)
        data = np.zeros(2*nym1)
        counter = 0
        for i in range(nym1):
            for j in range(2):
                rows_1[counter] = 6*(i+j)+3
                rows_0[counter] = 6*(i+j)+4
                if counter%2==0: #even
                    data[counter] = 1
                else:
                    data[counter] = -1
                cols[counter] = i
                counter += 1

        dswl__dbm3 = coo_matrix((data, (rows_1, cols)), shape=(6*self.ny,nym1)).tocsr()
        dswl__dbm4 = coo_matrix((data, (rows_0, cols)), shape=(6*self.ny,nym1)).tocsr()
        # print(dswl__dbm1)
        np.set_printoptions(linewidth=400)
        print()
        dswl__dbm = (dswl__dbm3*dbm3_dnodes+dswl__dbm4*dbm4_dnodes).tocoo()
        J['struct_weight_loads', 'nodes'] = -dswl__dbm.todense()
