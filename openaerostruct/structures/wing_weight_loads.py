from __future__ import division, print_function
import numpy as np

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

        self.add_input('element_weights', val=np.zeros((self.ny-1), dtype=complex), units='N')
        self.add_input('nodes', val=np.zeros((self.ny, 3), dtype=complex), units='m')
        self.add_input('load_factor', val=1.05)
        self.add_output('struct_weight_loads', val=np.zeros((self.ny, 6), dtype=complex), units='N')

        self.declare_partials('struct_weight_loads', 'nodes',  method='cs')

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

        loads = np.zeros((self.ny, 6), dtype=complex)
        if self.under_complex_step: # Why doesn't this trigger when running test_aerostruct_wingbox_+weight_analysis.py???
            loads = np.zeros((self.ny, 6), dtype=complex)

        # Loads in z-direction
        loads[:-1, 2] += -z_forces_for_each
        loads[1:, 2] += -z_forces_for_each

        # Bending moments for consistency
        bm = z_moments_for_each * deltas[: , 1] / element_lengths
        loads[:-1, 3] += -bm
        loads[1:, 3] += bm

        bm = z_moments_for_each * deltas[: , 0] / element_lengths
        loads[:-1, 4] += -bm
        loads[1:, 4] += bm
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
        dswl__dew = inputs['load_factor'] * element_lengths / 12. * \
                    (deltas[:, 0]**2 + deltas[:, 1]**2)**0.5 / element_lengths * \
                    deltas[: , 1] / element_lengths
        J['struct_weight_loads', 'element_weights'][2*nym1:3*nym1] = -dswl__dew
        J['struct_weight_loads', 'element_weights'][3*nym1:4*nym1] = dswl__dew


        J['struct_weight_loads', 'load_factor'][:nym1] = -inputs['element_weights']/2.
        J['struct_weight_loads', 'load_factor'][1:self.ny] += -inputs['element_weights']/2

        dswl__dlf = inputs['element_weights'] * element_lengths / 12. * \
                    (deltas[:, 0]**2 + deltas[:, 1]**2)**0.5 / element_lengths * \
                    deltas[: , 1] / element_lengths
        J['struct_weight_loads', 'load_factor'][self.ny:self.ny+nym1] = -dswl__dlf
        J['struct_weight_loads', 'load_factor'][self.ny+1:self.ny+nym1+1] += dswl__dlf


        dswl__dlf = inputs['element_weights'] * element_lengths / 12. * \
                    (deltas[:, 0]**2 + deltas[:, 1]**2)**0.5 / element_lengths * \
                    deltas[: , 0] / element_lengths
        J['struct_weight_loads', 'load_factor'][2*self.ny:2*self.ny+nym1] += -dswl__dlf
        J['struct_weight_loads', 'load_factor'][2*self.ny+1:2*self.ny+nym1+1] += dswl__dlf



