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

        self.declare_partials('*', '*',  method='cs')

    def compute(self, inputs, outputs):

        struct_weights = inputs['element_weights'] * inputs['load_factor']
        nodes = inputs['nodes']

        element_lengths = np.ones(self.ny - 1, dtype=complex)
        for i in range(self.ny - 1):
            element_lengths[i] = norm(nodes[i+1] - nodes[i])

        print(nodes.shape)
        print(element_lengths)
        # And we also need the deltas between consecutive nodes
        deltas = nodes[1:, :] - nodes[:-1, :]
        print(deltas)
        exit()
        # Assume weight coincides with the elastic axis
        z_forces_for_each = struct_weights / 2.
        z_moments_for_each = struct_weights * element_lengths / 12. \
                            * (deltas[:, 0]**2 + deltas[:, 1]**2)**0.5 / element_lengths

        loads = np.zeros((self.ny, 6), dtype=complex)

        # Loads in z-direction
        loads[:-1, 2] += -z_forces_for_each
        loads[1:, 2] += -z_forces_for_each

        # Bending moments for consistency
        loads[:-1, 3] += -z_moments_for_each * deltas[: , 1] / element_lengths
        loads[1:, 3] += z_moments_for_each * deltas[: , 1] / element_lengths

        loads[:-1, 4] += -z_moments_for_each * deltas[: , 0] / element_lengths
        loads[1:, 4] += z_moments_for_each * deltas[: , 0] / element_lengths

        outputs['struct_weight_loads'] = loads
