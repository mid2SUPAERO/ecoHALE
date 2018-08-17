from __future__ import division, print_function
import numpy as np

from openmdao.api import ExplicitComponent

try:
    from openaerostruct.fortran import OAS_API
    fortran_flag = True
    data_type = float
except:
    fortran_flag = False
    data_type = complex

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
        self.add_input('nodes', val=np.zeros((self.ny, 3)), units='m')
        self.add_output('struct_weight_loads', val=np.zeros((self.ny, 6)), units='N')

        self.declare_partials('*', '*',  method='cs')

    def compute(self, inputs, outputs):

        struct_weights = inputs['element_weights']
        nodes = inputs['nodes']

        element_lengths = np.ones(self.ny - 1, dtype=complex)
        for i in range(self.ny - 1):
            element_lengths[i] = norm(nodes[i+1] - nodes[i])

        # And we also need the deltas between consecutive nodes
        deltas = nodes[1:, :] - nodes[:-1, :]

        # Assume weight coincides with the elastic axis
        z_forces_for_each = struct_weights / 2.
        z_moments_for_each = struct_weights * element_lengths / 12. \
                            * (deltas[:, 0]**2 + deltas[:, 1]**2)**0.5 / element_lengths

        loads = np.zeros((self.ny, 6), dtype=complex)

        # Loads in z-direction
        loads[:-1, 2] = -z_forces_for_each
        loads[1:, 2] = -z_forces_for_each

        # Bending moments for consistency
        loads[:-1, 3] = -z_moments_for_each * deltas[: , 1] / self.element_lengths
        loads[1:, 3] = z_moments_for_each * deltas[: , 1] / self.element_lengths
        
        loads[:-1, 4] = -z_moments_for_each * deltas[: , 0] / self.element_lengths
        loads[1:, 4] = z_moments_for_each * deltas[: , 0] / self.element_lengths
        
        unknowns['struct_weight_loads'] = loads
