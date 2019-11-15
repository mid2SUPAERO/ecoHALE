from __future__ import division, print_function
import numpy as np

from openmdao.api import ExplicitComponent


def norm(vec):
    return np.sqrt(np.sum(vec**2))

class PVLoads(ExplicitComponent):
    """
    Compute the nodal loads from the distributed PV cells within the wing
    to be applied to the wing structure.

    Parameters
    ----------

    Returns
    -------

    """

    def initialize(self):
        self.options.declare('surface', types=dict)

    def setup(self):
        self.surface = surface = self.options['surface']
        self.ny = surface['mesh'].shape[1]

#        self.add_input('fuel_vols', val=np.ones((self.ny-1)), units='m**3')
        self.add_input('PV_areas', val=np.ones((self.ny-1)), units='m**2')
        self.add_input('nodes', val=np.zeros((self.ny, 3)), units='m')
#        self.add_input('fuel_mass', val=1., units='kg')
        self.add_input('PV_mass', val=1., units='kg')
        self.add_input('load_factor', val=1.)
        self.add_output('PV_weight_loads', val=np.zeros((self.ny, 6)), units='N')

        self.declare_partials('*', '*',  method='cs')

    def compute(self, inputs, outputs):
        nodes = inputs['nodes']

        element_lengths = np.ones(self.ny - 1, dtype=complex)
        for i in range(self.ny - 1):
            element_lengths[i] = norm(nodes[i+1] - nodes[i])

        # And we also need the deltas between consecutive nodes
        deltas = nodes[1:, :] - nodes[:-1, :]

        # Fuel weight
        PV_weight = (inputs['PV_mass']) * 9.81 * inputs['load_factor']

        if self.surface['symmetry']:
            PV_weight /= 2.

        areas = inputs['PV_areas']
        sum_areas = np.sum(areas)

        # Now we need the PV weight per segment
        # Assume it's divided evenly based on areas
        z_weights = areas * PV_weight / sum_areas
        
        # Assume weight coincides with the elastic axis
        z_forces_for_each = z_weights / 2.
        z_moments_for_each = z_weights * element_lengths / 12. * (deltas[:, 0]**2 + deltas[:,1]**2)**0.5 / element_lengths

        loads = np.zeros((self.ny, 6), dtype=complex)

        # Loads in z-direction  #half of the weight on each node of an element
        loads[:-1, 2] = loads[:-1, 2] - z_forces_for_each
        loads[1:, 2] = loads[1:, 2] - z_forces_for_each

        # Bending moments for consistency
        loads[:-1, 3] = loads[:-1, 3] - z_moments_for_each * deltas[: , 1] / element_lengths
        loads[1:, 3] = loads[1:, 3] + z_moments_for_each * deltas[: , 1] / element_lengths

        loads[:-1, 4] = loads[:-1, 4] - z_moments_for_each * deltas[: , 0] / element_lengths
        loads[1:, 4] = loads[1:, 4] + z_moments_for_each * deltas[: , 0] / element_lengths

        outputs['PV_weight_loads'] = loads