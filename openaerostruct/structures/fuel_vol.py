from __future__ import print_function, division
import numpy as np

from openmdao.api import ExplicitComponent


def norm(vec):
    return np.sqrt(np.sum(vec**2))

class WingboxFuelVol(ExplicitComponent):
    """
    Create a constraint to ensure the wingbox has enough internal volume to store the required fuel.

    parameters
    ----------
    nodes[ny, 3] : numpy array
        Coordinates of FEM nodes.
    A_int[ny-1] : numpy array
        Internal volume of each wingbox segment.
    fuelburn : float
        Fuel weight

    Returns
    -------
    fuel_vol_delta : numpy array
        If the value is negative, then there isn't enough volume for the fuel.
    fuel_vols[ny-1] : numpy array
        The magnitude of each individual panel's fuel-carrying volumes.
    """

    def initialize(self):
        self.options.declare('surface', types=dict)

    def setup(self):
        self.surface = surface = self.options['surface']

        self.ny = surface['num_y']

        self.add_input('nodes', val=np.zeros((self.ny, 3)), units='m')
        self.add_input('A_int', val=np.zeros((self.ny-1)), units='m**2')
        self.add_output('fuel_vols', val=np.zeros((self.ny-1)), units='m**3')

        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        nodes = inputs['nodes']

        element_lengths = np.zeros(self.ny-1, dtype=type(nodes[0, 0]))

        for i in range(self.ny - 1):
            element_lengths[i] = norm(nodes[i+1] - nodes[i])

        # Next we multiply the element lengths with the A_int for the internal volumes of the wingobox segments
        vols = element_lengths * inputs['A_int']

        outputs['fuel_vols'] = vols
