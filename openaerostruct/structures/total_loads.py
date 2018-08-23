from __future__ import division, print_function
import numpy as np

from openmdao.api import ExplicitComponent


class TotalLoads(ExplicitComponent):
    """
    Add the loads from the aerodynamics, structural weight, and fuel weight.

    Parameters
    ----------
    loads[ny, 6] : numpy array
        Array containing the loads applied on the FEM component,
        computed from the sectional forces.
    struct_weight_loads[ny, 6] : numpy array
        Array containing the loads applied on the FEM component,
        computed from the weight of the wing-structure segments.
    fuel_weight_loads[ny, 6] : numpy array
        Array containing the loads applied on the FEM component,
        computed from the weight of the fuel.

    Returns
    -------
    total_loads[ny, 6] : numpy array
        Flattened array containing the total loads applied on the FEM.
    """

    def initialize(self):
        self.options.declare('surface', types=dict)

    def setup(self):
        self.surface = surface = self.options['surface']
        self.ny = surface['num_y']

        self.add_input('loads', val=np.ones((self.ny, 6)), units='N')
        self.add_input('struct_weight_loads', val=np.zeros((self.ny, 6)), units='N')
        self.add_input('fuel_weight_loads', val=np.zeros((self.ny, 6)), units='N')

        self.add_output('total_loads', val=np.ones((self.ny, 6)), units='N')

        arange = np.arange(self.ny * 6)

        self.declare_partials('total_loads', 'loads',
            rows=arange, cols=arange, val=1.)

        if self.surface['struct_weight_relief']:
            self.declare_partials('total_loads', 'struct_weight_loads',
                rows=arange, cols=arange, val=1.)

        if self.surface['distributed_fuel_weight']:
            self.declare_partials('total_loads', 'fuel_weight_loads',
                rows=arange, cols=arange, val=1.)

    def compute(self, inputs, outputs):
        outputs['total_loads'] = inputs['loads']

        if self.surface['struct_weight_relief']:
            outputs['total_loads'] += inputs['struct_weight_loads']

        if self.surface['distributed_fuel_weight']:
            outputs['total_loads'] += inputs['fuel_weight_loads']
