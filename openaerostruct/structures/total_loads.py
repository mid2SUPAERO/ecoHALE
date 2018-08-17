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

class TotalLoads(ExplicitComponent):
    """
    Add the loads from the aerodynamics, structural weight, and fuel weight.

    Parameters
    ----------
    loads[ny, 6] : numpy array
        Flattened array containing the loads applied on the FEM component,
        computed from the sectional forces.
    struct_weight_loads[ny, 6] : numpy array
        Flattened array containing the loads applied on the FEM component,
        computed from the weight of the wing-structure segments.
    fuel_weight_loads[ny, 6] : numpy array
        Flattened array containing the loads applied on the FEM component,
        computed from the weight of the fuel.

    Returns
    -------
    total_loads[ny, 6] : numpy array
        Flattened array containing the loads applied on the FEM component,
        computed from the weight of the wing-structure segments.
    """

    def initialize(self):
        self.options.declare('surface', types=dict)

    def setup(self):
        self.surface = surface = self.options['surface']

        self.add_input('loads', val=np.zeros((self.ny, 6)), units='N')
        self.add_input('struct_weight_loads', val=np.zeros((self.ny, 6)), units='N')

        self.add_output('total_loads', val=np.zeros((self.ny, 6)), units='N')

        self.declare_partials('*', '*',  method='cs')

    def compute(self, inputs, outputs):

        outputs['total_loads'] = inputs['loads'] + inputs['struct_weight_loads']
