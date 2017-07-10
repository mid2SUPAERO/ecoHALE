from __future__ import print_function, division
import numpy as np

from openmdao.api import ExplicitComponent

try:
    from openaerostruct.fortran import OAS_API
    fortran_flag = True
    data_type = float
except:
    fortran_flag = False
    data_type = complex

class CreateRHS(ExplicitComponent):
    """
    Compute the right-hand-side of the K * u = f linear system to solve for the displacements.
    The RHS is based on the loads. For the aerostructural case, these are
    recomputed at each design point based on the aerodynamic loads.

    Parameters
    ----------
    loads[ny, 6] : numpy array
        Flattened array containing the loads applied on the FEM component,
        computed from the sectional forces.

    Returns
    -------
    forces[6*(ny+1)] : numpy array
        Right-hand-side of the linear system. The loads from the aerodynamic
        analysis or the user-defined loads.
    """

    def initialize(self):
        self.metadata.declare('surface', type_=dict)

    def setup(self):
        surface = self.metadata['surface']

        self.ny = surface['num_y']

        self.add_input('loads', val=np.random.random_sample((self.ny, 6)))# dtype=data_type))
        self.add_output('forces', val=np.random.random_sample(((self.ny+1)*6)))# dtype=data_type))

        n = self.ny * 6
        forces_loads = np.zeros((n + 6, n))
        forces_loads[:n, :n] = np.eye((n))

        self.declare_partials('forces', 'loads', val=forces_loads)

    def compute(self, inputs, outputs):
        # Populate the right-hand side of the linear system using the
        # prescribed or computed loads
        outputs['forces'][:6*self.ny] = inputs['loads'].reshape(self.ny*6)

        # Remove extremely small values from the RHS so the linear system
        # can more easily be solved
        outputs['forces'][np.abs(outputs['forces']) < 1e-6] = 0.
