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
        self.options.declare('surface', types=dict)

    def setup(self):
        surface = self.options['surface']

        self.ny = surface['num_y']

        self.add_input('loads', val=np.zeros((self.ny, 6)), units='N')# dtype=data_type))
        self.add_input('element_weights', val=np.ones((self.ny-1)), units='N')# dtype=data_type))
        self.add_output('forces', val=np.ones(((self.ny+1)*6)), units='N')# dtype=data_type))

        n = self.ny * 6
        forces_loads = np.zeros((n + 6, n))
        forces_loads[:n, :n] = np.eye((n))

        self.declare_partials('forces', 'loads', val=forces_loads)

        rows = np.arange(2, (self.ny-1)*6, 6)
        rows = np.hstack((rows, rows+6))
        cols = np.arange(self.ny-1)
        cols = np.hstack((cols, cols))
        # self.declare_partials('forces', 'element_weights', val=-.5, rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        outputs['forces'][:] = 0.
        # outputs['forces'][:6*(self.ny-1)][2::6] -= inputs['element_weights'] / 2
        # outputs['forces'][:6*self.ny][8::6] -= inputs['element_weights'] / 2

        # Populate the right-hand side of the linear system using the
        # prescribed or computed loads
        outputs['forces'][:6*self.ny] += inputs['loads'].reshape(self.ny*6)

        # Remove extremely small values from the RHS so the linear system
        # can more easily be solved
        outputs['forces'][np.abs(outputs['forces']) < 1e-6] = 0.
