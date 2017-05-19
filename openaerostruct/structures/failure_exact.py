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

class FailureExact(ExplicitComponent):
    """
    Output individual failure constraints on each FEM element.

    Parameters
    ----------
    vonmises[ny-1, 2] : numpy array
        von Mises stress magnitudes for each FEM element.

    Returns
    -------
    failure[ny-1, 2] : numpy array
        Array of failure conditions. Positive if element has failed.

    """

    def initialize(self):
        self.metadata.declare('surface', type_=dict)

    def initialize_variables(self):
        surface = self.metadata['surface']

        self.ny = surface['num_y']
        self.sigma = surface['yield']

        self.add_input('vonmises', val=np.random.random_sample((self.ny-1, 2)))
        self.add_output('failure', val=np.zeros((self.ny-1, 2)))

    def compute(self, inputs, outputs):
        outputs['failure'] = inputs['vonmises'] / self.sigma - 1

    def compute_partial_derivs(self, inputs, outputs, partials):
        partials['failure', 'vonmises'] = np.eye(((self.ny-1)*2)) / self.sigma
