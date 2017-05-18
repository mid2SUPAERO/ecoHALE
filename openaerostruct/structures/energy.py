from __future__ import print_function, division
import numpy as np

from openmdao.api import ExplicitComponent

try:
    import OAS_API
    fortran_flag = True
    data_type = float
except:
    fortran_flag = False
    data_type = complex

class Energy(ExplicitComponent):
    """ Compute strain energy.

    Parameters
    ----------
    disp[ny, 6] : numpy array
        Actual displacement array formed by truncating disp_aug.
    loads[ny, 6] : numpy array
        Array containing the loads applied on the FEM component,
        computed from the sectional forces.

    Returns
    -------
    energy : float
        Total strain energy of the structural component.

    """

    def initialize(self):
        self.metadata.declare('surface', type_=dict)

    def initialize_variables(self):
        surface = self.metadata['surface']

        ny = surface['num_y']

        self.add_input('disp', val=np.ones((ny, 6), dtype=data_type))
        self.add_input('loads', val=np.ones((ny, 6), dtype=data_type))
        self.add_output('energy', val=0.)

    def compute(self, inputs, outputs):
        outputs['energy'] = np.sum(inputs['disp'] * inputs['loads'])

    def compute_partial_derivs(self, inputs, outputs, partials):
        partials['energy', 'disp'][0, :] = inputs['loads'].real.flatten()
        partials['energy', 'loads'][0, :] = inputs['disp'].real.flatten()
