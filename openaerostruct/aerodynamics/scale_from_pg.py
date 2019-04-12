from __future__ import print_function, division
import numpy as np

from openmdao.api import ExplicitComponent

class ScaleFromPrandtlGlauert(ExplicitComponent):
    """
    Scale the Prandtl-Glauert transformed forces to get the physical forces
    Prandtl-Glauert transformed geometry.

    The inverse Prandtl-Glauert transformation for forces is defined as below:
        F_x_wind = F_x_pg/B^4
        F_y_wind = F_y_pg/B^3
        F_z_wind = F_z_pg/B^3

    where B = sqrt(1 - M^2).

    Parameters
    ----------
    sec_forces_pg[nx-1, ny-1, 3] : numpy array
        Force vectors on each panel (lattice) in PG domain.
    node_forces_pg[nx, ny, 3] : numpy array
        Equivilent force vector on each panel (lattice) node in PG domain.

    M : float
        Freestream Mach number.

    Returns
    -------
    sec_forces_w_frame[nx-1, ny-1, 3] : numpy array
        Force vectors on each panel (lattice) in wind frame.
    node_forces_w_frame[nx, ny, 3] : numpy array
        Equivilent force vector on each panel (lattice) node in wind frame.
    """

    def initialize(self):
        self.options.declare('surface', types=dict)

    def setup(self):
        self.surface = surface = self.options['surface']

        [nx, ny, _] = surface['mesh'].shape
        name = surface['name']

        self.add_input('sec_forces_pg', val=np.zeros((nx-1, ny-1, 3)), units='N')
        self.add_input('node_forces_pg', val=np.zeros((nx, ny, 3)), units='N')

        self.add_output('sec_forces_w_frame', val=np.zeros((nx-1, ny-1, 3)), units='N')
        self.add_output('node_forces_w_frame', val=np.zeros((nx, ny, 3)), units='N')

        self.add_input('M', val=0.)

        # We'll compute all of sensitivities associated with Mach number through
        # complex-step. Since it's a scalar this is pretty cheap.
        self.declare_partials('*', 'M', method='cs')

    def compute(self, inputs, outputs):
        M = inputs['M']
        betaPG = np.sqrt(1 - M**2)

        outputs['sec_forces_w_frame'] = inputs['sec_forces_pg']
        outputs['sec_forces_w_frame'][:,:,0] /= betaPG**4
        outputs['sec_forces_w_frame'][:,:,1] /= betaPG**3
        outputs['sec_forces_w_frame'][:,:,2] /= betaPG**3

        outputs['node_forces_w_frame'] = inputs['node_forces_pg']
        outputs['node_forces_w_frame'][:,:,0] /= betaPG**4
        outputs['node_forces_w_frame'][:,:,1] /= betaPG**3
        outputs['node_forces_w_frame'][:,:,2] /= betaPG**3

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        M = inputs['M']
        betaPG = np.sqrt(1 - M**2)

        if mode == 'fwd':
            if 'sec_forces_w_frame' in d_outputs and 'sec_forces_pg' in d_inputs:
                d_outputs['sec_forces_w_frame'][:,:,0] += d_inputs['sec_forces_pg'][:,:,0] / betaPG**4
                d_outputs['sec_forces_w_frame'][:,:,1] += d_inputs['sec_forces_pg'][:,:,1] / betaPG**3
                d_outputs['sec_forces_w_frame'][:,:,2] += d_inputs['sec_forces_pg'][:,:,2] / betaPG**3

            if 'node_forces_w_frame' in d_outputs and 'node_forces_pg' in d_inputs:
                d_outputs['node_forces_w_frame'][:,:,0] += d_inputs['node_forces_pg'][:,:,0] / betaPG**4
                d_outputs['node_forces_w_frame'][:,:,1] += d_inputs['node_forces_pg'][:,:,1] / betaPG**3
                d_outputs['node_forces_w_frame'][:,:,2] += d_inputs['node_forces_pg'][:,:,2] / betaPG**3

        if mode == 'rev':
            if 'sec_forces_w_frame' in d_outputs and 'sec_forces_pg' in d_inputs:
                d_inputs['sec_forces_pg'][:,:,0] += d_outputs['sec_forces_w_frame'][:,:,0] / betaPG**4
                d_inputs['sec_forces_pg'][:,:,1] += d_outputs['sec_forces_w_frame'][:,:,1] / betaPG**3
                d_inputs['sec_forces_pg'][:,:,2] += d_outputs['sec_forces_w_frame'][:,:,2] / betaPG**3

            if 'node_forces_w_frame' in d_outputs and 'node_forces_pg' in d_inputs:
                d_inputs['node_forces_pg'][:,:,0] += d_outputs['node_forces_w_frame'][:,:,0] / betaPG**4
                d_inputs['node_forces_pg'][:,:,1] += d_outputs['node_forces_w_frame'][:,:,1] / betaPG**3
                d_inputs['node_forces_pg'][:,:,2] += d_outputs['node_forces_w_frame'][:,:,2] / betaPG**3
