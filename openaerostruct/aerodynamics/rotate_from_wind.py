from __future__ import print_function, division
import numpy as np

from openmdao.api import ExplicitComponent


class RotateFromWindFrame(ExplicitComponent):
    """
    Rotate the aerodynamic sectional and nodal force vectors from the wind to
    the standard aerodynamic frame. This is the reverse operation of the
    RotateToWindFrame component.

    This transformation is given by the following rotation matrix:
         -        -     -                           -     -        -
        | F_x_aero |   | cosb*cosa, sinb*cosa, -sina |   | F_x_wind |
        | F_y_aero | = | -sinb,          cosb,     0 | . | F_y_wind |
        | F_z_aero |   | cosb*sina, sinb*sina,  cosa |   | F_z_wind |
         -        -     -                           -     -        -

    Where "a" is the angle of attack and "b" is the sideslip angle.

    Parameters
    ----------
    sec_forces_w_frame[nx-1, ny-1, 3] : numpy array
        Force vectors on each panel (lattice) in wind frame.
    node_forces_w_frame[nx, ny, 3] : numpy array
        Equivilent force vector on each panel (lattice) node in wind frame.

    alpha : float
        Angle of attack in degrees.
    beta : float
        Sideslip angle in degrees.

    Returns
    -------
    sec_forces[nx-1, ny-1, 3] : numpy array
        Force vectors on each panel (lattice) in aero frame.
    node_forces[nx-1, ny, 3] : numpy array
        Equivilent force vector on each panel (lattice) node in aero frame.
    """

    def initialize(self):
        self.options.declare('surface', types=dict)

    def setup(self):
        self.surface = surface = self.options['surface']

        [nx, ny, _] = surface['mesh'].shape
        name = surface['name']

        self.add_input('sec_forces_w_frame', val=np.zeros((nx-1, ny-1, 3)), units='N')
        self.add_input('node_forces_w_frame', val=np.zeros((nx, ny, 3)), units='N')
        self.add_input('alpha', val=0., units='rad')
        self.add_input('beta', val=0., units='rad')

        self.add_output('sec_forces', val=np.zeros((nx-1, ny-1, 3)), units='N')
        self.add_output('node_forces', val=np.zeros((nx, ny, 3)), units='N')

        # We'll compute all of sensitivities associated with angle of attack and
        # sideslip number through complex-step. Since it's a scalar this is
        # pretty cheap.
        self.declare_partials('*', 'alpha', method='cs')
        self.declare_partials('*', 'beta', method='cs')

    def compute(self, inputs, outputs):

        alpha = inputs['alpha']
        beta = inputs['beta']
        cosa = np.cos(alpha)
        sina = np.sin(alpha)
        cosb = np.cos(beta)
        sinb = np.sin(beta)
        # Define aero->wind rotation matrix
        Tw = np.array([[cosb*cosa, -sinb, cosb*sina],
                       [sinb*cosa,  cosb, sinb*sina],
                       [-sina,         0, cosa     ]], alpha.dtype)
        # wind->aero rotation matrix is given by transpose

        # Use einsum for fast vectorized matrix multiplication
        outputs['sec_forces'] = np.einsum('lk,ijk->ijl', Tw.T, inputs['sec_forces_w_frame'])
        outputs['node_forces'] = np.einsum('lk,ijk->ijl', Tw.T, inputs['node_forces_w_frame'])

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        alpha = inputs['alpha']
        beta = inputs['beta']
        cosa = np.cos(alpha)
        sina = np.sin(alpha)
        cosb = np.cos(beta)
        sinb = np.sin(beta)
        Tw = np.array([[cosb*cosa, -sinb, cosb*sina],
                       [sinb*cosa,  cosb, sinb*sina],
                       [-sina,         0, cosa     ]], alpha.dtype)
        if mode == 'fwd':
            if 'sec_forces' in d_outputs and 'sec_forces_w_frame' in d_inputs:
                d_outputs['sec_forces'] += np.einsum('lk,ijk->ijl', Tw.T, d_inputs['sec_forces_w_frame'])
            if 'node_forces' in d_outputs and 'node_forces_w_frame' in d_inputs:
                d_outputs['node_forces'] += np.einsum('lk,ijk->ijl', Tw.T, d_inputs['node_forces_w_frame'])

        if mode == 'rev':
            if 'sec_forces' in d_outputs and 'sec_forces_w_frame' in d_inputs:
                d_inputs['sec_forces_w_frame'] += np.einsum('lk,ijk->ijl', Tw, d_outputs['sec_forces'])
            if 'node_forces' in d_outputs and 'node_forces_w_frame' in d_inputs:
                d_inputs['node_forces_w_frame'] += np.einsum('lk,ijk->ijl', Tw, d_outputs['node_forces'])
