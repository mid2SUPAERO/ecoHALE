from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent


class ConvertVelocity(ExplicitComponent):
    """
    Convert the freestream velocity magnitude into a velocity vector at each
    evaluation point. In this case, each of the panels sees the same velocity.
    This really just helps us set up the velocities for use in the VLM analysis.

    Parameters
    ----------
    alpha : float
        The angle of attack for the aircraft (all lifting surfaces) in degrees.
    v : float
        The freestream velocity magnitude.

    Returns
    -------
    freestream_velocities[system_size, 3] : numpy array
        The rotated freestream velocities at each evaluation point for all
        lifting surfaces. system_size is the sum of the count of all panels
        for all lifting surfaces.
    """

    def initialize(self):
        self.options.declare('surfaces', types=list)

    def setup(self):
        surfaces = self.options['surfaces']

        system_size = 0

        # Loop through each surface and cumulatively add the number of panels
        # to obtain system_size.
        for surface in surfaces:
            mesh=surface['mesh']
            nx = mesh.shape[0]
            ny = mesh.shape[1]
            system_size += (nx - 1) * (ny - 1)

        self.system_size = system_size

        self.add_input('alpha', val=0., units='deg')
        self.add_input('v', val=1., units='m/s')

        self.add_output('freestream_velocities', shape=(system_size, 3), units='m/s')

        self.declare_partials('freestream_velocities', 'alpha')
        self.declare_partials('freestream_velocities', 'v')

    def compute(self, inputs, outputs):
        # Rotate the freestream velocities based on the angle of attack.
        # Here we assume there is no beta, or sideslip angle.
        alpha = inputs['alpha'][0] * np.pi / 180.
        cosa = np.cos(alpha)
        sina = np.sin(alpha)
        v_inf = inputs['v'][0] * np.array([cosa, 0., sina])
        outputs['freestream_velocities'][:, :] = v_inf

    def compute_partials(self, inputs, J):
        alpha = inputs['alpha'][0] * np.pi / 180.
        cosa = np.cos(alpha)
        sina = np.sin(alpha)

        J['freestream_velocities','v'] = np.tile(np.array([cosa, 0., sina]), self.system_size)
        J['freestream_velocities','alpha'] = np.tile(inputs['v'][0] * np.array([-sina, 0., cosa]) * np.pi/180., self.system_size)
