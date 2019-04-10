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
    beta : float
        The sideslip angle for the aircraft (all lifting surfaces) in degrees.
    v : float
        The freestream velocity magnitude.
    omega[num_surface, 3] : ndarray
        Angular velocity vector for each surface about center of gravity.
        Only available if the rotational options is set to True.
    cg[3] : ndarray
        The x, y, z coordinates of the center of gravity for the entire aircraft.
        Only available if the rotational options is set to True.
    coll_pts[num_eval_points, 3] : ndarray
        The xyz coordinates of the collocation points used in the VLM analysis.
        This array contains points for all lifting surfaces in the problem.
        Only available if the rotational options is set to True.

    Returns
    -------
    freestream_velocities[system_size, 3] : numpy array
        The rotated freestream velocities at each evaluation point for all
        lifting surfaces. system_size is the sum of the count of all panels
        for all lifting surfaces.
    """

    def initialize(self):
        self.options.declare('surfaces', types=list)
        self.options.declare('rotational', False, types=bool,
                             desc="Set to True to turn on support for computing angular velocities")

    def setup(self):
        surfaces = self.options['surfaces']
        rotational = self.options['rotational']

        system_size = 0
        sizes = []

        # Loop through each surface and cumulatively add the number of panels
        # to obtain system_size.
        for surface in surfaces:
            mesh = surface['mesh']
            nx = mesh.shape[0]
            ny = mesh.shape[1]
            size = (nx - 1) * (ny - 1)
            system_size += size
            sizes.append(size)

        self.system_size = system_size

        self.add_input('alpha', val=0., units='deg')
        self.add_input('beta', val=0., units='deg')
        self.add_input('v', val=1., units='m/s')

        if rotational:
            self.add_input('coll_pts', shape=(system_size, 3), units='m')
            self.add_input('omega', val=np.zeros((len(surfaces), 3)), units='rad/s')
            self.add_input('cg', val=np.ones((3, )), units='m')

        self.add_output('freestream_velocities', shape=(system_size, 3), units='m/s')

        self.declare_partials('freestream_velocities', 'alpha')
        self.declare_partials('freestream_velocities', 'beta')
        self.declare_partials('freestream_velocities', 'v')

        if rotational:

            # First Half of cross product
            row = np.array([1, 2, 0])
            col = np.array([2, 0, 1])

            rows1 = np.tile(row, system_size) + np.repeat(3*np.arange(system_size), 3)
            cols1 = np.tile(col, system_size)

            # Second Half of cross product
            rows2 = np.tile(col, system_size) + np.repeat(3*np.arange(system_size), 3)
            cols2 = np.tile(row, system_size)

            rows = np.concatenate([rows1, rows2])
            cols = np.concatenate([cols1, cols2])

            self.declare_partials('freestream_velocities', 'cg', rows=rows, cols=cols)

            cols1 = np.tile(col, system_size) + np.repeat(3*np.arange(system_size), 3)
            cols2 = np.tile(row, system_size) + np.repeat(3*np.arange(system_size), 3)
            cols = np.concatenate([cols1, cols2])

            self.declare_partials('freestream_velocities', 'coll_pts', rows=rows, cols=cols)

            kernel = np.repeat(np.arange(len(sizes), dtype=int), sizes)

            cols1 = np.tile(col, system_size) + np.repeat(3*kernel, 3)
            cols2 = np.tile(row, system_size) + np.repeat(3*kernel, 3)

            cols = np.concatenate([cols1, cols2])

            self.declare_partials('freestream_velocities', 'omega', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        # Rotate the freestream velocities based on the angle of attack and the sideslip angle.
        alpha = inputs['alpha'][0] * np.pi / 180.
        beta = inputs['beta'][0] * np.pi / 180.

        cosa = np.cos(alpha)
        sina = np.sin(alpha)
        cosb = np.cos(beta)
        sinb = np.sin(beta)

        v_inf = inputs['v'][0] * np.array([cosa*cosb, -sinb, sina*cosb])
        outputs['freestream_velocities'][:, :] = v_inf

        # Add angular velocity term
        if self.options['rotational']:
            cg = inputs['cg']
            omega = inputs['omega']
            c_pts = inputs['coll_pts']

            surfaces = self.options['surfaces']
            idx = 0
            for j, surface in enumerate(surfaces):
                mesh = surface['mesh']
                nx = mesh.shape[0]
                ny = mesh.shape[1]
                size = (nx - 1) * (ny - 1)

                r = c_pts[idx:idx+size, :] - cg

                outputs['freestream_velocities'][idx:idx+size, :] += np.cross(omega[j, :], r)

                idx += size

    def compute_partials(self, inputs, J):
        alpha = inputs['alpha'][0] * np.pi / 180.
        beta = inputs['beta'][0] * np.pi / 180.

        cosa = np.cos(alpha)
        sina = np.sin(alpha)
        cosb = np.cos(beta)
        sinb = np.sin(beta)

        J['freestream_velocities','v'] = np.tile(np.array([cosa*cosb, -sinb, sina*cosb]), self.system_size)
        J['freestream_velocities','alpha'] = np.tile(inputs['v'][0] * np.array([-sina*cosb, 0., cosa*cosb]) * np.pi/180.,
                                                     self.system_size)
        J['freestream_velocities','beta'] = np.tile(inputs['v'][0] * np.array([-cosa*sinb, -cosb, -sina*sinb]) * np.pi/180.,
                                                    self.system_size)

        if self.options['rotational']:
            cg = inputs['cg']
            omega = inputs['omega']
            c_pts = inputs['coll_pts']

            surfaces = self.options['surfaces']
            idx = jdx = 0
            ii = self.system_size * 3
            for j, surface in enumerate(surfaces):
                mesh = surface['mesh']
                nx = mesh.shape[0]
                ny = mesh.shape[1]
                size = (nx - 1) * (ny - 1)

                r = c_pts[jdx:jdx+size, :] - cg
                omega_j = omega[j, :]

                # Cross product derivatives organized so we can tile a variable directly into slices

                J['freestream_velocities', 'cg'][idx:idx+size*3] = np.tile(omega_j, size)
                J['freestream_velocities', 'cg'][idx+ii:idx+ii+size*3] = -np.tile(omega_j, size)

                J['freestream_velocities', 'coll_pts'][idx:idx+size*3] = -np.tile(omega_j, size)
                J['freestream_velocities', 'coll_pts'][idx+ii:idx+ii+size*3] = np.tile(omega_j, size)

                J['freestream_velocities', 'omega'][idx:idx+size*3] = r.flatten()
                J['freestream_velocities', 'omega'][idx+ii:idx+ii+size*3] = -r.flatten()

                idx += 3*size
                jdx += size
