from __future__ import division, print_function
import numpy as np

from openmdao.api import ExplicitComponent


class MomentCoefficient(ExplicitComponent):
    """
    Compute the coefficient of moment (CM) for the entire aircraft.

    Parameters
    ----------
    b_pts[nx-1, ny, 3] : numpy array
        Bound points for the horseshoe vortices, found along the 1/4 chord.
    widths[ny-1] : numpy array
        The spanwise widths of each individual panel.
    chords[ny] : numpy array
        The chordwise length of the entire airfoil following the camber line.
    S_ref : float
        The reference area of the lifting surface.
    sec_forces[nx-1, ny-1, 3] : numpy array
        Contains the sectional forces acting on each panel.
        Stored in Fortran order (only relevant with more than one chordwise
        panel).

    cg[3] : numpy array
        The x, y, z coordinates of the center of gravity for the entire aircraft.
    v : float
        Freestream air velocity in m/s.
    rho : float
        Air density in kg/m^3.

    Returns
    -------
    CM[3] : numpy array
        The coefficient of moment around the x-, y-, and z-axes at the cg point.
    """

    def initialize(self):
        self.options.declare('surfaces', types=list)

    def setup(self):
        for surface in self.options['surfaces']:
            name = surface['name']
            ny = surface['num_y']
            nx = surface['num_x']

            self.add_input(name + '_b_pts', val=np.ones((nx-1, ny, 3)), units='m')
            self.add_input(name + '_widths', val=np.ones((ny-1)), units='m')
            self.add_input(name + '_chords', val=np.ones((ny)), units='m')
            self.add_input(name + '_S_ref', val=1., units='m**2')
            self.add_input(name + '_sec_forces', val=np.ones((nx-1, ny-1, 3)), units='N')

        self.add_input('cg', val=np.ones((3)), units='m')
        self.add_input('v', val=10., units='m/s')
        self.add_input('rho', val=3., units='kg/m**3')
        self.add_input('S_ref_total', val=1., units='m**2')

        self.add_output('CM', val=np.ones((3)))

        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        rho = inputs['rho']
        cg = inputs['cg']

        S_ref_tot = 0.
        M = np.zeros((3))

        # Loop through each surface and find its contributions to the moment
        # of the aircraft based on the section forces and their location
        for j, surface in enumerate(self.options['surfaces']):
            name = surface['name']
            nx = surface['num_x']
            ny = surface['num_y']

            b_pts = inputs[name + '_b_pts']
            widths = inputs[name + '_widths']
            chords = inputs[name + '_chords']
            S_ref = inputs[name + '_S_ref']
            sec_forces = inputs[name + '_sec_forces']

            # Compute the average chord for each panel and then the
            # mean aerodynamic chord (MAC) based on these chords and the
            # computed area
            panel_chords = (chords[1:] + chords[:-1]) / 2.
            MAC = 1. / S_ref * np.sum(panel_chords**2 * widths)

            # If the surface is symmetric, then the previously computed MAC
            # is half what it should be
            if surface['symmetry']:
                MAC *= 2

            # Get the moment arm acting on each panel, relative to the cg
            pts = (inputs[name + '_b_pts'][:, 1:, :] + \
                inputs[name + '_b_pts'][:, :-1, :]) / 2
            diff = (pts - cg) / MAC

            # Compute the moment based on the previously computed moment
            # arm and the section forces
            moment = np.zeros((ny - 1, 3))
            for ind in range(nx-1):
                moment = moment + np.cross(diff[ind, :, :], sec_forces[ind, :, :], axis=1)

            # If the surface is symmetric, set the x- and z-direction moments
            # to 0 and double the y-direction moment
            if surface['symmetry']:
                moment[:, 0] = 0.
                moment[:, 1] *= 2
                moment[:, 2] = 0.
            M = M + np.sum(moment, axis=0)

            # For the first (main) lifting surface, we save the MAC to correctly
            # normalize CM
            if j == 0:
                self.MAC_wing = MAC
            S_ref_tot = S_ref_tot + S_ref

        self.M = M

        # Compute the normalized CM
        outputs['CM'] = M / (0.5 * rho * inputs['v']**2 * inputs['S_ref_total'] * self.MAC_wing)
