from __future__ import print_function, division
import numpy as np
from numpy import cos, sin, tan


def rotate(mesh, theta_y, symmetry, rotate_x=True):
    """
    Compute rotation matrices given mesh and rotation angles in degrees.

    Parameters
    ----------
    mesh[nx, ny, 3] : numpy array
        Nodal mesh defining the initial aerodynamic surface.
    theta_y[ny] : numpy array
        1-D array of rotation angles about y-axis for each wing slice in degrees.
    symmetry : boolean
        Flag set to True if surface is reflected about y=0 plane.
    rotate_x : boolean
        Flag set to True if the user desires the twist variable to always be
        applied perpendicular to the wing (say, in the case of a winglet).

    Returns
    -------
    mesh[nx, ny, 3] : numpy array
        Nodal mesh defining the twisted aerodynamic surface.

    """
    te = mesh[-1]
    le = mesh[ 0]
    quarter_chord = 0.25 * te + 0.75 * le

    ny = mesh.shape[1]
    nx = mesh.shape[0]

    if rotate_x:
        # Compute spanwise z displacements along quarter chord
        if symmetry:
            dz_qc = quarter_chord[:-1,2] - quarter_chord[1:,2]
            dy_qc = quarter_chord[:-1,1] - quarter_chord[1:,1]
            theta_x = np.arctan(dz_qc/dy_qc)

            # Prepend with 0 so that root is not rotated
            rad_theta_x = np.append(theta_x, 0.0)
        else:
            root_index = int((ny - 1) / 2)
            dz_qc_left = quarter_chord[:root_index,2] - quarter_chord[1:root_index+1,2]
            dy_qc_left = quarter_chord[:root_index,1] - quarter_chord[1:root_index+1,1]
            theta_x_left = np.arctan(dz_qc_left/dy_qc_left)
            dz_qc_right = quarter_chord[root_index+1:,2] - quarter_chord[root_index:-1,2]
            dy_qc_right = quarter_chord[root_index+1:,1] - quarter_chord[root_index:-1,1]
            theta_x_right = np.arctan(dz_qc_right/dy_qc_right)

            # Concatenate thetas
            rad_theta_x = np.concatenate((theta_x_left, np.zeros(1), theta_x_right))

    else:
        rad_theta_x = 0.0

    rad_theta_y = theta_y * np.pi / 180.

    mats = np.zeros((ny, 3, 3), dtype="complex")
    mats[:, 0, 0] = cos(rad_theta_y)
    mats[:, 0, 2] = sin(rad_theta_y)
    mats[:, 1, 0] = sin(rad_theta_x)*sin(rad_theta_y)
    mats[:, 1, 1] = cos(rad_theta_x)
    mats[:, 1, 2] = -sin(rad_theta_x)*cos(rad_theta_y)
    mats[:, 2, 0] = -cos(rad_theta_x)*sin(rad_theta_y)
    mats[:, 2, 1] = sin(rad_theta_x)
    mats[:, 2, 2] = cos(rad_theta_x)*cos(rad_theta_y)
    for ix in range(nx):
        row = mesh[ix]
        row[:] = np.einsum("ikj, ij -> ik", mats, row - quarter_chord)
        row += quarter_chord

def scale_x(mesh, chord_dist):
    """
    Modify the chords along the span of the wing by scaling only the x-coord.

    Parameters
    ----------
    mesh[nx, ny, 3] : numpy array
        Nodal mesh defining the initial aerodynamic surface.
    chord_dist[ny] : numpy array
        Chord length for each panel edge.

    Returns
    -------
    mesh[nx, ny, 3] : numpy array
        Nodal mesh with the new chord lengths.
    """
    te = mesh[-1]
    le = mesh[ 0]
    quarter_chord = 0.25 * te + 0.75 * le

    ny = mesh.shape[1]

    for i in range(ny):
        mesh[:, i, 0] = (mesh[:, i, 0] - quarter_chord[i, 0]) * chord_dist[i] + \
            quarter_chord[i, 0]

def shear_x(mesh, xshear):
    """
    Shear the wing in the x direction (distributed sweep).

    Parameters
    ----------
    mesh[nx, ny, 3] : numpy array
        Nodal mesh defining the initial aerodynamic surface.
    xshear[ny] : numpy array
        Distance to translate wing in x direction.

    Returns
    -------
    mesh[nx, ny, 3] : numpy array
        Nodal mesh with the new chord lengths.
    """
    mesh[:, :, 0] += xshear

def shear_z(mesh, zshear):
    """
    Shear the wing in the z direction (distributed dihedral).

    Parameters
    ----------
    mesh[nx, ny, 3] : numpy array
        Nodal mesh defining the initial aerodynamic surface.
    zshear[ny] : numpy array
        Distance to translate wing in z direction.

    Returns
    -------
    mesh[nx, ny, 3] : numpy array
        Nodal mesh with the new chord lengths.
    """
    mesh[:, :, 2] += zshear

def sweep(mesh, sweep_angle, symmetry):
    """
    Apply shearing sweep. Positive sweeps back.

    Parameters
    ----------
    mesh[nx, ny, 3] : numpy array
        Nodal mesh defining the initial aerodynamic surface.
    sweep_angle : float
        Shearing sweep angle in degrees.
    symmetry : boolean
        Flag set to true if surface is reflected about y=0 plane.

    Returns
    -------
    mesh[nx, ny, 3] : numpy array
        Nodal mesh defining the swept aerodynamic surface.

    """

    # Get the mesh parameters and desired sweep angle
    num_x, num_y, _ = mesh.shape
    le = mesh[0]
    p180 = np.pi / 180
    tan_theta = tan(p180*sweep_angle)

    # If symmetric, simply vary the x-coord based on the distance from the
    # center of the wing
    if symmetry:
        y0 = le[-1, 1]
        dx = -(le[:, 1] - y0) * tan_theta

    # Else, vary the x-coord on either side of the wing
    else:
        ny2 = (num_y - 1) // 2
        y0 = le[ny2, 1]

        dx_right = (le[ny2:, 1] - y0) * tan_theta
        dx_left = -(le[:ny2, 1] - y0) * tan_theta
        dx = np.hstack((dx_left, dx_right))

    for i in range(num_x):
        mesh[i, :, 0] += dx

def dihedral(mesh, dihedral_angle, symmetry):
    """
    Apply dihedral angle. Positive angles up.

    Parameters
    ----------
    mesh[nx, ny, 3] : numpy array
        Nodal mesh defining the initial aerodynamic surface.
    dihedral_angle : float
        Dihedral angle in degrees.
    symmetry : boolean
        Flag set to true if surface is reflected about y=0 plane.

    Returns
    -------
    mesh[nx, ny, 3] : numpy array
        Nodal mesh defining the aerodynamic surface with dihedral angle.

    """

    # Get the mesh parameters and desired sweep angle
    num_x, num_y, _ = mesh.shape
    le = mesh[0]
    p180 = np.pi / 180
    tan_theta = tan(p180*dihedral_angle)

    # If symmetric, simply vary the z-coord based on the distance from the
    # center of the wing
    if symmetry:
        y0 = le[-1, 1]
        dz = -(le[:, 1] - y0) * tan_theta

    else:
        ny2 = (num_y-1) // 2
        y0 = le[ny2, 1]
        dz_right = (le[ny2:, 1] - y0) * tan_theta
        dz_left = -(le[:ny2, 1] - y0) * tan_theta
        dz = np.hstack((dz_left, dz_right))

    for i in range(num_x):
        mesh[i, :, 2] += dz


def stretch(mesh, span, symmetry):
    """
    Stretch mesh in spanwise direction to reach specified span.

    Parameters
    ----------
    mesh[nx, ny, 3] : numpy array
        Nodal mesh defining the initial aerodynamic surface.
    span : float
        Relative stetch ratio in the spanwise direction.
    symmetry : boolean
        Flag set to true if surface is reflected about y=0 plane.

    Returns
    -------
    mesh[nx, ny, 3] : numpy array
        Nodal mesh defining the stretched aerodynamic surface.

    """

    # Set the span along the quarter-chord line
    le = mesh[0]
    te = mesh[-1]
    quarter_chord = 0.25 * te + 0.75 * le

    # The user always deals with the full span, so if they input a specific
    # span value and have symmetry enabled, we divide this value by 2.
    if symmetry:
        span /= 2.

    # Compute the previous span and determine the scalar needed to reach the
    # desired span
    prev_span = quarter_chord[-1, 1] - quarter_chord[0, 1]
    s = quarter_chord[:,1] / prev_span
    mesh[:, :, 1] = s * span

def taper(mesh, taper_ratio, symmetry):
    """
    Alter the spanwise chord linearly to produce a tapered wing. Note that
    we apply taper around the quarter-chord line.

    Parameters
    ----------
    mesh[nx, ny, 3] : numpy array
        Nodal mesh defining the initial aerodynamic surface.
    taper_ratio : float
        Taper ratio for the wing; 1 is untapered, 0 goes to a point.
    symmetry : boolean
        Flag set to true if surface is reflected about y=0 plane.

    Returns
    -------
    mesh[nx, ny, 3] : numpy array
        Nodal mesh defining the tapered aerodynamic surface.

    """

    # Get mesh parameters and the quarter-chord
    le = mesh[0]
    te = mesh[-1]
    num_x, num_y, _ = mesh.shape
    quarter_chord = 0.25 * te + 0.75 * le

    # If symmetric, solve for the correct taper ratio, which is a linear
    # interpolation problem
    if symmetry:
        x = quarter_chord[:, 1]
        span = x[-1] - x[0]
        xp = np.array([-span, 0.])
        fp = np.array([taper_ratio, 1.])
        taper = np.interp(x.real, xp.real, fp.real)

        # Modify the mesh based on the taper amount computed per spanwise section
        for i in range(num_x):
            for ind in range(3):
                mesh[i, :, ind] = (mesh[i, :, ind] - quarter_chord[:, ind]) * \
                    taper + quarter_chord[:, ind]

    # Otherwise, we set up an interpolation problem for the entire wing, which
    # consists of two linear segments
    else:
        x = quarter_chord[:, 1]
        span = x[-1] - x[0]
        xp = np.array([-span/2, 0., span/2])
        fp = np.array([taper_ratio, 1., taper_ratio])
        taper = np.interp(x.real, xp.real, fp.real)

        for i in range(num_x):
            for ind in range(3):
                mesh[i, :, ind] = (mesh[i, :, ind] - quarter_chord[:, ind]) * \
                    taper + quarter_chord[:, ind]
