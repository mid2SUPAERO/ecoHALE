""" Components that manipulate geometry mesh based on high-level design parameters. """

from __future__ import division, print_function

import numpy as np

from openmdao.api import ExplicitComponent


class Taper(ExplicitComponent):
    """
    OpenMDAO component that manipulates the mesh by altering the spanwise chord linearly to produce
    a tapered wing. Note that we apply taper around the quarter-chord line.

    Parameters
    ----------
    taper : float
        Taper ratio for the wing; 1 is untapered, 0 goes to a point at the tip.

    Returns
    -------
    mesh[nx, ny, 3] : numpy array
        Nodal mesh defining the tapered aerodynamic surface..
    """

    def initialize(self):
        """
        Declare options.
        """
        self.options.declare('val',
                             desc='Initial value for the taper ratio.')
        self.options.declare('mesh',
                             desc='Nodal mesh defining the initial aerodynamic surface.')
        self.options.declare('symmetry', default=False,
                             desc='Flag set to true if surface is reflected about y=0 plane.')

    def setup(self):
        mesh = self.options['mesh']
        val = self.options['val']

        self.add_input('taper', val=val)

        self.add_output('mesh', val=mesh, units='m')

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        mesh = self.options['mesh']
        symmetry = self.options['symmetry']
        taper_ratio = inputs['taper'][0]

        # Get mesh parameters and the quarter-chord
        le = mesh[0]
        te = mesh[-1]
        num_x, num_y, _ = mesh.shape
        quarter_chord = 0.25 * te + 0.75 * le
        x = quarter_chord[:, 1]
        span = x[-1] - x[0]

        # If symmetric, solve for the correct taper ratio, which is a linear
        # interpolation problem
        if symmetry:
            xp = np.array([-span, 0.])
            fp = np.array([taper_ratio, 1.])

        # Otherwise, we set up an interpolation problem for the entire wing, which
        # consists of two linear segments
        else:
            xp = np.array([-span/2, 0., span/2])
            fp = np.array([taper_ratio, 1., taper_ratio])

        taper = np.interp(x.real, xp.real, fp.real)

        # Modify the mesh based on the taper amount computed per spanwise section
        outputs['mesh'] = np.einsum('ijk,j->ijk', mesh - quarter_chord, taper) + quarter_chord

    def compute_partials(self, inputs, partials):
        mesh = self.options['mesh']
        symmetry = self.options['symmetry']
        taper_ratio = inputs['taper'][0]

        # Get mesh parameters and the quarter-chord
        le = mesh[0]
        te = mesh[-1]
        num_x, num_y, _ = mesh.shape
        quarter_chord = 0.25 * te + 0.75 * le
        x = quarter_chord[:, 1]
        span = x[-1] - x[0]

        # If symmetric, solve for the correct taper ratio, which is a linear
        # interpolation problem
        if symmetry:
            xp = np.array([-span, 0.])
            fp = np.array([taper_ratio, 1.])

        # Otherwise, we set up an interpolation problem for the entire wing, which
        # consists of two linear segments
        else:
            xp = np.array([-span/2, 0., span/2])
            fp = np.array([taper_ratio, 1., taper_ratio])

        taper = np.interp(x, xp, fp)
        dtaper = (1.0 - taper) / (1.0 - taper_ratio)

        # Modify the mesh based on the taper amount computed per spanwise section
        partials['mesh', 'taper'] = np.einsum('ijk, j->ijk', mesh - quarter_chord, dtaper)


class ScaleX(ExplicitComponent):
    """
    OpenMDAO component that manipulates the mesh by modifying the chords along the span of the
    wing by scaling only the x-coord.

    Parameters
    ----------
    mesh[nx, ny, 3] : numpy array
        Nodal mesh defining the initial aerodynamic surface.
    chord[ny] : numpy array
        Chord length for each panel edge.

    Returns
    -------
    mesh[nx, ny, 3] : numpy array
        Nodal mesh with the new chord lengths.
    """

    def initialize(self):
        """
        Declare options.
        """
        self.options.declare('val', desc='Initial value for chord lengths')
        self.options.declare('mesh_shape', desc='Tuple containing mesh shape (nx, ny).')

    def setup(self):
        mesh_shape = self.options['mesh_shape']
        val = self.options['val']

        self.add_input('chord', units='m', val=val)
        self.add_input('in_mesh', shape=mesh_shape, units='m')

        self.add_output('mesh', shape=mesh_shape, units='m')

        nx, ny, _ = mesh_shape
        nn = nx * ny * 3

        rows = np.arange(nn)
        col = np.tile(np.zeros(3), ny) + np.repeat(np.arange(ny), 3)
        cols = np.tile(col, nx)

        self.declare_partials('mesh', 'chord', rows=rows, cols=cols)

        p_rows = np.arange(nn)
        p_cols = np.arange(nn)
        te_rows = np.arange(((nx-1) * ny * 3))
        le_rows = te_rows + ny*3
        le_cols = np.tile(np.arange(3 * ny), nx-1)
        te_cols = le_cols + ny*3*(nx-1)
        rows = np.concatenate([p_rows, te_rows, le_rows])
        cols = np.concatenate([p_rows, te_cols, le_cols])

        self.declare_partials('mesh', 'in_mesh', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        mesh = inputs['in_mesh']
        chord_dist = inputs['chord']

        te = mesh[-1]
        le = mesh[ 0]
        quarter_chord = 0.25 * te + 0.75 * le

        outputs['mesh'] = np.einsum('ijk,j->ijk', mesh - quarter_chord, chord_dist) + quarter_chord

    def compute_partials(self, inputs, partials):
        mesh = inputs['in_mesh']
        chord_dist = inputs['chord']

        te = mesh[-1]
        le = mesh[ 0]
        quarter_chord = 0.25 * te + 0.75 * le

        partials['mesh', 'chord'] = (mesh - quarter_chord).flatten()

        nx, ny, _ = mesh.shape
        nn = nx * ny * 3
        d_mesh = np.einsum('i,ij->ij', chord_dist, np.ones((ny, 3))).flatten()
        partials['mesh', 'in_mesh'][:nn] = np.tile(d_mesh, nx)

        d_qc = (np.einsum('ij,i->ij', np.ones((ny, 3)), 1.0 - chord_dist)).flatten()
        nnq = (nx-1) * ny * 3
        partials['mesh', 'in_mesh'][nn:nn + nnq] = np.tile(0.25 * d_qc, nx-1)
        partials['mesh', 'in_mesh'][nn + nnq:] = np.tile(0.75 * d_qc, nx-1)

        nnq = ny*3
        partials['mesh', 'in_mesh'][nn - nnq:nn] += 0.25 * d_qc
        partials['mesh', 'in_mesh'][:nnq] += 0.75 * d_qc


class Sweep(ExplicitComponent):
    """
    OpenMDAO component that manipulates the mesh applying shearing sweep. Positive sweeps back.

    Parameters
    ----------
    mesh[nx, ny, 3] : numpy array
        Nodal mesh defining the initial aerodynamic surface.
    sweep : float
        Shearing sweep angle in degrees.
    symmetry : boolean
        Flag set to true if surface is reflected about y=0 plane.

    Returns
    -------
    mesh[nx, ny, 3] : numpy array
        Nodal mesh defining the swept aerodynamic surface.
    """

    def initialize(self):
        """
        Declare options.
        """
        self.options.declare('val',
                             desc='Initial value for x shear.')
        self.options.declare('mesh_shape', desc='Tuple containing mesh shape (nx, ny).')
        self.options.declare('symmetry', default=False,
                             desc='Flag set to true if surface is reflected about y=0 plane.')

    def setup(self):
        mesh_shape = self.options['mesh_shape']
        val = self.options['val']

        self.add_input('sweep', val=val, units='deg')
        self.add_input('in_mesh', shape=mesh_shape, units='m')

        self.add_output('mesh', shape=mesh_shape, units='m')

        nx, ny, _ = mesh_shape
        nn = nx * ny
        rows = 3 * np.arange(nn)
        cols = np.zeros(nn)

        self.declare_partials('mesh', 'sweep', rows=rows, cols=cols)

        nn = nx * ny * 3
        n_rows = np.arange(nn)
        n_cols = np.arange(nn)

        if self.options['symmetry']:
            y_cp = ny*3 - 2
            te_cols = np.tile(y_cp, nx * (ny-1))
            te_rows = np.tile(3 * np.arange(ny-1), nx) + np.repeat(3*ny*np.arange(nx), ny-1)
            se_cols = np.tile(3 * np.arange(ny-1) + 1, nx)
        else:
            y_cp = 3*(ny+1) // 2 - 2
            n_sym = (ny-1) // 2

            te_row = np.tile(3*np.arange(n_sym), 2) + np.repeat([0, 3*(n_sym+1)], n_sym)
            te_rows = np.tile(te_row, nx) + np.repeat(3*ny*np.arange(nx), ny-1)

            te_col = np.tile(y_cp, n_sym)
            se_col1 = 3*np.arange(n_sym) + 1
            se_col2 = 3*np.arange(n_sym) + 4 + 3*n_sym

            # neat trick: swap columns on reflected side so we can assign in just two operations
            te_cols = np.tile(np.concatenate([te_col, se_col2]), nx)
            se_cols = np.tile(np.concatenate([se_col1, te_col]), nx)

        rows = np.concatenate(([n_rows, te_rows, te_rows]))
        cols = np.concatenate(([n_rows, te_cols, se_cols]))

        self.declare_partials('mesh', 'in_mesh', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        symmetry = self.options['symmetry']
        sweep_angle = inputs['sweep'][0]
        mesh = inputs['in_mesh']

        # Get the mesh parameters and desired sweep angle
        nx, ny, _ = mesh.shape
        le = mesh[0]
        p180 = np.pi / 180
        tan_theta = np.tan(p180*sweep_angle)

        # If symmetric, simply vary the x-coord based on the distance from the
        # center of the wing
        if symmetry:
            y0 = le[-1, 1]
            dx = -(le[:, 1] - y0) * tan_theta

        # Else, vary the x-coord on either side of the wing
        else:
            ny2 = (ny - 1) // 2
            y0 = le[ny2, 1]

            dx_right = (le[ny2:, 1] - y0) * tan_theta
            dx_left = -(le[:ny2, 1] - y0) * tan_theta
            dx = np.hstack((dx_left, dx_right))

        # dx added spanwise.
        outputs['mesh'][:] = mesh
        outputs['mesh'][:, :, 0] += dx

    def compute_partials(self, inputs, partials):
        symmetry = self.options['symmetry']
        sweep_angle = inputs['sweep'][0]
        mesh = inputs['in_mesh']

        # Get the mesh parameters and desired sweep angle
        nx, ny, _ = mesh.shape
        le = mesh[0]
        p180 = np.pi / 180
        tan_theta = np.tan(p180*sweep_angle)
        dtan_dtheta = p180 / np.cos(p180*sweep_angle)**2

        # If symmetric, simply vary the x-coord based on the distance from the
        # center of the wing
        if symmetry:
            y0 = le[-1, 1]

            dx_dtheta = -(le[:, 1] - y0)

        # Else, vary the x-coord on either side of the wing
        else:
            ny2 = (ny - 1) // 2
            y0 = le[ny2, 1]

            dx_dtheta_right = (le[ny2:, 1] - y0)
            dx_dtheta_left = -(le[:ny2, 1] - y0)
            dx_dtheta = np.hstack((dx_dtheta_left, dx_dtheta_right))

        partials['mesh', 'sweep'] = np.tile(dx_dtheta * dtan_dtheta, nx)

        nn = nx * ny * 3
        partials['mesh', 'in_mesh'][:nn] = 1.0

        nn2 = nx * (ny-1)
        partials['mesh', 'in_mesh'][nn:nn + nn2] = tan_theta
        partials['mesh', 'in_mesh'][nn + nn2:] = -tan_theta


class ShearX(ExplicitComponent):
    """
    OpenMDAO component that manipulates the mesh by shearing the wing in the x direction
    (distributed sweep).

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

    def initialize(self):
        """
        Declare options.
        """
        self.options.declare('val', desc='Initial value for x shear.')
        self.options.declare('mesh_shape', desc='Tuple containing mesh shape (nx, ny).')

    def setup(self):
        mesh_shape = self.options['mesh_shape']
        val = self.options['val']

        self.add_input('xshear', val=val, units='m')
        self.add_input('in_mesh', shape=mesh_shape, units='m')

        self.add_output('mesh', shape=mesh_shape, units='m')

        nx, ny, _ = mesh_shape

        nn = nx * ny
        rows = 3.0*np.arange(nn)
        cols = np.tile(np.arange(ny), nx)
        val = np.ones(nn)

        self.declare_partials('mesh', 'xshear', rows=rows, cols=cols, val=val)

        nn = nx * ny * 3
        rows = np.arange(nn)
        cols = np.arange(nn)
        val = np.ones(nn)

        self.declare_partials('mesh', 'in_mesh', rows=rows, cols=cols, val=val)

    def compute(self, inputs, outputs):
        outputs['mesh'][:] = inputs['in_mesh']
        outputs['mesh'][:, :, 0] += inputs['xshear']


class Stretch(ExplicitComponent):
    """
    OpenMDAO component that manipulates the mesh by stretching the mesh in spanwise direction to
    reach specified span

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

    def initialize(self):
        """
        Declare options.
        """
        self.options.declare('val', desc='Initial value for span.')
        self.options.declare('mesh_shape', desc='Tuple containing mesh shape (nx, ny).')
        self.options.declare('symmetry', default=False,
                             desc='Flag set to true if surface is reflected about y=0 plane.')

    def setup(self):
        mesh_shape = self.options['mesh_shape']
        val = self.options['val']

        self.add_input('span', val=val, units='m')
        self.add_input('in_mesh', shape=mesh_shape, units='m')

        self.add_output('mesh', shape=mesh_shape, units='m')

        nx, ny, _ = mesh_shape
        nn = nx * ny
        rows = 3 * np.arange(nn) + 1
        cols = np.zeros(nn)

        self.declare_partials('mesh', 'span', rows=rows, cols=cols)

        # First: x and z on diag is identity.
        nn = nx * ny
        xz_diag = 3*np.arange(nn)

        # Four columns at le (root, tip) and te (root, tip)
        i_le0 = 1
        i_le1 = ny*3 - 2
        i_te0 = (nx-1)*ny*3 + 1
        i_te1 = nn*3 - 2

        rows_4c = np.tile(3*np.arange(nn) + 1, 4)
        cols_4c = np.concatenate([np.tile(i_le0, nn),
                                  np.tile(i_le1, nn),
                                  np.tile(i_te0, nn),
                                  np.tile(i_te1, nn)
                                  ])

        # Diagonal stripes
        base = 3*np.arange(1, ny-1) + 1
        row_dg = np.tile(base, nx) + np.repeat(ny*3*np.arange(nx), ny-2)
        rows_dg = np.tile(row_dg, 2)
        col_dg = np.tile(base, nx)
        cols_dg = np.concatenate([col_dg, col_dg + 3*ny*(nx-1)])

        rows = np.concatenate([xz_diag, xz_diag + 2, rows_4c, rows_dg])
        cols = np.concatenate([xz_diag, xz_diag + 2, cols_4c, cols_dg])

        self.declare_partials('mesh', 'in_mesh', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        symmetry = self.options['symmetry']
        span = inputs['span'][0]
        mesh = inputs['in_mesh']

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
        s = quarter_chord[:, 1] / prev_span

        outputs['mesh'][:] = mesh
        outputs['mesh'][:, :, 1] = s * span

    def compute_partials(self, inputs, partials):
        symmetry = self.options['symmetry']
        span = inputs['span'][0]
        mesh = inputs['in_mesh']
        nx, ny, _ = mesh.shape

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
        s = quarter_chord[:, 1] / prev_span

        d_prev_span = -quarter_chord[:, 1] / prev_span**2
        d_prev_span_qc0 = np.zeros((ny, ))
        d_prev_span_qc1 = np.zeros((ny, ))
        d_prev_span_qc0[0] = d_prev_span_qc1[-1] = 1.0 / prev_span

        if symmetry:
            partials['mesh', 'span'] = np.tile(0.5 * s, nx)
        else:
            partials['mesh', 'span'] = np.tile(s, nx)

        nn = nx * ny * 2
        partials['mesh', 'in_mesh'][:nn] = 1.0

        nn2 = nx * ny
        partials['mesh', 'in_mesh'][nn:nn + nn2] = np.tile(-0.75 * span * (d_prev_span - d_prev_span_qc0), nx)
        nn3 = nn + nn2 * 2
        partials['mesh', 'in_mesh'][nn + nn2:nn3] = np.tile(0.75 * span * (d_prev_span + d_prev_span_qc1), nx)
        nn4 = nn3 + nn2
        partials['mesh', 'in_mesh'][nn3:nn4] = np.tile(-0.25 * span * (d_prev_span - d_prev_span_qc0), nx)
        nn5 = nn4 + nn2
        partials['mesh', 'in_mesh'][nn4:nn5] = np.tile(0.25 * span * (d_prev_span + d_prev_span_qc1), nx)

        nn6 = nn5 + nx*(ny-2)
        partials['mesh', 'in_mesh'][nn5:nn6] = 0.75 * span / prev_span
        partials['mesh', 'in_mesh'][nn6:] = 0.25 * span / prev_span


class ShearY(ExplicitComponent):
    """
    OpenMDAO component that manipulates the mesh by shearing the wing in the y direction
    (distributed sweep).

    Parameters
    ----------
    mesh[nx, ny, 3] : numpy array
        Nodal mesh defining the initial aerodynamic surface.
    yshear[ny] : numpy array
        Distance to translate wing in y direction.

    Returns
    -------
    mesh[nx, ny, 3] : numpy array
        Nodal mesh with the new chord lengths.
    """

    def initialize(self):
        """
        Declare options.
        """
        self.options.declare('val', desc='Initial value for y shear.')
        self.options.declare('mesh_shape', desc='Tuple containing mesh shape (nx, ny).')

    def setup(self):
        mesh_shape = self.options['mesh_shape']
        val = self.options['val']

        self.add_input('yshear', val=val, units='m')
        self.add_input('in_mesh', shape=mesh_shape, units='m')

        self.add_output('mesh', shape=mesh_shape, units='m')

        nx, ny, _ = mesh_shape

        nn = nx * ny
        rows = 3.0*np.arange(nn) + 1
        cols = np.tile(np.arange(ny), nx)
        val = np.ones(nn)

        self.declare_partials('mesh', 'yshear', rows=rows, cols=cols, val=val)

        nn = nx * ny * 3
        rows = np.arange(nn)
        cols = np.arange(nn)
        val = np.ones(nn)

        self.declare_partials('mesh', 'in_mesh', rows=rows, cols=cols, val=val)

    def compute(self, inputs, outputs):
        outputs['mesh'][:] = inputs['in_mesh']
        outputs['mesh'][:, :, 1] += inputs['yshear']


class Dihedral(ExplicitComponent):
    """
    OpenMDAO component that manipulates the mesh by applying dihedral angle. Positive angles up.

    Parameters
    ----------
    mesh[nx, ny, 3] : numpy array
        Nodal mesh defining the initial aerodynamic surface.
    dihedral : float
        Dihedral angle in degrees.
    symmetry : boolean
        Flag set to true if surface is reflected about y=0 plane.

    Returns
    -------
    mesh[nx, ny, 3] : numpy array
        Nodal mesh defining the aerodynamic surface with dihedral angle.
    """

    def initialize(self):
        """
        Declare options.
        """
        self.options.declare('val', desc='Initial value for dihedral.')
        self.options.declare('mesh_shape', desc='Tuple containing mesh shape (nx, ny).')
        self.options.declare('symmetry', default=False,
                             desc='Flag set to true if surface is reflected about y=0 plane.')

    def setup(self):
        mesh_shape = self.options['mesh_shape']
        val = self.options['val']

        self.add_input('dihedral', val=val, units='deg')
        self.add_input('in_mesh', shape=mesh_shape, units='m')

        self.add_output('mesh', shape=mesh_shape, units='m')

        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        symmetry = self.options['symmetry']
        dihedral_angle = inputs['dihedral'][0]
        mesh = inputs['in_mesh']

        # Get the mesh parameters and desired sweep angle
        _, num_y, _ = inputs['in_mesh'].shape
        le = mesh[0]
        p180 = np.pi / 180
        tan_theta = np.tan(p180 * dihedral_angle)

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

        # dz added spanwise.
        outputs['mesh'][:] = mesh
        outputs['mesh'][:, :, 2] += dz


class ShearZ(ExplicitComponent):
    """
    OpenMDAO component that manipulates the mesh by shearing the wing in the z direction
    (distributed sweep).

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

    def initialize(self):
        """
        Declare options.
        """
        self.options.declare('val', desc='Initial value for z shear.')
        self.options.declare('mesh_shape', desc='Tuple containing mesh shape (nx, ny).')

    def setup(self):
        mesh_shape = self.options['mesh_shape']
        val = self.options['val']

        self.add_input('zshear', val=val, units='m')
        self.add_input('in_mesh', shape=mesh_shape, units='m')

        self.add_output('mesh', shape=mesh_shape, units='m')

        nx, ny, _ = mesh_shape

        nn = nx * ny
        rows = 3.0*np.arange(nn) + 2
        cols = np.tile(np.arange(ny), nx)
        val = np.ones(nn)

        self.declare_partials('mesh', 'zshear', rows=rows, cols=cols, val=val)

        nn = nx * ny * 3
        rows = np.arange(nn)
        cols = np.arange(nn)
        val = np.ones(nn)

        self.declare_partials('mesh', 'in_mesh', rows=rows, cols=cols, val=val)

    def compute(self, inputs, outputs):
        outputs['mesh'][:] = inputs['in_mesh']
        outputs['mesh'][:, :, 2] += inputs['zshear']


class Rotate(ExplicitComponent):
    """
    OpenMDAO component that manipulates the mesh by compute rotation matrices given mesh and
    rotation angles in degrees.

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

    def initialize(self):
        """
        Declare options.
        """
        self.options.declare('val', desc='Initial value for dihedral.')
        self.options.declare('mesh_shape', desc='Tuple containing mesh shape (nx, ny).')
        self.options.declare('symmetry', default=False,
                             desc='Flag set to true if surface is reflected about y=0 plane.')
        self.options.declare('rotate_x', default=True,
                             desc='Flag set to True if the user desires the twist variable to '
                             'always be applied perpendicular to the wing (say, in the case of '
                             'a winglet).')

    def setup(self):
        mesh_shape = self.options['mesh_shape']
        val = self.options['val']

        self.add_input('twist', val=val, units='deg')
        self.add_input('in_mesh', shape=mesh_shape, units='m')

        self.add_output('mesh', shape=mesh_shape, units='m')

        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        symmetry = self.options['symmetry']
        rotate_x = self.options['rotate_x']
        theta_y = inputs['twist']
        mesh = inputs['in_mesh']

        te = mesh[-1]
        le = mesh[ 0]
        quarter_chord = 0.25 * te + 0.75 * le

        _, ny, _ = mesh.shape

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

        mats = np.zeros((ny, 3, 3), dtype=type(rad_theta_y[0]))

        cos_rtx = np.cos(rad_theta_x)
        cos_rty = np.cos(rad_theta_y)
        sin_rtx = np.sin(rad_theta_x)
        sin_rty = np.sin(rad_theta_y)

        mats[:, 0, 0] = cos_rty
        mats[:, 0, 2] = sin_rty
        mats[:, 1, 0] = sin_rtx * sin_rty
        mats[:, 1, 1] = cos_rtx
        mats[:, 1, 2] = -sin_rtx * cos_rty
        mats[:, 2, 0] = -cos_rtx * sin_rty
        mats[:, 2, 1] = sin_rtx
        mats[:, 2, 2] = cos_rtx*cos_rty

        outputs['mesh'] = np.einsum("ikj, mij -> mik", mats, mesh - quarter_chord) + quarter_chord