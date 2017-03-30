""" Manipulate geometry mesh based on high-level design parameters. """

from __future__ import division, print_function
import numpy
from numpy import cos, sin, tan

from openmdao.api import Component
from b_spline import get_bspline_mtx

try:
    import OAS_API
    fortran_flag = True
    data_type = float
except:
    fortran_flag = False
    data_type = complex

def view_mat(mat):
    """ Helper function used to visually examine matrices. """
    import matplotlib.pyplot as plt
    if len(mat.shape) > 2:
        mat = numpy.sum(mat, axis=2)
    im = plt.imshow(mat.real, interpolation='none')
    plt.colorbar(im, orientation='horizontal')
    plt.show()

def rotate(mesh, thetas):
    """ Compute rotation matrices given mesh and rotation angles in degrees.

    Parameters
    ----------
    mesh : array_like
        Nodal mesh defining the initial aerodynamic surface.
    thetas : array_like
        1-D array of rotation angles for each wing slice in degrees.

    Returns
    -------
    mesh : array_like
        Nodal mesh defining the twisted aerodynamic surface.

    """
    te = mesh[-1]
    le = mesh[ 0]
    quarter_chord = 0.25 * te + 0.75 * le

    ny = mesh.shape[1]
    nx = mesh.shape[0]

    rad_thetas = thetas * numpy.pi / 180.

    mats = numpy.zeros((ny, 3, 3), dtype=data_type)
    mats[:, 0, 0] = cos(rad_thetas)
    mats[:, 0, 2] = sin(rad_thetas)
    mats[:, 1, 1] = 1
    mats[:, 2, 0] = -sin(rad_thetas)
    mats[:, 2, 2] = cos(rad_thetas)
    for ix in range(nx):
        row = mesh[ix]
        row[:] = numpy.einsum("ikj, ij -> ik", mats, row - quarter_chord)
        row += quarter_chord

    return mesh

def scale_x(mesh, chord_dist):
    te = mesh[-1]
    le = mesh[ 0]
    quarter_chord = 0.25 * te + 0.75 * le

    ny = mesh.shape[1]
    nx = mesh.shape[0]

    for i in range(ny):
        mesh[:, i, 0] = (mesh[:, i, 0] - quarter_chord[i, 0]) * chord_dist[i] + \
            quarter_chord[i, 0]

def scale_x_d(mesh, chord_dist):
    derivs = numpy.zeros((numpy.prod(mesh.shape), len(chord_dist)))

    te = mesh[-1]
    le = mesh[ 0]
    quarter_chord = 0.25 * te + 0.75 * le

    ny = mesh.shape[1]
    nx = mesh.shape[0]

    chord_distd = numpy.zeros((len(chord_dist)))
    for j in range(len(chord_dist)):
        chord_distd[:] = 0.
        chord_distd[j] = 1.

        meshd = numpy.zeros(mesh.shape)
        for i in range(ny):
            meshd[:, i, 0] = (mesh[:, i, 0] - quarter_chord[i, 0]) * chord_distd[i]

        derivs[:, j] = meshd.flatten()

    return derivs


def sweep(mesh, angle, symmetry):
    """ Apply shearing sweep. Positive sweeps back.

    Parameters
    ----------
    mesh : array_like
        Nodal mesh defining the initial aerodynamic surface.
    angle : float
        Shearing sweep angle in degrees.
    symmetry : boolean
        Flag set to true if surface is reflected about y=0 plane.

    Returns
    -------
    mesh : array_like
        Nodal mesh defining the swept aerodynamic surface.

    """

    num_x, num_y, _ = mesh.shape
    le = mesh[0]
    p180 = numpy.pi / 180
    tan_theta = tan(p180*angle)

    if symmetry:
        y0 = le[-1, 1]
        dx = -(le[:, 1] - y0) * tan_theta

    else:
        ny2 = int((num_y - 1) / 2)
        y0 = le[ny2, 1]

        dx_right = (le[ny2:, 1] - y0) * tan_theta
        dx_left = -(le[:ny2, 1] - y0) * tan_theta
        dx = numpy.hstack((dx_left, dx_right))

    for i in xrange(num_x):
        mesh[i, :, 0] += dx

def dihedral(mesh, angle, symmetry):
    """ Apply dihedral angle. Positive angles up.

    Parameters
    ----------
    mesh : array_like
        Nodal mesh defining the initial aerodynamic surface.
    angle : float
        Dihedral angle in degrees.
    symmetry : boolean
        Flag set to true if surface is reflected about y=0 plane.

    Returns
    -------
    mesh : array_like
        Nodal mesh defining the aerodynamic surface with dihedral angle.

    """

    num_x, num_y, _ = mesh.shape
    le = mesh[0]
    p180 = numpy.pi / 180
    tan_theta = tan(p180*angle)

    if symmetry:
        y0 = le[-1, 1]
        dx = -(le[:, 1] - y0) * tan_theta

    else:
        ny2 = int((num_y-1) / 2)
        y0 = le[ny2, 1]
        dx_right = (le[ny2:, 1] - y0) * tan_theta
        dx_left = -(le[:ny2, 1] - y0) * tan_theta
        dx = numpy.hstack((dx_left, dx_right))

    for i in xrange(num_x):
        mesh[i, :, 2] += dx


def stretch(mesh, length):
    """ Stretch mesh in spanwise direction to reach specified length.

    Parameters
    ----------
    mesh : array_like
        Nodal mesh defining the initial aerodynamic surface.
    length : float
        Relative stetch ratio in the spanwise direction.

    Returns
    -------
    mesh : array_like
        Nodal mesh defining the stretched aerodynamic surface.

    """

    le = mesh[0]

    num_x, num_y, _ = mesh.shape

    span = le[-1, 1] - le[0, 1]
    dy = (length - span) / (num_y - 1) * numpy.arange(1, num_y)

    for i in xrange(num_x):
        mesh[i, 1:, 1] += dy


def taper(mesh, taper_ratio, symmetry):
    """ Alter the spanwise chord to produce a tapered wing.

    Parameters
    ----------
    mesh : array_like
        Nodal mesh defining the initial aerodynamic surface.
    taper_ratio : float
        Taper ratio for the wing; 1 is untapered, 0 goes to a point.
    symmetry : boolean
        Flag set to true if surface is reflected about y=0 plane.

    Returns
    -------
    mesh : array_like
        Nodal mesh defining the tapered aerodynamic surface.

    """

    le = mesh[0]
    te = mesh[-1]
    num_x, num_y, _ = mesh.shape
    center_chord = .5 * te + .5 * le

    if symmetry:
        taper = numpy.linspace(1, taper_ratio, num_y)[::-1]

        for i in xrange(num_x):
            for ind in xrange(3):
                mesh[i, :, ind] = (mesh[i, :, ind] - center_chord[:, ind]) * \
                    taper + center_chord[:, ind]

    else:
        ny2 = int((num_y + 1) / 2)
        taper = numpy.linspace(1, taper_ratio, ny2)[::-1]

        dx = numpy.hstack((taper, taper[::-1][1:]))

        for i in xrange(num_x):
            for ind in xrange(3):
                mesh[i, :, ind] = (mesh[i, :, ind] - center_chord[:, ind]) * \
                    dx + center_chord[:, ind]


class GeometryMesh(Component):
    """
    OpenMDAO component that performs mesh manipulation functions.

    """

    def __init__(self, surface):
        super(GeometryMesh, self).__init__()

        ny = surface['num_y']
        self.mesh = surface['mesh']

        self.add_param('span', val=58.7630524)
        self.add_param('sweep', val=0.)
        self.add_param('dihedral', val=0.)
        self.add_param('twist', val=numpy.zeros(ny), dtype=data_type)
        self.add_param('chord_dist', val=numpy.zeros(ny), dtype=data_type)
        self.add_param('taper', val=1.)
        self.add_output('mesh', val=self.mesh)

        self.symmetry = surface['symmetry']

        if not fortran_flag:
            self.deriv_options['type'] = 'fd'
            self.deriv_options['form'] = 'central'

    def solve_nonlinear(self, params, unknowns, resids):
        mesh = self.mesh.copy()

        if fortran_flag:
            # Does not have span stretching coded yet
            mesh = OAS_API.oas_api.manipulate_mesh(mesh, params['sweep'],
                params['twist'], params['chord_dist'], params['dihedral'],
                params['taper'], self.symmetry)

        else:

            mesh = self.mesh.copy()
            # stretch(mesh, params['span'])
            sweep(mesh, params['sweep'], self.symmetry)
            scale_x(mesh, params['chord_dist'])
            rotate(mesh, params['twist'])
            dihedral(mesh, params['dihedral'], self.symmetry)
            taper(mesh, params['taper'], self.symmetry)

        unknowns['mesh'] = mesh

    def apply_linear(self, params, unknowns, dparams, dunknowns, dresids, mode):

        mesh = self.mesh.copy()

        if mode == 'fwd':
            if 'sweep' in dparams:
                sweepd = dparams['sweep']
            else:
                sweepd = 0.
            if 'twist' in dparams:
                twistd = dparams['twist']
            else:
                twistd = numpy.zeros(params['twist'].shape)
            if 'chord_dist' in dparams:
                chord_distd = dparams['chord_dist']
            else:
                chord_distd = numpy.zeros(params['chord_dist'].shape)
            if 'dihedral' in dparams:
                dihedrald = dparams['dihedral']
            else:
                dihedrald = 0.
            if 'taper' in dparams:
                taperd = dparams['taper']
            else:
                taperd = 0.

            mesh, dresids['mesh'] = OAS_API.oas_api.manipulate_mesh_d(mesh, params['sweep'],
            sweepd, params['twist'], twistd, params['chord_dist'],
            chord_distd, params['dihedral'], dihedrald,
            params['taper'], taperd, self.symmetry)

        if mode == 'rev':
            sweepb, twistb, chord_distb, dihedralb, taperb, mesh = OAS_API.oas_api.manipulate_mesh_b(mesh, params['sweep'], params['twist'], params['chord_dist'], params['dihedral'],
            params['taper'], self.symmetry, dresids['mesh'])

            if 'sweep' in dparams:
                dparams['sweep'] = sweepb
            if 'twist' in dparams:
                dparams['twist'] = twistb
            if 'chord_dist' in dparams:
                dparams['chord_dist'] = chord_distb
            if 'dihedral' in dparams:
                dparams['dihedral'] = dihedralb
            if 'taper' in dparams:
                dparams['taper'] = taperb

class MonotonicTaper(Component):
    def __init__(self, surface):
        super(MonotonicTaper, self).__init__()
        self.nvars = surface['num_y']
        self.add_param('chord_dist', val=numpy.zeros(self.nvars), dtype=data_type)
        self.add_output('monotonic', val=numpy.zeros(self.nvars-1))

    def solve_nonlinear(self, params, unknowns, resids):
        # Monotonic chord constraint
        for i in range(self.nvars-1):
            unknowns['monotonic'][i] = params['chord_dist'][i+1] - params['chord_dist'][i]

    def linearize(self, params, unknowns, resids):
        jac = self.alloc_jacobian()
        for i in range(self.nvars - 1):
            jac['monotonic', 'chord_dist'][i,i] = -1.0
            jac['monotonic', 'chord_dist'][i,i+1] = 1.0

        return jac

def gen_crm_mesh(num_x, num_y, span, chord, span_cos_spacing=0., chord_cos_spacing=0., wing_type="CRM:jig"):
    """ Generate simple rectangular wing mesh.

    Parameters
    ----------
    num_x : float
        Desired number of chordwise node points for the final mesh.
    num_y : float
        Desired number of chordwise node points for the final mesh.
    span : float
        Total wingspan.
    chord : float
        Root chord.
    span_cos_spacing : float (optional)
        Blending ratio of uniform and cosine spacing in the spanwise direction.
        A value of 0. corresponds to uniform spacing and a value of 1.
        corresponds to regular cosine spacing. This increases the number of
        spanwise node points near the wingtips.
    chord_cos_spacing : float (optional)
        Blending ratio of uniform and cosine spacing in the chordwise direction.
        A value of 0. corresponds to uniform spacing and a value of 1.
        corresponds to regular cosine spacing. This increases the number of
        chordwise node points near the wingtips.
    wing_type : string (optional)
        Describes the desired CRM shape. Current options are:
        "CRM:jig" (undeformed jig shape),
        "CRM:alpha_2.75" (shape from wind tunnel testing at a=2.75 from DPW6)

    Returns
    -------
    mesh : array_like
        Rectangular nodal mesh defining the final aerodynamic surface with the
        specified parameters.
    eta : array_like
        Spanwise locations of the airfoil slices. Later used in the
        interpolation function to obtain correct twist values during at
        points along the span that are not aligned with these slices.
    twist : array_like
        Twist along the span at the spanwise eta locations. We use these twists
        as training points for interpolation to obtain twist values at
        arbitrary points along the span.

    """

    # First we check if the user requested a deformed CRM wing
    if "alpha_2.75" in wing_type:
        # eta, xle, yle, zle, twist, chord
        # Info taken from DPW6_wb_medium.cgns by taking slices in TecPlot
        # Processed using `process_crm_slices.py`.
        # This corresponds to the CRM shape in the wind tunnel when alpha is set to 2.75.
        #
        # Note that the first line is copied from the jig shape because we do not
        # have information about the wing section inside of the fuselage
        # because the cgns file has the wing-body together with cut meshes.
        raw_crm_points = numpy.array([
[0.,            904.294,           0.0,          174.126,         6.7166,       536.181], # 0
[0.1049403748,  993.7138118110,  121.2598425197, 175.8828985433,  4.3064288580, 466.4631457464],
[0.1475622617, 1030.0125685039,  170.5099362598, 176.8216161417,  3.5899506711, 437.2201981238],
[0.1901841486, 1067.3502531496,  219.7600300000, 177.1741468504,  3.0704027125, 408.1221709540],
[0.2328060355, 1104.8785736220,  269.0101237402, 177.6574524409,  2.4218173307, 379.0574671904],
[0.2754279225, 1142.4137122047,  318.2602174803, 178.3693226772,  1.7348508885, 350.0486268673],
[0.3180498094, 1179.9041996063,  367.5103112205, 179.5740685039,  1.1260783099, 321.1282460136],
[0.3606716964, 1217.4135641732,  416.7604051181, 181.3039487402,  0.6320326304, 292.2667642082],
[0.4032935833, 1254.9198700787,  466.0104988189, 183.4083569685,  0.1001945603, 276.4326600724],
[0.4459154702, 1292.4185027559,  515.2605925197, 186.2613112205, -0.4313792294, 264.3934743324],
[0.4885373571, 1329.9177011811,  564.5106862205, 189.7670859055, -0.8832854545, 252.3697629697],
[0.5311592440, 1367.4235448819,  613.7607799213, 194.1769490157, -1.2064052719, 240.3458791762],
[0.5737811308, 1404.9283440945,  663.0108736220, 199.1855426378, -1.4882476942, 228.3242182875],
[0.6164030177, 1442.4362011811,  712.2609673228, 204.8247526772, -1.7694295897, 216.3000209815],
[0.6590249046, 1479.9483543307,  761.5110610236, 211.1363005512, -2.0434689434, 204.2772529463],
[0.7016467915, 1517.4654637795,  810.7611547244, 218.1652856693, -2.2699133205, 192.2467756712],
[0.7442686784, 1554.9898948819,  860.0112484252, 226.0318298031, -2.4841677570, 180.2040428725],
[0.7868905656, 1592.4996003937,  909.2613425197, 234.5204538976, -2.7327835422, 168.1745654077],
[0.8295124525, 1630.0325484252,  958.5114362205, 243.7741411811, -2.9822838340, 156.1260647832],
[0.8721343394, 1667.5602492126, 1007.7615299213, 253.6186485827, -3.2409127562, 144.0854789941],
[0.9147562262, 1705.0931177165, 1057.0116236220, 263.9955130315, -3.6316327683, 132.0598826266],
[0.9573781131, 1742.6251350394, 1106.2617173228, 274.7061785039, -4.1417911628, 120.0493724390],
[1.0000000000, 1780.1757874016, 1155.5118110236, 285.5045883858, -4.6778406881, 108.0150257616]])

    # If no special wing was requested, we'll use the jig shape
    else:
        # eta, xle, yle, zle, twist, chord
        # Info taken from AIAA paper 2008-6919 by Vassberg
        raw_crm_points = numpy.array([
         [0.,   904.294,    0.0,   174.126, 6.7166,  536.181], # 0
         [.1,   989.505,  115.675, 175.722, 4.4402,  468.511],
         [.15, 1032.133,  173.513, 176.834, 3.6063,  434.764],
         [.2,  1076.030,  231.351, 177.912, 2.2419,  400.835],
         [.25, 1120.128,  289.188, 177.912, 2.2419,  366.996],
         [.3,  1164.153,  347.026, 178.886, 1.5252,  333.157],
         [.35, 1208.203,  404.864, 180.359,  .9379,  299.317], # 6 yehudi break
         [.4,  1252.246,  462.701, 182.289,  .4285,  277.288],
         [.45, 1296.289,  520.539, 184.904, -.2621,  263],
         [.5,  1340.329,  578.377, 188.389, -.6782,  248.973],
         [.55, 1384.375,  636.214, 192.736, -.9436,  234.816],
         [.60, 1428.416,  694.052, 197.689, -1.2067, 220.658],
         [.65, 1472.458,  751.890, 203.294, -1.4526, 206.501],
         [.7,  1516.504,  809.727, 209.794, -1.6350, 192.344],
         [.75, 1560.544,  867.565, 217.084, -1.8158, 178.186],
         [.8,  1604.576,  925.402, 225.188, -2.0301, 164.029],
         [.85, 1648.616,  983.240, 234.082, -2.2772, 149.872],
         [.9,  1692.659, 1041.078, 243.625, -2.5773, 135.714],
         [.95, 1736.710, 1098.915, 253.691, -3.1248, 121.557],
         [1.,  1780.737, 1156.753, 263.827, -3.75,   107.4] # 19
        ])

    # Get the leading edge of the raw crm points
    le = numpy.vstack((raw_crm_points[:,1],
                    raw_crm_points[:,2],
                    raw_crm_points[:,3]))

    # Get the chord, twist(in correct order), and eta values from the points
    chord = raw_crm_points[:, 5]
    twist = raw_crm_points[:, 4][::-1]
    eta = raw_crm_points[:, 0]

    # Get the trailing edge of the crm points, based on the chord + le distance.
    # Note that we do not account for twist here; instead we set that using
    # the twist design variable later in run_classes.py.
    te = numpy.vstack((raw_crm_points[:,1] + chord,
                       raw_crm_points[:,2],
                       raw_crm_points[:,3]))

    # Get the number of points that define this CRM shape and create a mesh
    # array based on this size
    n_raw_points = raw_crm_points.shape[0]
    mesh = numpy.empty((2, n_raw_points, 3), dtype=data_type)

    # Set the leading and trailing edges of the mesh matrix
    mesh[0, :, :] = le.T
    mesh[1, :, :] = te.T

    # Convert the mesh points to meters from inches.
    raw_mesh = mesh * 0.0254

    # Create the blended spacing using the user input for span_cos_spacing
    ny2 = (num_y + 1) // 2
    beta = numpy.linspace(0, numpy.pi/2, ny2)

    # Distribution for cosine spacing
    cosine = numpy.cos(beta)

    # Distribution for uniform spacing
    uniform = numpy.linspace(0, 1., ny2)[::-1]

    # Combine the two distrubtions using span_cos_spacing as the weighting factor.
    # span_cos_spacing == 1. is for fully cosine, 0. for uniform
    lins = cosine * span_cos_spacing + (1 - span_cos_spacing) * uniform

    # Populate a mesh object with the desired num_y dimension based on
    # interpolated values from the raw CRM points.
    mesh = numpy.empty((2, ny2, 3), dtype=data_type)
    for j in range(2):
        for i in range(3):
            mesh[j, :, i] = numpy.interp(lins[::-1], eta, raw_mesh[j, :, i].real)

    # That is just one half of the mesh and we later expect the full mesh,
    # even if we're using symmetry == True.
    # So here we mirror and stack the two halves of the wing.
    left_half = mesh.copy()
    left_half[:, :, 1] *= -1.
    mesh = numpy.hstack((left_half[:, ::-1, :], mesh[:, 1:, :]))

    # If we need to add chordwise panels, do so
    if num_x > 2:
        mesh = add_chordwise_panels(mesh, num_x, chord_cos_spacing)

    return mesh, eta, twist


def add_chordwise_panels(mesh, num_x, chord_cos_spacing):
    """ Divide the wing into multiple chordwise panels.

    Parameters
    ----------
    mesh : array_like
        Nodal mesh defining the initial aerodynamic surface with only
        the leading and trailing edges defined.
    num_x : float
        Desired number of chordwise node points for the final mesh.
    chord_cos_spacing : float
        Blending ratio of uniform and cosine spacing in the chordwise direction.
        A value of 0. corresponds to uniform spacing and a value of 1.
        corresponds to regular cosine spacing. This increases the number of
        chordwise node points near the wingtips.


    Returns
    -------
    new_mesh : array_like
        Nodal mesh defining the final aerodynamic surface with the
        specified number of chordwise node points.

    """

    # Obtain mesh and num properties
    num_y = mesh.shape[1]
    ny2 = (num_y + 1) / 2
    nx2 = (num_x + 1) / 2

    # Create beta, an array of linear sampling points to pi/2
    beta = numpy.linspace(0, numpy.pi/2, nx2)

    # Obtain the two spacings that we will use to blend
    cosine = .5 * numpy.cos(beta)  # cosine spacing
    uniform = numpy.linspace(0, .5, nx2)[::-1]  # uniform spacing

    # Create half of the wing in the chordwise direction
    half_wing = cosine * chord_cos_spacing + (1 - chord_cos_spacing) * uniform

    # Mirror this half wing into a full wing; offset by 0.5 so it goes 0 to 1
    full_wing_x = numpy.hstack((-half_wing[:-1], half_wing[::-1])) + .5

    # Obtain the leading and trailing edges
    le = mesh[ 0, :, :]
    te = mesh[-1, :, :]

    # Create a new mesh with the desired num_x and set the leading and trailing edge values
    new_mesh = numpy.zeros((num_x, num_y, 3), dtype=data_type)
    new_mesh[ 0, :, :] = le
    new_mesh[-1, :, :] = te

    for i in xrange(1, num_x-1):
        w = full_wing_x[i]
        new_mesh[i, :, :] = (1 - w) * le + w * te

    return new_mesh


def gen_rect_mesh(num_x, num_y, span, chord, span_cos_spacing=0., chord_cos_spacing=0.):
    """ Generate simple rectangular wing mesh.

    Parameters
    ----------
    num_x : float
        Desired number of chordwise node points for the final mesh.
    num_y : float
        Desired number of chordwise node points for the final mesh.
    span : float
        Total wingspan.
    chord : float
        Root chord.
    span_cos_spacing : float (optional)
        Blending ratio of uniform and cosine spacing in the spanwise direction.
        A value of 0. corresponds to uniform spacing and a value of 1.
        corresponds to regular cosine spacing. This increases the number of
        spanwise node points near the wingtips.
    chord_cos_spacing : float (optional)
        Blending ratio of uniform and cosine spacing in the chordwise direction.
        A value of 0. corresponds to uniform spacing and a value of 1.
        corresponds to regular cosine spacing. This increases the number of
        chordwise node points near the wingtips.

    Returns
    -------
    mesh : array_like
        Rectangular nodal mesh defining the final aerodynamic surface with the
        specified parameters.

    """

    mesh = numpy.zeros((num_x, num_y, 3), dtype=data_type)
    ny2 = (num_y + 1) / 2
    beta = numpy.linspace(0, numpy.pi/2, ny2)

    # mixed spacing with span_cos_spacing as a weighting factor
    # this is for the spanwise spacing
    cosine = .5 * numpy.cos(beta)  # cosine spacing
    uniform = numpy.linspace(0, .5, ny2)[::-1]  # uniform spacing
    half_wing = cosine * span_cos_spacing + (1 - span_cos_spacing) * uniform
    full_wing = numpy.hstack((-half_wing[:-1], half_wing[::-1])) * span

    nx2 = (num_x + 1) / 2
    beta = numpy.linspace(0, numpy.pi/2, nx2)

    # mixed spacing with span_cos_spacing as a weighting factor
    # this is for the chordwise spacing
    cosine = .5 * numpy.cos(beta)  # cosine spacing
    uniform = numpy.linspace(0, .5, nx2)[::-1]  # uniform spacing
    half_wing = cosine * chord_cos_spacing + (1 - chord_cos_spacing) * uniform
    full_wing_x = numpy.hstack((-half_wing[:-1], half_wing[::-1])) * chord

    # Special case if there are only 2 chordwise nodes
    if num_x <= 2:
        full_wing_x = numpy.array([0., chord])

    for ind_x in xrange(num_x):
        for ind_y in xrange(num_y):
            mesh[ind_x, ind_y, :] = [full_wing_x[ind_x], full_wing[ind_y], 0]

    return mesh


class Bspline(Component):
    """
    General function to translate from control points to actual points
    using a b-spline representation.

    """

    def __init__(self, cpname, ptname, n_input, n_output):
        super(Bspline, self).__init__()
        self.cpname = cpname
        self.ptname = ptname
        self.jac = get_bspline_mtx(n_input, n_output, order=min(n_input, 4))
        self.add_param(cpname, val=numpy.zeros(n_input))
        self.add_output(ptname, val=numpy.zeros(n_output))

    def solve_nonlinear(self, params, unknowns, resids):
        unknowns[self.ptname] = self.jac.dot(params[self.cpname])

    def linearize(self, params, unknowns, resids):
        return {(self.ptname, self.cpname): self.jac}


class LinearInterp(Component):
    """ Linear interpolation used to create linearly varying parameters. """

    def __init__(self, num_y, name):
        super(LinearInterp, self).__init__()

        self.add_param('linear_'+name, val=numpy.zeros(2))
        self.add_output(name, val=numpy.zeros(num_y))

        self.deriv_options['type'] = 'cs'
        self.deriv_options['form'] = 'central'
        #self.deriv_options['extra_check_partials_form'] = "central"

        self.num_y = num_y
        self.vname = name

    def solve_nonlinear(self, params, unknowns, resids):
        a, b = params['linear_'+self.vname]

        if self.num_y % 2 == 0:
            imax = int(self.num_y/2)
        else:
            imax = int((self.num_y+1)/2)
        for ind in xrange(imax):
            w = 1.0*ind/(imax-1)

            unknowns[self.vname][ind] = a*(1-w) + b*w
            unknowns[self.vname][-1-ind] = a*(1-w) + b*w


if __name__ == "__main__":
    """ Test mesh generation and view results in .html file. """

    import plotly.offline as plt
    import plotly.graph_objs as go

    from plot_tools import wire_mesh, build_layout

    thetas = numpy.zeros(20)
    thetas[10:] += 10

    mesh = gen_crm_mesh(3, 3)

    # new_mesh = rotate(mesh, thetas)

    # new_mesh = sweep(mesh, 20)

    new_mesh = stretch(mesh, 100)

    # wireframe_orig = wire_mesh(mesh)
    wireframe_new = wire_mesh(new_mesh)
    layout = build_layout()

    fig = go.Figure(data=wireframe_new, layout=layout)
    plt.plot(fig, filename="wing_3d.html")
