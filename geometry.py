""" Manipulate geometry mesh based on high-level design parameters. """

from __future__ import division, print_function
import numpy as np
from numpy import cos, sin, tan

from openmdao.api import Component
from b_spline import get_bspline_mtx
from spatialbeam import radii
from CRM_definitions import get_crm_points

try:
    import OAS_API
    fortran_flag = True
    data_type = float
except:
    fortran_flag = False
    data_type = complex


class GeometryMesh(Component):
    """
    OpenMDAO component that performs mesh manipulation functions. It reads in
    the initial mesh from the surface dictionary and outputs the altered
    mesh based on the geometric design variables.

    Depending on the design variables selected or the supplied geometry information,
    only some of the follow parameters will actually be given to this component.
    If parameters are not active (they do not deform the mesh), then
    they will not be given to this component.

    Parameters
    ----------
    sweep : float
        Shearing sweep angle in degrees.
    dihedral : float
        Dihedral angle in degrees.
    twist[ny] : numpy array
        1-D array of rotation angles for each wing slice in degrees.
    chord_dist[ny] : numpy array
        Chord length for each panel edge.
    taper : float
        Taper ratio for the wing; 1 is untapered, 0 goes to a point at the tip.

    Returns
    -------
    mesh[nx, ny, 3] : numpy array
        Modified mesh based on the initial mesh in the surface dictionary and
        the geometric design variables.
    """

    def __init__(self, surface, desvars={}):
        super(GeometryMesh, self).__init__()

        name = surface['name']
        self.surface = surface

        # Strip the surface names from the desvars list and save this
        # modified list as self.desvars
        desvar_names = []
        for desvar in desvars.keys():

            # Check to make sure that the surface's name is in the design
            # variable and only add the desvar to the list if it corresponds
            # to this surface.
            if name[:-1] in desvar:
                desvar_names.append(''.join(desvar.split('.')[1:]))

        self.desvar_names = desvar_names

        ny = surface['num_y']
        self.mesh = surface['mesh']

        # Variables that should be initialized to one
        ones_list = ['taper', 'chord_cp']

        # Variables that should be initialized to zero
        zeros_list = ['sweep', 'dihedral', 'twist_cp', 'xshear_cp', 'zshear_cp']

        # Variables that should be initialized to given value
        set_list = ['span']

        # Make a list of all geometry variables by adding all individual lists
        all_geo_vars = ones_list + zeros_list + set_list
        self.geo_params = {}
        for var in all_geo_vars:
            if len(var.split('_')) > 1:
                param = var.split('_')[0]
                if var in ones_list:
                    val = np.ones(ny)
                elif var in zeros_list:
                    val = np.zeros(ny)
                else:
                    val = surface[var]
            else:
                param = var
                if var in ones_list:
                    val = 1.0
                elif var in zeros_list:
                    val = 0.0
                else:
                    val = surface[var]
            self.geo_params[param] = val

            # If the user supplied a variable or it's a desvar, we add it as a
            # parameter.
            if var in desvar_names or var in surface['initial_geo']:
                self.add_param(param, val=val)

        self.add_output('mesh', val=self.mesh)

        # If the user doesn't provide the radius or it's not a desver, then we must
        # compute it here
        if 'radius_cp' not in desvar_names and 'radius_cp' not in surface['initial_geo']:
            self.compute_radius = True
            self.add_output('radius', val=np.zeros((ny - 1)))
        else:
            self.compute_radius = False

        self.symmetry = surface['symmetry']

        # This flag determines whether or not changes in z (dihedral) add an
        # additional rotation matrix to modify the twist direction
        self.rotate_x = True

    def initialize_partials(self):
        if not fortran_flag:
            self.approx_partials('*', '*')

    def solve_nonlinear(self, params, unknowns, resids):
        mesh = self.mesh.copy()
        self.geo_params.update(params)

        if fortran_flag:
            mesh = OAS_API.oas_api.manipulate_mesh(mesh, self.geo_params['taper'],
                self.geo_params['chord'], self.geo_params['sweep'], self.geo_params['xshear'],
                self.geo_params['dihedral'], self.geo_params['zshear'],
                self.geo_params['twist'], self.geo_params['span'], self.symmetry, self.rotate_x)

        else:
            taper(mesh, self.geo_params['taper'], self.symmetry)
            scale_x(mesh, self.geo_params['chord'])
            stretch(mesh, self.geo_params['span'], self.symmetry)
            sweep(mesh, self.geo_params['sweep'], self.symmetry)
            shear_x(mesh, self.geo_params['xshear'])
            dihedral(mesh, self.geo_params['dihedral'], self.symmetry)
            shear_z(mesh, self.geo_params['zshear'])
            rotate(mesh, self.geo_params['twist'], self.symmetry, self.rotate_x)

        # Only compute the radius on the first iteration
        if self.compute_radius and 'radius_cp' not in self.desvar_names:
            # Get spar radii and interpolate to radius control points.
            # Need to refactor this at some point since the derivatives are wrong.
            unknowns['radius'] = radii(mesh, self.surface['t_over_c'])
            self.compute_radius = False

        unknowns['mesh'] = mesh

    def apply_linear(self, params, unknowns, dparams, dunknowns, dresids, mode):
        mesh = self.mesh.copy()

        # We actually use the values in self.geo_params to modify the mesh,
        # but we update self.geo_params using the OpenMDAO params here.
        # This makes the geometry manipulation process work for any combination
        # of design variables without having special logic.
        self.geo_params.update(params)

        if mode == 'fwd':

            # We don't know which parameters will be used for a given case
            # so we must check
            if 'sweep' in dparams:
                sweepd = dparams['sweep']
            else:
                sweepd = 0.
            if 'twist' in dparams:
                twistd = dparams['twist']
            else:
                twistd = np.zeros(self.geo_params['twist'].shape)
            if 'chord' in dparams:
                chordd = dparams['chord']
            else:
                chordd = np.zeros(self.geo_params['chord'].shape)
            if 'dihedral' in dparams:
                dihedrald = dparams['dihedral']
            else:
                dihedrald = 0.
            if 'taper' in dparams:
                taperd = dparams['taper']
            else:
                taperd = 0.
            if 'xshear' in dparams:
                xsheard = dparams['xshear']
            else:
                xsheard = np.zeros(self.geo_params['xshear'].shape)
            if 'zshear' in dparams:
                zsheard = dparams['zshear']
            else:
                zsheard = np.zeros(self.geo_params['zshear'].shape)
            if 'span' in dparams:
                spand = dparams['span']
            else:
                spand = 0.

            mesh, dresids['mesh'] = OAS_API.oas_api.manipulate_mesh_d(mesh,
            self.geo_params['taper'], taperd, self.geo_params['chord'], chordd,
            self.geo_params['sweep'], sweepd, self.geo_params['xshear'], xsheard,
            self.geo_params['dihedral'], dihedrald, self.geo_params['zshear'],
            zsheard, self.geo_params['twist'], twistd, self.geo_params['span'],
            spand, self.symmetry, self.rotate_x)

        if mode == 'rev':
            taperb, chordb, sweepb, xshearb, dihedralb, zshearb, twistb, spanb, mesh = \
            OAS_API.oas_api.manipulate_mesh_b(mesh, self.geo_params['taper'],
            self.geo_params['chord'], self.geo_params['sweep'],
            self.geo_params['xshear'], self.geo_params['dihedral'],
            self.geo_params['zshear'], self.geo_params['twist'],
            self.geo_params['span'], self.symmetry, self.rotate_x, dresids['mesh'])

            if 'sweep' in dparams:
                dparams['sweep'] = sweepb
            if 'twist' in dparams:
                dparams['twist'] = twistb
            if 'chord' in dparams:
                dparams['chord'] = chordb
            if 'dihedral' in dparams:
                dparams['dihedral'] = dihedralb
            if 'taper' in dparams:
                dparams['taper'] = taperb
            if 'xshear' in dparams:
                dparams['xshear'] = xshearb
            if 'zshear' in dparams:
                dparams['zshear'] = zshearb
            if 'span' in dparams:
                dparams['span'] = spanb

class MonotonicConstraint(Component):
    """
    Produce a constraint that is violated if a user-chosen measure on the
    wing does not decrease monotonically from the root to the tip.

    Parameters
    ----------
    var_name : string
        The variable to which the user would like to apply the monotonic constraint.

    Returns
    -------
    monotonic[ny-1] : numpy array
        Values are greater than 0 if the constraint is violated.

    """
    def __init__(self, var_name, surface):
        super(MonotonicConstraint, self).__init__()

        self.var_name = var_name
        self.con_name = 'monotonic_' + var_name

        self.symmetry = surface['symmetry']
        self.ny = surface['num_y']

        self.add_param(self.var_name, val=np.zeros(self.ny))
        self.add_output(self.con_name, val=np.zeros(self.ny-1))

    def solve_nonlinear(self, params, unknowns, resids):
        # Compute the difference between adjacent variable values
        diff = params[self.var_name][:-1] - params[self.var_name][1:]
        if self.symmetry:
            unknowns[self.con_name] = diff
        else:
            ny2 = (self.ny - 1) // 2
            unknowns[self.con_name][:ny2] = diff[:ny2]
            unknowns[self.con_name][ny2:] = -diff[ny2:]

    def linearize(self, params, unknowns, resids):
        jac = self.alloc_jacobian()

        np.fill_diagonal(jac[self.con_name, self.var_name][:, :], 1)
        np.fill_diagonal(jac[self.con_name, self.var_name][:, 1:], -1)

        if not self.symmetry:
            ny2 = (self.ny - 1) // 2
            jac[self.con_name, self.var_name][ny2:, :] *= -1

        return jac

def gen_crm_mesh(num_x, num_y, span, chord, span_cos_spacing=0., chord_cos_spacing=0., wing_type="CRM:jig"):
    """
    Generate Common Research Model wing mesh.

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
    mesh[nx, ny, 3] : numpy array
        Rectangular nodal mesh defining the final aerodynamic surface with the
        specified parameters.
    eta : numpy array
        Spanwise locations of the airfoil slices. Later used in the
        interpolation function to obtain correct twist values at
        points along the span that are not aligned with these slices.
    twist : numpy array
        Twist along the span at the spanwise eta locations. We use these twists
        as training points for interpolation to obtain twist values at
        arbitrary points along the span.

    """

    # Call an external function to get the data points for the specific CRM
    # type requested. See `CRM_definitions.py` for more information and the
    # raw data.
    raw_crm_points = get_crm_points(wing_type)

    # Get the leading edge of the raw crm points
    le = np.vstack((raw_crm_points[:,1],
                    raw_crm_points[:,2],
                    raw_crm_points[:,3]))

    # Get the chord, twist(in correct order), and eta values from the points
    chord = raw_crm_points[:, 5]
    twist = raw_crm_points[:, 4][::-1]
    eta = raw_crm_points[:, 0]

    # Get the trailing edge of the crm points, based on the chord + le distance.
    # Note that we do not account for twist here; instead we set that using
    # the twist design variable later in run_classes.py.
    te = np.vstack((raw_crm_points[:,1] + chord,
                       raw_crm_points[:,2],
                       raw_crm_points[:,3]))

    # Get the number of points that define this CRM shape and create a mesh
    # array based on this size
    n_raw_points = raw_crm_points.shape[0]
    mesh = np.empty((2, n_raw_points, 3))

    # Set the leading and trailing edges of the mesh matrix
    mesh[0, :, :] = le.T
    mesh[1, :, :] = te.T

    # Convert the mesh points to meters from inches.
    raw_mesh = mesh * 0.0254

    # Create the blended spacing using the user input for span_cos_spacing
    ny2 = (num_y + 1) // 2
    beta = np.linspace(0, np.pi/2, ny2)

    # Distribution for cosine spacing
    cosine = np.cos(beta)

    # Distribution for uniform spacing
    uniform = np.linspace(0, 1., ny2)[::-1]

    # Combine the two distrubtions using span_cos_spacing as the weighting factor.
    # span_cos_spacing == 1. is for fully cosine, 0. for uniform
    lins = cosine * span_cos_spacing + (1 - span_cos_spacing) * uniform

    # Populate a mesh object with the desired num_y dimension based on
    # interpolated values from the raw CRM points.
    mesh = np.empty((2, ny2, 3))
    for j in range(2):
        for i in range(3):
            mesh[j, :, i] = np.interp(lins[::-1], eta, raw_mesh[j, :, i].real)

    # That is just one half of the mesh and we later expect the full mesh,
    # even if we're using symmetry == True.
    # So here we mirror and stack the two halves of the wing.
    left_half = mesh.copy()
    left_half[:, :, 1] *= -1.
    mesh = np.hstack((left_half[:, ::-1, :], mesh[:, 1:, :]))

    # If we need to add chordwise panels, do so
    if num_x > 2:
        mesh = add_chordwise_panels(mesh, num_x, chord_cos_spacing)

    return mesh, eta, twist


def add_chordwise_panels(mesh, num_x, chord_cos_spacing):
    """
    Generate a new mesh with multiple chordwise panels.

    Parameters
    ----------
    mesh[nx, ny, 3] : numpy array
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
    new_mesh[nx, ny, 3] : numpy array
        Nodal mesh defining the final aerodynamic surface with the
        specified number of chordwise node points.

    """

    # Obtain mesh and num properties
    num_y = mesh.shape[1]
    ny2 = (num_y + 1) // 2
    nx2 = (num_x + 1) // 2

    # Create beta, an array of linear sampling points to pi/2
    beta = np.linspace(0, np.pi/2, nx2)

    # Obtain the two spacings that we will use to blend
    cosine = .5 * np.cos(beta)  # cosine spacing
    uniform = np.linspace(0, .5, nx2)[::-1]  # uniform spacing

    # Create half of the wing in the chordwise direction
    half_wing = cosine * chord_cos_spacing + (1 - chord_cos_spacing) * uniform

    if chord_cos_spacing == 0.:
        full_wing_x = np.linspace(0, 1., num_x)

    else:
        # Mirror this half wing into a full wing; offset by 0.5 so it goes 0 to 1
        full_wing_x = np.hstack((-half_wing[:-1], half_wing[::-1])) + .5

    # Obtain the leading and trailing edges
    le = mesh[ 0, :, :]
    te = mesh[-1, :, :]

    # Create a new mesh with the desired num_x and set the leading and trailing edge values
    new_mesh = np.zeros((num_x, num_y, 3))
    new_mesh[ 0, :, :] = le
    new_mesh[-1, :, :] = te

    for i in range(1, num_x-1):
        w = full_wing_x[i]
        new_mesh[i, :, :] = (1 - w) * le + w * te

    return new_mesh


def gen_rect_mesh(num_x, num_y, span, chord, span_cos_spacing=0., chord_cos_spacing=0.):
    """
    Generate simple rectangular wing mesh.

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
    mesh[nx, ny, 3] : numpy array
        Rectangular nodal mesh defining the final aerodynamic surface with the
        specified parameters.
    """

    mesh = np.zeros((num_x, num_y, 3))
    ny2 = (num_y + 1) // 2
    beta = np.linspace(0, np.pi/2, ny2)

    # mixed spacing with span_cos_spacing as a weighting factor
    # this is for the spanwise spacing
    cosine = .5 * np.cos(beta)  # cosine spacing
    uniform = np.linspace(0, .5, ny2)[::-1]  # uniform spacing
    half_wing = cosine * span_cos_spacing + (1 - span_cos_spacing) * uniform
    full_wing = np.hstack((-half_wing[:-1], half_wing[::-1])) * span

    if chord_cos_spacing == 0.:
        full_wing_x = np.linspace(0, chord, num_x)

    else:
        nx2 = (num_x + 1) / 2
        beta = np.linspace(0, np.pi/2, nx2)

        # mixed spacing with span_cos_spacing as a weighting factor
        # this is for the chordwise spacing
        cosine = .5 * np.cos(beta)  # cosine spacing
        uniform = np.linspace(0, .5, nx2)[::-1]  # uniform spacing
        half_wing = cosine * chord_cos_spacing + (1 - chord_cos_spacing) * uniform
        full_wing_x = np.hstack((-half_wing[:-1], half_wing[::-1])) * chord

    for ind_x in range(num_x):
        for ind_y in range(num_y):
            mesh[ind_x, ind_y, :] = [full_wing_x[ind_x], full_wing[ind_y], 0]

    return mesh


class Bspline(Component):
    """
    General function to translate from control points to actual points
    using a b-spline representation.

    Parameters
    ----------
    cpname : string
        Name of the OpenMDAO component containing the control point values.
    ptname : string
        Name of the OpenMDAO component that will contain the interpolated
        b-spline values.
    n_input : int
        Number of input control points.
    n_output : int
        Number of outputted interpolated b-spline points.
    """

    def __init__(self, cpname, ptname, n_input, n_output):
        super(Bspline, self).__init__()
        self.cpname = cpname
        self.ptname = ptname
        self.jac = get_bspline_mtx(n_input, n_output, order=min(n_input, 4))
        self.add_param(cpname, val=np.zeros(n_input))
        self.add_output(ptname, val=np.zeros(n_output))

    def solve_nonlinear(self, params, unknowns, resids):
        unknowns[self.ptname] = self.jac.dot(params[self.cpname])

    def linearize(self, params, unknowns, resids):
        return {(self.ptname, self.cpname): self.jac}
