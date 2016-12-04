"""
Define the aerodynamic analysis component using a vortex lattice method.

We input a nodal mesh and properties of the airflow to calculate the
circulations of the horseshoe vortices. We then compute the forces, lift,
and drag acting on the lifting surfaces.

Note that some of the parameters and unknowns has the surface name
prepended on it. E.g., 'def_mesh' on a surface called 'wing' would be
'wing_def_mesh', etc. Please see an N2 diagram (the .html file produced by
this code) for more information about which parameters are renamed.

.. todo:: fix depiction of wing
Depiction of wing:

    y -->
 x  -----------------  Leading edge
 |  | + | + | + | + |
 v  |   |   |   |   |
    | o | o | o | o |
    -----------------  Trailing edge

The '+' symbols represent the bound points on the 1/4 chord line, which are
used to compute drag.
The 'o' symbols represent the collocation points on the 3/4 chord line, where
the flow tangency condition is inforced.

For this depiction, num_x = 2 and num_y = 5.
"""

from __future__ import division
import numpy

from openmdao.api import Component, Group
from scipy.linalg import lu_factor, lu_solve
try:
    import OAS_API
    fortran_flag = True
except:
    fortran_flag = False

def view_mat(mat):
    """ Helper function used to visually examine matrices. """
    import matplotlib.pyplot as plt
    if len(mat.shape) > 2:
        mat = numpy.sum(mat, axis=2)
    im = plt.imshow(mat.real, interpolation='none')
    plt.colorbar(im, orientation='horizontal')
    plt.show()


def norm(vec):
    """ Finds the 2-norm of a vector. """
    return numpy.sqrt(numpy.sum(vec**2))


def _calc_vorticity(A, B, P):
    """ Calculates the influence coefficient for a vortex filament.

    Parameters
    ----------
    A[3] : array_like
        Coordinates for the start point of the filament.
    B[3] : array_like
        Coordinates for the end point of the filament.
    P[3] : array_like
        Coordinates for the collocation point where the influence coefficient
        is computed.

    Returns
    -------
    out[3] : array_like
        Influence coefficient contribution for the described filament.

    """

    r1 = P - A
    r2 = P - B

    r1_mag = norm(r1)
    r2_mag = norm(r2)

    return (r1_mag + r2_mag) * numpy.cross(r1, r2) / \
           (r1_mag * r2_mag * (r1_mag * r2_mag + r1.dot(r2)))


def _assemble_AIC_mtx(mtx, params, surfaces, skip=False):
    """
    Compute the aerodynamic influence coefficient matrix
    for either solving the linear system or solving for the drag.

    We use a nested for loop structure to loop through the lifting surfaces to
    obtain the corresponding mesh, then for each mesh we again loop through
    the lifting surfaces to obtain the collocation points used to compute
    the horseshoe vortex influence coefficients.

    This creates mtx with blocks corresponding to each lifting surface's
    effects on other lifting surfaces. The block diagonal portions
    correspond to each lifting surface's influencen on itself. For a single
    lifting surface, this is the entire mtx.

    Parameters
    ----------
    mtx[num_y-1, num_y-1, 3] : array_like
        Aerodynamic influence coefficient (AIC) matrix, or the
        derivative of v w.r.t. circulations.
    params : dictionary
        OpenMDAO params dictionary for a given aero problem
    surfaces : dictionary
        Dictionary containing all surfaces in an aero problem.
    skip : boolean
        If false, the bound vortex contributions on the collocation point
        corresponding to the same panel are not included. Used for the drag
        computation.

    Returns
    -------
    mtx[tot_panels, tot_panels, 3] : array_like
        Aerodynamic influence coefficient (AIC) matrix, or the
        derivative of v w.r.t. circulations.
    """

    alpha = params['alpha']
    mtx[:, :, :] = 0.0
    cosa = numpy.cos(alpha * numpy.pi / 180.)
    sina = numpy.sin(alpha * numpy.pi / 180.)
    u = numpy.array([cosa, 0, sina])

    i_ = 0
    i_bpts_ = 0
    i_panels_ = 0

    # Loop over the lifting surfaces to compute their influence on the flow
    # velocity at the collocation points
    for surface_ in surfaces:

        # Variable names with a trailing underscore correspond to the lifting
        # surface being examined, not the collocation point
        name_ = surface_['name']
        nx_ = surface_['num_x']
        ny_ = surface_['num_y']
        n_ = nx_ * ny_
        n_bpts_ = (nx_ - 1) * ny_
        n_panels_ = (nx_ - 1) * (ny_ - 1)

        # Obtain the lifting surface mesh in the form expected by the solver,
        # with shape [nx_, ny_, 3]
        mesh = params[name_+'def_mesh']
        bpts = params[name_+'b_pts']

        # Set counters to know where to index the sub-matrix within the full mtx
        i = 0
        i_bpts = 0
        i_panels = 0

        for surface in surfaces:
            # These variables correspond to the collocation points
            name = surface['name']
            nx = surface['num_x']
            ny = surface['num_y']
            n = nx * ny
            n_bpts = (nx - 1) * ny
            n_panels = (nx - 1) * (ny - 1)
            symmetry = surface['symmetry']

            # Obtain the collocation points used to compute the AIC mtx.
            # If setting up the AIC mtx, we use the collocation points (c_pts),
            # but if setting up the matrix to solve for drag, we use the
            # midpoints of the bound vortices.
            if skip:
                # Find the midpoints of the bound points, used in drag computations
                pts = (params[name+'b_pts'][:, 1:, :] + \
                    params[name+'b_pts'][:, :-1, :]) / 2
            else:
                pts = params[name+'c_pts']

            # Initialize sub-matrix to populate within full mtx
            small_mat = numpy.zeros((n_panels, n_panels_, 3), dtype='complex')

            # Dense fortran assembly for the AIC matrix
            if fortran_flag:
                small_mat[:, :, :] = OAS_API.oas_api.assembleaeromtx(alpha, pts, bpts,
                                                         mesh, skip, symmetry)
            # Python matrix assembly
            else:
                # Spanwise loop through horseshoe elements
                for el_j in xrange(ny_ - 1):
                    el_loc_j = el_j * (nx_ - 1)
                    C_te = mesh[-1, el_j + 1, :]
                    D_te = mesh[-1, el_j + 0, :]

                    # Mirror the horseshoe vortex points
                    if symmetry:
                        C_te_sym = C_te.copy()
                        D_te_sym = D_te.copy()
                        C_te_sym[1] = -C_te_sym[1]
                        D_te_sym[1] = -D_te_sym[1]

                    # Spanwise loop through control points
                    for cp_j in xrange(ny - 1):
                        cp_loc_j = cp_j * (nx - 1)

                        # Chordwise loop through control points
                        for cp_i in xrange(nx - 1):
                            cp_loc = cp_i + cp_loc_j

                            P = pts[cp_i, cp_j]

                            r1 = P - D_te
                            r2 = P - C_te

                            r1_mag = norm(r1)
                            r2_mag = norm(r2)

                            t1 = numpy.cross(u, r2) / \
                                (r2_mag * (r2_mag - u.dot(r2)))
                            t3 = numpy.cross(u, r1) / \
                                (r1_mag * (r1_mag - u.dot(r1)))

                            # AIC contribution from trailing vortex filaments
                            # coming off the trailing edge
                            trailing = t1 - t3

                            # Calculate the effects across the symmetry plane
                            if symmetry:
                                r1 = P - D_te_sym
                                r2 = P - C_te_sym

                                r1_mag = norm(r1)
                                r2_mag = norm(r2)

                                t1 = numpy.cross(u, r2) / \
                                    (r2_mag * (r2_mag - u.dot(r2)))
                                t3 = numpy.cross(u, r1) / \
                                    (r1_mag * (r1_mag - u.dot(r1)))

                                trailing += t3 - t1

                            edges = 0

                            # Chordwise loop through horseshoe elements in
                            # reversed order, starting with the panel closest
                            # to the leading edge. This is done to sum the
                            # AIC contributions from the side vortex filaments
                            # as we loop through the elements
                            for el_i in reversed(xrange(nx_ - 1)):
                                el_loc = el_i + el_loc_j

                                A = bpts[el_i, el_j + 0, :]
                                B = bpts[el_i, el_j + 1, :]

                                # Check if this is the last panel; if so, use
                                # the trailing edge mesh points for C & D, else
                                # use the directly aft panel's bound points
                                # for C & D
                                if el_i == nx_ - 2:
                                    C = mesh[-1, el_j + 1, :]
                                    D = mesh[-1, el_j + 0, :]
                                else:
                                    C = bpts[el_i + 1, el_j + 1, :]
                                    D = bpts[el_i + 1, el_j + 0, :]

                                # Calculate and store the contributions from
                                # the vortex filaments on the sides of the
                                # panels, adding as we progress through the
                                # panels
                                edges += _calc_vorticity(B, C, P)
                                edges += _calc_vorticity(D, A, P)

                                # Mirror the horseshoe vortex points and
                                # calculate the effects across
                                # the symmetry plane
                                if symmetry:
                                    A_sym = A.copy()
                                    B_sym = B.copy()
                                    C_sym = C.copy()
                                    D_sym = D.copy()
                                    A_sym[1] = -A_sym[1]
                                    B_sym[1] = -B_sym[1]
                                    C_sym[1] = -C_sym[1]
                                    D_sym[1] = -D_sym[1]

                                    edges += _calc_vorticity(C_sym, B_sym, P)
                                    edges += _calc_vorticity(A_sym, D_sym, P)

                                # If skip, do not include the contributions
                                # from the panel's bound vortex filament, as
                                # this causes a singularity when we're taking
                                # the influence of a panel on its own
                                # collocation point. This true for the drag
                                # computation and false for circulation
                                # computation, due to the different collocation
                                # points.
                                if skip and el_loc == cp_loc:
                                    if symmetry:
                                        bound = _calc_vorticity(B_sym, A_sym, P)
                                    else:
                                        bound = numpy.zeros((3))
                                    small_mat[cp_loc, el_loc, :] = \
                                        trailing + edges + bound
                                else:
                                    bound = _calc_vorticity(A, B, P)

                                    # Account for symmetry by including the
                                    # mirrored bound vortex
                                    if symmetry:
                                        bound += _calc_vorticity(B_sym, A_sym, P)

                                    small_mat[cp_loc, el_loc, :] = \
                                        trailing + edges + bound

            # Populate the full-size matrix with these surface-surface AICs
            mtx[i_panels:i_panels+n_panels,
                i_panels_:i_panels_+n_panels_, :] = small_mat

            i += n
            i_bpts += n_bpts
            i_panels += n_panels

        i_ += n_
        i_bpts_ += n_bpts_
        i_panels_ += n_panels_

    mtx /= 4 * numpy.pi

def _assemble_AIC_mtx_d(mtx, params, surfaces, skip=False):
    """
    Compute the aerodynamic influence coefficient matrix
    for either solving the linear system or solving for the drag.

    We use a nested for loop structure to loop through the lifting surfaces to
    obtain the corresponding mesh, then for each mesh we again loop through
    the lifting surfaces to obtain the collocation points used to compute
    the horseshoe vortex influence coefficients.

    This creates mtx with blocks corresponding to each lifting surface's
    effects on other lifting surfaces. The block diagonal portions
    correspond to each lifting surface's influencen on itself. For a single
    lifting surface, this is the entire mtx.

    Parameters
    ----------
    mtx[num_y-1, num_y-1, 3] : array_like
        Aerodynamic influence coefficient (AIC) matrix, or the
        derivative of v w.r.t. circulations.
    params : dictionary
        OpenMDAO params dictionary for a given aero problem
    surfaces : dictionary
        Dictionary containing all surfaces in an aero problem.
    skip : boolean
        If false, the bound vortex contributions on the collocation point
        corresponding to the same panel are not included. Used for the drag
        computation.

    Returns
    -------
    mtx[tot_panels, tot_panels, 3] : array_like
        Aerodynamic influence coefficient (AIC) matrix, or the
        derivative of v w.r.t. circulations.
    """

    alpha = params['alpha']
    mtx[:, :, :] = 0.0
    cosa = numpy.cos(alpha * numpy.pi / 180.)
    sina = numpy.sin(alpha * numpy.pi / 180.)
    u = numpy.array([cosa, 0, sina])

    i_ = 0
    i_bpts_ = 0
    i_panels_ = 0

    # Loop over the lifting surfaces to compute their influence on the flow
    # velocity at the collocation points
    for surface_ in surfaces:

        # Variable names with a trailing underscore correspond to the lifting
        # surface being examined, not the collocation point
        name_ = surface_['name']
        nx_ = surface_['num_x']
        ny_ = surface_['num_y']
        n_ = nx_ * ny_
        n_bpts_ = (nx_ - 1) * ny_
        n_panels_ = (nx_ - 1) * (ny_ - 1)

        # Obtain the lifting surface mesh in the form expected by the solver,
        # with shape [nx_, ny_, 3]
        mesh = params['def_mesh']
        bpts = params['b_pts']

        # Set counters to know where to index the sub-matrix within the full mtx
        i = 0
        i_bpts = 0
        i_panels = 0

        for surface in surfaces:
            # These variables correspond to the collocation points
            name = surface['name']
            nx = surface['num_x']
            ny = surface['num_y']
            n = nx * ny
            n_bpts = (nx - 1) * ny
            n_panels = (nx - 1) * (ny - 1)
            symmetry = surface['symmetry']

            # Obtain the collocation points used to compute the AIC mtx.
            # If setting up the AIC mtx, we use the collocation points (c_pts),
            # but if setting up the matrix to solve for drag, we use the
            # midpoints of the bound vortices.
            if skip:
                # Find the midpoints of the bound points, used in drag computations
                pts = (params['b_pts'][:, 1:, :] + \
                    params['b_pts'][:, :-1, :]) / 2
            else:
                pts = params['c_pts']

            # Initialize sub-matrix to populate within full mtx
            small_mat = numpy.zeros((n_panels, n_panels_, 3), dtype='complex')

            # Dense fortran assembly for the AIC matrix
            if fortran_flag:
                small_mat[:, :, :] = OAS_API.oas_api.assembleaeromtx(alpha, pts, bpts,
                                                         mesh, skip, symmetry)

            # Populate the full-size matrix with these surface-surface AICs
            mtx[i_panels:i_panels+n_panels,
                i_panels_:i_panels_+n_panels_, :] = small_mat

            i += n
            i_bpts += n_bpts
            i_panels += n_panels

        i_ += n_
        i_bpts_ += n_bpts_
        i_panels_ += n_panels_

    mtx /= 4 * numpy.pi


class VLMGeometry(Component):
    """ Compute various geometric properties for VLM analysis.

    Parameters
    ----------
    def_mesh[nx, ny, 3] : array_like
        Array defining the nodal coordinates of the lifting surface.

    Returns
    -------
    b_pts[nx-1, ny, 3] : array_like
        Bound points for the horseshoe vortices, found along the 1/4 chord.
    c_pts[nx-1, ny-1, 3] : array_like
        Collocation points on the 3/4 chord line where the flow tangency
        condition is satisfed. Used to set up the linear system.
    widths[nx-1, ny-1] : array_like
        The spanwise widths of each individual panel.
    normals[nx-1, ny-1, 3] : array_like
        The normal vector for each panel, computed as the cross of the two
        diagonals from the mesh points.
    S_ref : float
        The reference area of the lifting surface.

    """

    def __init__(self, surface):
        super(VLMGeometry, self).__init__()

        self.surface = surface

        self.ny = surface['num_y']
        self.nx = surface['num_x']
        self.n = self.nx * self.ny
        self.mesh = surface['mesh']
        name = surface['name']
        self.fem_origin = surface['fem_origin']

        self.add_param('def_mesh', val=numpy.zeros((self.nx, self.ny, 3),
                       dtype="complex"))
        self.add_output('b_pts', val=numpy.zeros((self.nx-1, self.ny, 3),
                        dtype="complex"))
        self.add_output('c_pts', val=numpy.zeros((self.nx-1, self.ny-1, 3)))
        self.add_output('widths', val=numpy.zeros((self.nx-1, self.ny-1)))
        self.add_output('normals', val=numpy.zeros((self.nx-1, self.ny-1, 3)))
        self.add_output('S_ref', val=0.)

    def _get_lengths(self, A, B, axis):
        return numpy.sqrt(numpy.sum((B - A)**2, axis=axis))

    def solve_nonlinear(self, params, unknowns, resids):
        name = self.surface['name']
        mesh = params['def_mesh']

        # Compute the bound points at 1/4 chord
        b_pts = mesh[:-1, :, :] * .75 + mesh[1:, :, :] * .25

        # Compute the collocation points at the midpoints of each
        # panel's 3/4 chord line
        c_pts = 0.5 * 0.25 * mesh[:-1, :-1, :] + \
                0.5 * 0.75 * mesh[1:, :-1, :] + \
                0.5 * 0.25 * mesh[:-1,  1:, :] + \
                0.5 * 0.75 * mesh[1:,  1:, :]

        # Compute the widths of each panel
        widths = self._get_lengths(b_pts[:, 1:, :], b_pts[:, :-1, :], 2)

        # Compute the normal of each panel by taking the cross-product of
        # its diagonals. Note that this could be a nonplanar surface
        normals = numpy.cross(
            mesh[:-1,  1:, :] - mesh[1:, :-1, :],
            mesh[:-1, :-1, :] - mesh[1:,  1:, :],
            axis=2)

        norms = numpy.sqrt(numpy.sum(normals**2, axis=2))

        for j in xrange(3):
            normals[:, :, j] /= norms

        # Store each array
        unknowns['b_pts'] = b_pts
        unknowns['c_pts'] = c_pts
        unknowns['widths'] = widths
        unknowns['normals'] = normals
        unknowns['S_ref'] = 0.5 * numpy.sum(norms)

    def linearize(self, params, unknowns, resids):
        """ Jacobian for geometry."""

        jac = self.alloc_jacobian()
        name = self.surface['name']

        fd_jac = self.complex_step_jacobian(params, unknowns, resids,
                                            fd_params=['def_mesh'],
                                            fd_unknowns=['widths', 'normals',
                                                         'S_ref'],
                                            fd_states=[])
        jac.update(fd_jac)

        nx = self.surface['num_x']
        ny = self.surface['num_y']

        for iz, v in zip((0, ny*3), (.75, .25)):
            numpy.fill_diagonal(jac['b_pts', 'def_mesh'][:, iz:], v)


        for iz, v in zip((0, 3, ny*3, (ny+1)*3),
                         (.125, .125, .375, .375)):
            for ix in range(nx-1):
                numpy.fill_diagonal(jac['c_pts', 'def_mesh']
                    [(ix*(ny-1))*3:((ix+1)*(ny-1))*3, iz+ix*ny*3:], v)

        return jac


class VLMCirculations(Component):
    """
    Compute the circulations based on the AIC matrix and the panel velocities.
    Note that the flow tangency condition is enforced at the 3/4 chord point.

    Parameters
    ----------
    def_mesh[nx, ny, 3] : array_like
        Array defining the nodal coordinates of the lifting surface.
    b_pts[nx-1, ny, 3] : array_like
        Bound points for the horseshoe vortices, found along the 1/4 chord.
    c_pts[nx-1, ny-1, 3] : array_like
        Collocation points on the 3/4 chord line where the flow tangency
        condition is satisfed. Used to set up the linear system.
    normals[nx-1, ny-1, 3] : array_like
        The normal vector for each panel, computed as the cross of the two
        diagonals from the mesh points.
    v : float
        Freestream air velocity in m/s.
    alpha : float
        Angle of attack in degrees.

    Returns
    -------
    circulations : array_like
        Flattened vector of horseshoe vortex strengths calculated by solving
        the linear system of AIC_mtx * circulations = rhs, where rhs is
        based on the air velocity at each collocation point.

    """

    def __init__(self, surfaces, prob_dict):
        super(VLMCirculations, self).__init__()

        self.surfaces = surfaces

        for surface in surfaces:
            self.surface = surface
            self.ny = surface['num_y']
            self.nx = surface['num_x']
            self.n = self.nx * self.ny
            self.mesh = surface['mesh']
            name = surface['name']

            self.add_param(name+'def_mesh', val=numpy.zeros((self.nx, self.ny, 3),
                           dtype="complex"))
            self.add_param(name+'b_pts', val=numpy.zeros((self.nx-1, self.ny, 3),
                           dtype="complex"))
            self.add_param(name+'c_pts', val=numpy.zeros((self.nx-1, self.ny-1, 3),
                           dtype="complex"))
            self.add_param(name+'normals', val=numpy.zeros((self.nx-1, self.ny-1, 3)))

        self.add_param('v', val=prob_dict['v'])
        self.add_param('alpha', val=prob_dict['alpha'])

        self.deriv_options['linearize'] = True  # only for circulations

        tot_panels = 0
        for surface in surfaces:
            tot_panels += (surface['num_x'] - 1) * (surface['num_y'] - 1)
        self.tot_panels = tot_panels

        self.add_state('circulations', val=numpy.zeros((tot_panels),
                       dtype="complex"))

        self.AIC_mtx = numpy.zeros((tot_panels, tot_panels, 3),
                                   dtype="complex")
        self.mtx = numpy.zeros((tot_panels, tot_panels), dtype="complex")
        self.rhs = numpy.zeros((tot_panels), dtype="complex")

    def _assemble_system(self, params):

        # Actually assemble the AIC matrix
        _assemble_AIC_mtx(self.AIC_mtx, params, self.surfaces)

        # Construct an flattend array with the normals of each surface in order
        # so we can do the normals with velocities to set up the right-hand-side
        # of the system.
        flattened_normals = numpy.zeros((self.tot_panels, 3), dtype='complex')
        i = 0
        for surface in self.surfaces:
            name = surface['name']
            num_panels = (surface['num_x'] - 1) * (surface['num_y'] - 1)
            flattened_normals[i:i+num_panels, :] = params[name+'normals'].reshape(-1, 3, order='F')
            i += num_panels

        # Construct a matrix that is the AIC_mtx dotted by the normals at each
        # collocation point. This is used to compute the circulations
        self.mtx[:, :] = 0.
        for ind in xrange(3):
            self.mtx[:, :] += (self.AIC_mtx[:, :, ind].T *
                flattened_normals[:, ind]).T

        # Obtain the freestream velocity direction and magnitude by taking
        # alpha into account
        alpha = params['alpha'] * numpy.pi / 180.
        cosa = numpy.cos(alpha)
        sina = numpy.sin(alpha)
        v_inf = params['v'] * numpy.array([cosa, 0., sina], dtype="complex")

        # Populate the right-hand side of the linear system with the
        # expected velocities at each collocation point
        self.rhs[:] = -flattened_normals.\
            reshape(-1, flattened_normals.shape[-1], order='F').dot(v_inf)

    def solve_nonlinear(self, params, unknowns, resids):
        """ Solve the linear system to obtain circulations. """
        name = self.surface['name']
        self._assemble_system(params)
        unknowns['circulations'] = numpy.linalg.solve(self.mtx, self.rhs)

    def apply_nonlinear(self, params, unknowns, resids):
        """ Compute the residuals of the linear system. """
        name = self.surface['name']
        self._assemble_system(params)

        circ = unknowns['circulations']
        resids['circulations'] = self.mtx.dot(circ) - self.rhs

    def linearize(self, params, unknowns, resids):
        """ Jacobian for circulations."""
        name = self.surface['name']
        self.lup = lu_factor(self.mtx.real)
        jac = self.alloc_jacobian()

        for surface in self.surfaces:
            name = surface['name']

            fd_jac = self.complex_step_jacobian(params, unknowns, resids,
                                             fd_params=[name+'normals', 'alpha',
                                                        name+'def_mesh', name+'b_pts',
                                                        name+'c_pts'],
                                             fd_states=[])
            jac.update(fd_jac)

        jac['circulations', 'circulations'] = self.mtx.real

        jac['circulations', 'v'][:, 0] = -self.rhs.real / params['v'].real

        return jac

    def solve_linear(self, dumat, drmat, vois, mode=None):

        if mode == 'fwd':
            sol_vec, rhs_vec = self.dumat, self.drmat
            t = 0
        else:
            sol_vec, rhs_vec = self.drmat, self.dumat
            t = 1

        for voi in vois:
            sol_vec[voi].vec[:] = lu_solve(self.lup, rhs_vec[voi].vec, trans=t)


class VLMForces(Component):
    """ Compute aerodynamic forces acting on each section.

    Note that some of the parameters and unknowns has the surface name
    prepended on it. E.g., 'def_mesh' on a surface called 'wing' would be
    'wing_def_mesh', etc.

    Parameters
    ----------
    def_mesh[nx, ny, 3] : array_like
        Array defining the nodal coordinates of the lifting surface.
    b_pts[nx-1, ny, 3] : array_like
        Bound points for the horseshoe vortices, found along the 1/4 chord.
    circulations : array_like
        Flattened vector of horseshoe vortex strengths calculated by solving
        the linear system of AIC_mtx * circulations = rhs, where rhs is
        based on the air velocity at each collocation point.
    alpha : float
        Angle of attack in degrees.
    v : float
        Freestream air velocity in m/s.
    rho : float
        Air density in kg/m^3.

    Returns
    -------
    sec_forces[nx-1, ny-1, 3] : array_like
        Flattened array containing the sectional forces acting on each panel.
        Stored in Fortran order (only relevant when more than one chordwise
        panel).

    """

    def __init__(self, surfaces, prob_dict):
        super(VLMForces, self).__init__()

        tot_panels = 0
        for surface in surfaces:
            name = surface['name']
            tot_panels += (surface['num_x'] - 1) * (surface['num_y'] - 1)
            self.ny = surface['num_y']
            self.nx = surface['num_x']

            self.add_param(name+'def_mesh', val=numpy.zeros((self.nx, self.ny, 3), dtype='complex'))
            self.add_param(name+'b_pts', val=numpy.zeros((self.nx-1, self.ny, 3), dtype='complex'))
            self.add_output(name+'sec_forces', val=numpy.zeros((self.nx-1, self.ny-1, 3), dtype='complex'))

        self.tot_panels = tot_panels
        self.add_param('circulations', val=numpy.zeros((tot_panels)))
        self.add_param('alpha', val=3.)
        self.add_param('v', val=10.)
        self.add_param('rho', val=3.)
        self.surfaces = surfaces

        self.mtx = numpy.zeros((tot_panels, tot_panels, 3), dtype="complex")
        self.v = numpy.zeros((tot_panels, 3), dtype="complex")

        # self.deriv_options['type'] = 'fd'
        # self.deriv_options['form'] = 'central'

    def solve_nonlinear(self, params, unknowns, resids):
        circ = params['circulations']
        alpha = params['alpha'] * numpy.pi / 180.
        cosa = numpy.cos(alpha)
        sina = numpy.sin(alpha)

        # Assemble a different matrix here than the AIC_mtx from above; Note
        # that the collocation points used here are the midpoints of each
        # bound vortex filament, not the collocation points from above
        _assemble_AIC_mtx(self.mtx, params, self.surfaces, skip=True)

        # Compute the induced velocities at the midpoints of the
        # bound vortex filaments
        for ind in xrange(3):
            self.v[:, ind] = self.mtx[:, :, ind].dot(circ)

        # Add the freestream velocity to the induced velocity so that
        # self.v is the total velocity seen at the point
        self.v[:, 0] += cosa * params['v']
        self.v[:, 2] += sina * params['v']

        i = 0
        for surface in self.surfaces:
            name = surface['name']
            num_panels = (surface['num_x'] - 1) * (surface['num_y'] - 1)

            b_pts = params[name+'b_pts']

            bound = b_pts[:, 1:, :] - b_pts[:, :-1, :]

            # Cross the obtained velocities with the bound vortex filament
            # vectors
            cross = numpy.cross(self.v[i:i+num_panels],
                                bound.reshape(-1, bound.shape[-1], order='F'))

            sec_forces = numpy.zeros(((surface['num_x']-1)*(surface['num_y']-1), 3), dtype='complex')
            # Compute the sectional forces acting on each panel
            for ind in xrange(3):
                sec_forces[:, ind] = \
                    (params['rho'] * circ[i:i+num_panels] * cross[:, ind])
            unknowns[name+'sec_forces'] = sec_forces.reshape((surface['num_x']-1, surface['num_y']-1, 3), order='F')

            i += num_panels

    def linearize(self, params, unknowns, resids):
        """ Jacobian for forces."""

        jac = self.alloc_jacobian()

        fd_jac = self.complex_step_jacobian(params, unknowns, resids,
                                         fd_params=['alpha', 'circulations', 'v'],
                                         fd_states=[])
        jac.update(fd_jac)

        rho = params['rho'].real

        for surface in self.surfaces:
            name = surface['name']

            fd_jac = self.complex_step_jacobian(params, unknowns, resids,
                                             fd_params=[name+'b_pts',
                                                name+'def_mesh'],
                                             fd_states=[])
            jac.update(fd_jac)

            sec_forces = unknowns[name+'sec_forces'].real
            jac[name+'sec_forces', 'rho'] = sec_forces.flatten() / rho

        return jac

    # def apply_linear(self, params, unknowns, dparams, dunknowns, dresids, mode):
    #     print mode, dresids['sec_forces']


class VLMLiftDrag(Component):
    """
    Calculate total lift and drag in force units based on section forces.

    If the simulation is given a non-zero Reynolds number, it is used to
    compute the skin friction drag. If Reynolds number == 0, then there is
    no skin friction drag dependence. Currently, the Reynolds number
    must be set by the user in the run script and used as in IndepVarComp.

    Parameters
    ----------
    sec_forces[nx-1, ny-1, 3] : array_like
        Flattened array containing the sectional forces acting on each panel.
        Stored in Fortran order (only relevant when more than one chordwise
        panel).
    alpha : float
        Angle of attack in degrees.
    Re : float
        Reynolds number.
    M : float
        Mach number.
    v : float
        Freestream air velocity in m/s.
    rho : float
        Air density in kg/m^3.
    S_ref : float
        The reference area of the lifting surface.

    Returns
    -------
    L : float
        Total lift force for the lifting surface.
    D : float
        Total drag force for the lifting surface.

    """

    def __init__(self, surface):
        super(VLMLiftDrag, self).__init__()

        self.surface = surface
        self.ny = surface['num_y']
        self.nx = surface['num_x']
        self.n = self.nx * self.ny
        self.mesh = surface['mesh']
        name = surface['name']

        self.add_param('sec_forces', val=numpy.zeros((self.nx - 1, self.ny - 1, 3)))
        self.add_param('alpha', val=3.)
        self.add_param('Re', val=5.e6)
        self.add_param('M', val=.84)
        self.add_param('v', val=10.)
        self.add_param('rho', val=3.)
        self.add_param('S_ref', val=0.)
        self.add_output('L', val=0.)
        self.add_output('D', val=0.)

        self.deriv_options['type'] = 'cs'
        self.deriv_options['form'] = 'central'

    def solve_nonlinear(self, params, unknowns, resids):
        name = self.surface['name']
        alpha = params['alpha'] * numpy.pi / 180.
        forces = params['sec_forces'].reshape(-1, 3)
        cosa = numpy.cos(alpha)
        sina = numpy.sin(alpha)

        Re = params['Re']
        M = params['M']
        v = params['v']
        rho = params['rho']
        S_ref = params['S_ref']

        # Compute the skin friction coefficient
        # Use eq. 12.27 of Raymer for turbulent Cf
        # Avoid divide by zero warning if Re == 0
        if Re == 0:
            Cf = 0.
        else:
            Cf = 0.455 / (numpy.log10(Re)**2.58 * (1 + .144 * M**2)**.65)

        # Compute the induced lift force on each lifting surface
        unknowns['L'] = \
            numpy.sum(-forces[:, 0] * sina + forces[:, 2] * cosa)

        # Compute the induced drag force on each lifting surface
        unknowns['D'] = \
            numpy.sum( forces[:, 0] * cosa + forces[:, 2] * sina)

        # Compute the drag contribution from skin friction
        D_f = Cf * rho * v**2 / 2. * S_ref
        unknowns['D'] += D_f

        if self.surface['symmetry']:
            unknowns['D'] *= 2
            unknowns['L'] *= 2

    def linearize(self, params, unknowns, resids):
        """ Jacobian for lift and drag."""

        jac = self.alloc_jacobian()
        name = self.surface['name']

        # Analytic derivatives for sec_forces
        alpha = params['alpha'] * numpy.pi / 180.
        cosa = numpy.cos(alpha)
        sina = numpy.sin(alpha)

        forces = params['sec_forces']

        tmp = numpy.array([-sina, 0, cosa])
        panel_start = i_panels*3
        panel_end = i_panels*3+n_panels*3
        jac['L', 'sec_forces'][i_surf, panel_start:panel_end] = \
            numpy.atleast_2d(numpy.tile(tmp, n_panels))
        tmp = numpy.array([cosa, 0, sina])
        jac['D', 'sec_forces'][i_surf, panel_start:panel_end] = \
            numpy.atleast_2d(numpy.tile(tmp, n_panels))

        p180 = numpy.pi / 180.
        jac['L', 'alpha'][i_surf] = p180 * \
            numpy.sum(-forces[:, :, 0] * cosa - forces[:, :, 2] * sina)
        jac['D', 'alpha'][i_surf] = p180 * \
            numpy.sum(-forces[:, :, 0] * sina + forces[:, :, 2] * cosa)

        return jac


class VLMCoeffs(Component):
    """ Compute lift and drag coefficients.

    Parameters
    ----------
    S_ref : float
        The reference areas of the lifting surface.
    L : float
        Total lift for the lifting surface.
    D : float
        Total drag for the lifting surface.
    v : float
        Freestream air velocity in m/s.
    rho : float
        Air density in kg/m^3.

    Returns
    -------
    CL1 : float
        Induced coefficient of lift (CL) for the lifting surface.
    CDi : float
        Induced coefficient of drag (CD) for the lifting surface.

    """


    def __init__(self, surface):
        super(VLMCoeffs, self).__init__()

        self.surface = surface
        self.ny = surface['num_y']
        self.nx = surface['num_x']
        self.n = self.nx * self.ny
        self.mesh = surface['mesh']
        name = surface['name']

        self.add_param('S_ref', val=0.)
        self.add_param('L', val=0.)
        self.add_param('D', val=0.)
        self.add_param('v', val=0.)
        self.add_param('rho', val=0.)
        self.add_output('CL1', val=0.)
        self.add_output('CDi', val=0.)

        self.deriv_options['type'] = 'cs'
        self.deriv_options['form'] = 'central'

    def solve_nonlinear(self, params, unknowns, resids):
        name = self.surface['name']
        S_ref = params['S_ref']
        rho = params['rho']
        v = params['v']
        L = params['L']
        D = params['D']

        if self.surface['symmetry']:
            S_ref *= 2

        unknowns['CL1'] = L / (0.5 * rho * v**2 * S_ref)
        unknowns['CDi'] = D / (0.5 * rho * v**2 * S_ref)

class TotalLift(Component):
    """ Calculate total lift in force units.

    Parameters
    ----------
    CL1 : array_like
        Induced coefficient of lift (CL) for each lifting surface.

    Returns
    -------
    CL : array_like
        Total coefficient of lift (CL) for each lifting surface.
    CL_wing : float
        CL of the main wing, used for CL constrained optimization.

    """

    def __init__(self, surface):
        super(TotalLift, self).__init__()

        self.surface = surface
        self.ny = surface['num_y']
        self.nx = surface['num_x']
        self.n = self.nx * self.ny
        self.mesh = surface['mesh']
        name = surface['name']

        self.add_param('CL1', val=0.)
        self.add_output('CL', val=0.)
        self.CL0 = surface['CL0']

        self.deriv_options['type'] = 'cs'
        self.deriv_options['form'] = 'central'

    def solve_nonlinear(self, params, unknowns, resids):
        name = self.surface['name']
        unknowns['CL'] = params['CL1'] + self.CL0

    def linearize(self, params, unknowns, resids):
        jac = self.alloc_jacobian()
        jac['CL', 'CL1'][:] = numpy.eye(self.num_surf)
        return jac


class TotalDrag(Component):
    """ Calculate total drag in force units.

    Parameters
    ----------
    CDi : float
        Induced coefficient of drag (CD) the each lifting surface.

    Returns
    -------
    CD : float
        Total coefficient of drag (CD) the each lifting surface.
    CD_wing : float
        CD of the main wing, used for CD minimization.

    """

    def __init__(self, surface):
        super(TotalDrag, self).__init__()

        self.surface = surface
        self.ny = surface['num_y']
        self.nx = surface['num_x']
        self.n = self.nx * self.ny
        self.mesh = surface['mesh']
        name = surface['name']

        self.add_param('CDi', val=0.)
        self.add_output('CD', val=0.)

        self.CD0 = surface['CD0']

        self.deriv_options['type'] = 'cs'
        self.deriv_options['form'] = 'central'

    def solve_nonlinear(self, params, unknowns, resids):
        name = self.surface['name']
        unknowns['CD'] = params['CDi'] + self.CD0

    def linearize(self, params, unknowns, resids):
        jac = self.alloc_jacobian()
        jac['CD', 'CDi'][:] = numpy.eye(self.num_surf)
        return jac


class VLMStates(Group):
    """ Group that contains the aerodynamic states. """

    def __init__(self, surfaces, prob_dict):
        super(VLMStates, self).__init__()

        self.add('circulations',
                 VLMCirculations(surfaces, prob_dict),
                 promotes=['*'])
        self.add('forces',
                 VLMForces(surfaces, prob_dict),
                 promotes=['*'])



class VLMFunctionals(Group):
    """ Group that contains the aerodynamic functionals used to evaluate
    performance. """

    def __init__(self, surface):
        super(VLMFunctionals, self).__init__()

        self.add('liftdrag',
                 VLMLiftDrag(surface),
                 promotes=['*'])
        self.add('coeffs',
                 VLMCoeffs(surface),
                 promotes=['*'])
        self.add('CL',
                 TotalLift(surface),
                 promotes=['*'])
        self.add('CD',
                 TotalDrag(surface),
                 promotes=['*'])
