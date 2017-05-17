"""
Define the aerodynamic analysis component using a vortex lattice method.

We input a nodal mesh and properties of the airflow to calculate the
circulations of the horseshoe vortices. We then compute the forces, lift,
and drag acting on the lifting surfaces. Currently we can compute the induced
and viscous drag.

"""

from __future__ import division, print_function
import numpy as np

from openmdao.api import Component, Group
from scipy.linalg import lu_factor, lu_solve

try:
    import OAS_API
    fortran_flag = True
    data_type = float
except:
    fortran_flag = False
    data_type = complex


def norm(vec):
    """ Finds the 2-norm of a vector. """
    return np.sqrt(np.sum(vec**2))


def _calc_vorticity(A, B, P):
    """ Calculates the influence coefficient for a vortex filament.

    Parameters
    ----------
    A[3] : numpy array
        Coordinates for the start point of the filament.
    B[3] : numpy array
        Coordinates for the end point of the filament.
    P[3] : numpy array
        Coordinates for the collocation point where the influence coefficient
        is computed.

    Returns
    -------
    out[3] : numpy array
        Influence coefficient contribution for the described filament.

    """

    r1 = P - A
    r2 = P - B

    r1_mag = norm(r1)
    r2_mag = norm(r2)

    return (r1_mag + r2_mag) * np.cross(r1, r2) / \
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
    mtx[(nx-1)*(ny-1), (nx-1)*(ny-1), 3] : numpy array
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
    mtx[tot_panels, tot_panels, 3] : numpy array
        Aerodynamic influence coefficient (AIC) matrix, or the
        derivative of v w.r.t. circulations.
    """

    # TODO: check if the float is needed; remove hopefully
    alpha = float(params['alpha'])
    mtx[:, :, :] = 0.0
    cosa = np.cos(alpha * np.pi / 180.)
    sina = np.sin(alpha * np.pi / 180.)
    u = np.array([cosa, 0, sina])

    i_ = 0
    i_panels_ = 0

    # Loop over the lifting surfaces to compute their influence on the flow
    # velocity at the collocation points
    for surface_ in surfaces:

        # Variable names with a trailing underscore correspond to the lifting
        # surface being examined, not the collocation point
        name_ = surface_['name']
        nx_ = surface_['num_x']
        ny_ = surface_['num_y']
        n_panels_ = (nx_ - 1) * (ny_ - 1)

        # Obtain the lifting surface mesh in the form expected by the solver,
        # with shape [nx_, ny_, 3]
        mesh = params[name_+'def_mesh']
        bpts = params[name_+'b_pts']

        # Set a counter to know where to index the sub-matrix within the full mtx
        i_panels = 0

        for surface in surfaces:
            # These variables correspond to the collocation points
            name = surface['name']
            nx = surface['num_x']
            ny = surface['num_y']
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
            small_mat = np.zeros((n_panels, n_panels_, 3), dtype=data_type)

            # Dense fortran assembly for the AIC matrix
            if fortran_flag:
                small_mat[:, :, :] = OAS_API.oas_api.assembleaeromtx(alpha, pts, bpts,
                                                         mesh, skip, symmetry)
            # Python matrix assembly
            else:
                # Spanwise loop through horseshoe elements
                for el_j in range(ny_ - 1):
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
                    for cp_j in range(ny - 1):
                        cp_loc_j = cp_j * (nx - 1)

                        # Chordwise loop through control points
                        for cp_i in range(nx - 1):
                            cp_loc = cp_i + cp_loc_j

                            P = pts[cp_i, cp_j]

                            r1 = P - D_te
                            r2 = P - C_te

                            r1_mag = norm(r1)
                            r2_mag = norm(r2)

                            t1 = np.cross(u, r2) / \
                                (r2_mag * (r2_mag - u.dot(r2)))
                            t3 = np.cross(u, r1) / \
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

                                t1 = np.cross(u, r2) / \
                                    (r2_mag * (r2_mag - u.dot(r2)))
                                t3 = np.cross(u, r1) / \
                                    (r1_mag * (r1_mag - u.dot(r1)))

                                trailing += t3 - t1

                            edges = 0

                            # Chordwise loop through horseshoe elements in
                            # reversed order, starting with the panel closest
                            # to the leading edge. This is done to sum the
                            # AIC contributions from the side vortex filaments
                            # as we loop through the elements
                            for el_i in reversed(range(nx_ - 1)):
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
                                        bound = np.zeros((3))
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

            i_panels += n_panels

        i_panels_ += n_panels_

    mtx /= 4 * np.pi

def _assemble_AIC_mtx_d(mtxd, params, dparams, dunknowns, surfaces, skip=False):
    """
    Differentiated code to get the forward mode seeds for the AIC matrix assembly.
    """

    alpha = params['alpha']
    alphad = dparams['alpha']

    i_panels_ = 0

    # Loop over the lifting surfaces to compute their influence on the flow
    # velocity at the collocation points
    for surface_ in surfaces:

        # Variable names with a trailing underscore correspond to the lifting
        # surface being examined, not the collocation point
        name_ = surface_['name']
        nx_ = surface_['num_x']
        ny_ = surface_['num_y']
        n_panels_ = (nx_ - 1) * (ny_ - 1)

        # Obtain the lifting surface mesh in the form expected by the solver,
        # with shape [nx_, ny_, 3]
        mesh = params[name_+'def_mesh']
        bpts = params[name_+'b_pts']

        meshd = dparams[name_+'def_mesh']
        bptsd = dparams[name_+'b_pts']

        # Set a counter to know where to index the sub-matrix within the full mtx
        i_panels = 0

        for surface in surfaces:
            # These variables correspond to the collocation points
            name = surface['name']
            nx = surface['num_x']
            ny = surface['num_y']
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
                ptsd = (dparams[name+'b_pts'][:, 1:, :] + \
                    dparams[name+'b_pts'][:, :-1, :]) / 2
            else:
                pts = params[name+'c_pts']
                ptsd = dparams[name+'c_pts']

            _, small_mat = OAS_API.oas_api.assembleaeromtx_d(alpha, alphad, pts, ptsd,
                                                          bpts, bptsd, mesh, meshd,
                                                          skip, symmetry)

            # Populate the full-size matrix with these surface-surface AICs
            mtxd[i_panels:i_panels+n_panels,
                 i_panels_:i_panels_+n_panels_, :] = small_mat

            i_panels += n_panels

        i_panels_ += n_panels_

    mtxd /= 4 * np.pi

def _assemble_AIC_mtx_b(mtxb, params, dparams, dunknowns, surfaces, skip=False):
    """
    Differentiated code to get the reverse mode seeds for the AIC matrix assembly.
    """

    alpha = params['alpha']

    mtxb /= 4 * np.pi

    i_panels_ = 0

    # Loop over the lifting surfaces to compute their influence on the flow
    # velocity at the collocation points
    for surface_ in surfaces:

        # Variable names with a trailing underscore correspond to the lifting
        # surface being examined, not the collocation point
        name_ = surface_['name']
        nx_ = surface_['num_x']
        ny_ = surface_['num_y']
        n_panels_ = (nx_ - 1) * (ny_ - 1)

        # Obtain the lifting surface mesh in the form expected by the solver,
        # with shape [nx_, ny_, 3]
        mesh = params[name_+'def_mesh']
        bpts = params[name_+'b_pts']

        # Set a counter to know where to index the sub-matrix within the full mtx
        i_panels = 0

        for surface in surfaces:
            # These variables correspond to the collocation points
            name = surface['name']
            nx = surface['num_x']
            ny = surface['num_y']
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

            small_mtxb = mtxb[i_panels:i_panels+n_panels, i_panels_:i_panels_+n_panels_, :]

            alphab, ptsb, bptsb, meshb, mtx = OAS_API.oas_api.assembleaeromtx_b(alpha, pts, bpts,
                                                     mesh, skip, symmetry, small_mtxb)

            dparams[name_+'def_mesh'] += meshb.real
            dparams[name_+'b_pts'] += bptsb.real

            if skip:
                dparams[name+'b_pts'][:, 1:, :] += ptsb.real / 2
                dparams[name+'b_pts'][:, :-1, :] += ptsb.real / 2
            else:
                dparams[name+'c_pts'] += ptsb.real
            dparams['alpha'] += alphab

            i_panels += n_panels

        i_panels_ += n_panels_
